import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score,
                             accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset, DataLoader

# 设置随机种子，确保可重复性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 超参数设置
EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
HIDDEN_LAYERS = [128, 64]
WEIGHT_DECAY = 5e-4
DROPOUT_RATE = 0.2
DATA_PATH = 'emergency_hospital_data.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_P0(lambd, mu, s):
    if s * mu <= lambd:
        return None
    rho = lambd / (s * mu)
    sum_part = 0.0
    for k in range(s):
        sum_part += ((lambd / mu) ** k) / math.factorial(k)
    term = ((lambd / mu) ** s) / (math.factorial(s) * (1 - rho))
    return 1 / (sum_part + term)


def compute_metrics(lambd, mu, s):
    """
    计算排队指标
    返回: (Wq: 排队等待时间, W_total: 总等待时间)
    """
    P0 = compute_P0(lambd, mu, s)
    if P0 is None:
        return None, None
    rho = lambd / (s * mu)
    numerator = P0 * ((lambd / mu) ** s) * mu
    denominator = math.factorial(s) * (s * mu - lambd) * (1 - rho)
    Wq = numerator / denominator
    W_total = Wq + 1 / mu
    return Wq, W_total


class ClinicalPAN(nn.Module):
    """
    约束敏感的临床金字塔注意力网络
    核心创新：将Tmax/Smax等医疗约束直接嵌入注意力机制
    """
    def __init__(self, input_dim):
        super(ClinicalPAN, self).__init__()
        self.input_dim = input_dim

        # 多分支特征提取器（替代卷积，使用全连接层）
        self.branch_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(3)  # 3个分支对应不同的时间粒度
        ])

        # 约束注意力生成器（核心创新）
        # 输入包含：λ, μ_nurse, μ_doctor, Tmax, Smax等约束信息
        self.constraint_attn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # 输出32维约束敏感权重
            nn.Sigmoid()  # 输出[0,1]权重，约束越紧权重越高
        )

        # 约束强度感知器（动态调节注意力强度）
        self.constraint_intensity = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(), 
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出约束紧迫度[0,1]
        )

        # 动态特征融合
        self.fusion = nn.Linear(32 * 3, input_dim)  # 3个分支特征拼接后映射回原维度
        self.dropout = nn.Dropout(0.1)

        # 残差连接的权重调节
        self.residual_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        约束引导的前向传播
        数学表达：α = σ(Wc[λ,μ,Tmax,Smax] + bc)
        """
        batch_size = x.size(0)
        input_dim = x.size(1)
        
        # 生成约束敏感权重 [B, 32]
        constraint_weights = self.constraint_attn(x)  # [B, 32]
        constraint_intensity = self.constraint_intensity(x)  # [B, 1]
        
        # 多分支特征提取（使用可学习的全连接层）
        branch_features = []
        for i, branch_transform in enumerate(self.branch_transforms):
            # 通过可学习的变换提取特征 [B, input_dim] -> [B, 32]
            feat = branch_transform(x)  # [B, 32]
            
            # 应用约束敏感权重
            weighted_feat = feat * constraint_weights  # [B, 32]
            
            # 根据约束紧迫度动态调节特征强度
            intensity_scaled = weighted_feat * constraint_intensity  # [B, 32]
            
            branch_features.append(intensity_scaled)
        
        # 多分支特征融合 [B, 32*3] = [B, 96]
        fused_features = torch.cat(branch_features, dim=1)  # [B, 96]
        
        # 确保融合层输出正确维度 [B, 96] -> [B, input_dim]
        fused_output = self.fusion(fused_features)  # [B, input_dim]
        
        # 约束感知的残差连接
        residual_weight = self.residual_gate(x)  # [B, input_dim]
        output = fused_output + residual_weight * x  # [B, input_dim]
        
        return self.dropout(output)


# 保持向后兼容的PANLayer别名
PANLayer = ClinicalPAN


class HospitalPANDNNModel(nn.Module):
    """PAN+DNN混合架构模型"""
    def __init__(self, input_dim, hidden_layers, n_nurse_classes, n_doctor_classes):
        super().__init__()
        self.input_dim = input_dim

        # PAN特征提取层
        self.pan_layers = nn.ModuleList([
            PANLayer(input_dim),
            PANLayer(input_dim),
            PANLayer(input_dim)
        ])

        # DNN主干网络
        layers = []
        prev_size = input_dim
        for i, layer_size in enumerate(hidden_layers):
            layers += [
                nn.Linear(prev_size, layer_size),
                nn.BatchNorm1d(layer_size),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE)
            ]
            self._init_weights(layers[-4])
            prev_size = layer_size
        self.shared_layers = nn.Sequential(*layers)

        # 任务特定头部
        self.nurse_head = self._create_head(prev_size, n_nurse_classes)
        self.doctor_head = self._create_head(prev_size, n_doctor_classes)

        # 等待时间预测头（用于损失函数）
        self.wait_time_head = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 注意力模块（第三阶段启用）
        self.attention_enabled = False
        self.global_attention = nn.MultiheadAttention(
            embed_dim=prev_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    def _create_head(self, in_features, out_features):
        head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_features)
        )
        self._init_weights(head[0])
        self._init_weights(head[3])
        return head
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def enable_attention(self):
        """启用注意力模块（第三阶段）"""
        self.attention_enabled = True

    def freeze_fc_layers(self):
        """冻结全连接层（第一阶段）"""
        for param in self.shared_layers.parameters():
            param.requires_grad = False
        for param in self.nurse_head.parameters():
            param.requires_grad = False
        for param in self.doctor_head.parameters():
            param.requires_grad = False
        for param in self.wait_time_head.parameters():
            param.requires_grad = False

    def unfreeze_last_layers(self):
        """解冻最后三层（第二阶段）"""
        # 解冻最后三个shared_layers
        layers_list = list(self.shared_layers.children())
        for layer in layers_list[-12:]:  # 最后3个block，每个block 4层
            for param in layer.parameters():
                param.requires_grad = True

        # 解冻任务头
        for param in self.nurse_head.parameters():
            param.requires_grad = True
        for param in self.doctor_head.parameters():
            param.requires_grad = True
        for param in self.wait_time_head.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """解冻所有层（第三阶段）"""
        for param in self.parameters():
            param.requires_grad = True
    def forward(self, x):
        s_nurse_max = x[:, -4].long()
        s_doctor_max = x[:, -3].long()

        # PAN特征提取
        pan_out = x
        for pan_layer in self.pan_layers:
            pan_out = pan_layer(pan_out)

        # DNN主干网络
        shared_out = self.shared_layers(pan_out)

        # 全局注意力（第三阶段）
        if self.attention_enabled:
            shared_out_expanded = shared_out.unsqueeze(1)  # [batch, 1, features]
            attended_out, _ = self.global_attention(
                shared_out_expanded, shared_out_expanded, shared_out_expanded
            )
            shared_out = attended_out.squeeze(1) + shared_out  # 残差连接

        # 任务预测
        nurse_logits = self.nurse_head(shared_out)
        doctor_logits = self.doctor_head(shared_out)
        wait_time_pred = self.wait_time_head(shared_out)

        # 应用约束掩码
        batch_size = x.size(0)
        device = x.device
        nurse_mask = torch.zeros_like(nurse_logits, device=device)
        for i in range(batch_size):
            nurse_mask[i, :s_nurse_max[i] + 1] = 1
        nurse_logits = nurse_logits * nurse_mask + (1 - nurse_mask) * -1e9
        doctor_mask = torch.zeros_like(doctor_logits, device=device)
        for i in range(batch_size):
            doctor_mask[i, :s_doctor_max[i] + 1] = 1
        doctor_logits = doctor_logits * doctor_mask + (1 - doctor_mask) * -1e9

        return nurse_logits, doctor_logits, wait_time_pred


class ConstraintAwareLoss(nn.Module):
    """
    约束感知损失函数：动态调节损失权重
    核心思想：根据约束紧迫程度自适应调整惩罚强度
    """
    def __init__(self, alpha=1.0, beta=2.0, gamma=1.5):
        super().__init__()
        self.alpha = alpha  # 等待时间权重
        self.beta = beta    # 流失权重  
        self.gamma = gamma  # 超限权重
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

        # 约束违反检测器
        self.constraint_detector = nn.Sequential(
            nn.Linear(9, 32),  # 输入特征维度（假设9个关键特征）
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(), 
            nn.Linear(16, 3),  # 输出3个约束违反概率
            nn.Sigmoid()
        )

    def compute_constraint_urgency(self, features):
        """
        计算约束紧迫度
        返回：[wait_urgency, loss_urgency, overload_urgency]
        """
        # 提取关键约束特征：λ, μ_nurse, μ_doctor, Tmax, Smax等
        key_features = features[:, [1, 2, 3, 6, 4, 5, 7, 8]]  # 选择关键特征

        # 计算利用率和紧迫度指标
        lambda_val = features[:, 1]  # 到达率
        mu_nurse = features[:, 2]    # 护士服务率
        mu_doctor = features[:, 3]   # 医生服务率
        tmax = features[:, 6]        # 最大等待时间
        s_nurse_max = features[:, 4] # 最大护士数
        s_doctor_max = features[:, 5] # 最大医生数

        # 系统利用率（排队论指标）
        nurse_utilization = lambda_val / (s_nurse_max * mu_nurse + 1e-8)
        doctor_utilization = lambda_val / (s_doctor_max * mu_doctor + 1e-8)

        # 时间紧迫度（基于Tmax）
        time_pressure = torch.clamp(lambda_val / (mu_nurse + mu_doctor + 1e-8) / tmax, 0, 1)

        # 构造约束特征向量
        constraint_features = torch.stack([
            nurse_utilization, doctor_utilization, time_pressure,
            lambda_val / 100.0,  # 标准化到达率
            mu_nurse / 10.0,     # 标准化服务率
            mu_doctor / 10.0,
            tmax / 60.0,         # 标准化时间限制
            s_nurse_max / 20.0,  # 标准化资源上限
            s_doctor_max / 20.0
        ], dim=1)

        # 通过神经网络预测约束违反概率
        urgency_scores = self.constraint_detector(constraint_features)
        return urgency_scores

    def forward(self, nurse_logits, doctor_logits, wait_time_pred,
                nurse_targets, doctor_targets, wait_time_targets,
                patient_loss, hospital_overload, features=None):

        # 基础分类损失
        nurse_loss = self.ce(nurse_logits, nurse_targets)
        doctor_loss = self.ce(doctor_logits, doctor_targets)

        # 等待时间MSE损失
        wait_time_loss = self.mse(wait_time_pred.squeeze(), wait_time_targets)

        # 流失和超限损失
        loss_penalty = patient_loss.float().mean()
        overload_penalty = hospital_overload.float().mean()

        # 动态权重调节（核心创新）
        if features is not None:
            urgency_scores = self.compute_constraint_urgency(features)
            wait_urgency = urgency_scores[:, 0].mean()    # 等待时间紧迫度
            loss_urgency = urgency_scores[:, 1].mean()    # 流失紧迫度  
            overload_urgency = urgency_scores[:, 2].mean() # 超限紧迫度

            # 自适应权重调节
            adaptive_alpha = self.alpha * (1.0 + 2.0 * wait_urgency)
            adaptive_beta = self.beta * (1.0 + 3.0 * loss_urgency)
            adaptive_gamma = self.gamma * (1.0 + 4.0 * overload_urgency)
        else:
            # 回退到固定权重
            adaptive_alpha = self.alpha
            adaptive_beta = self.beta
            adaptive_gamma = self.gamma

        # 计算总损失
        total_loss = (nurse_loss + doctor_loss +
                     adaptive_alpha * wait_time_loss +
                     adaptive_beta * loss_penalty +
                     adaptive_gamma * overload_penalty)

        return total_loss, {
            'nurse_loss': nurse_loss.item(),
            'doctor_loss': doctor_loss.item(),
            'wait_time_loss': wait_time_loss.item(),
            'loss_penalty': loss_penalty.item(),
            'overload_penalty': overload_penalty.item(),
            'adaptive_alpha': adaptive_alpha.item() if hasattr(adaptive_alpha, 'item') else adaptive_alpha,
            'adaptive_beta': adaptive_beta.item() if hasattr(adaptive_beta, 'item') else adaptive_beta,
            'adaptive_gamma': adaptive_gamma.item() if hasattr(adaptive_gamma, 'item') else adaptive_gamma
        }


class AdaptiveLoss(nn.Module):
    """
    渐进式对抗训练损失函数
    核心创新：动态调节损失权重，重点关注边界样本
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha  # 主任务权重（固定）
        self.beta = beta    # 时间约束权重（动态）
        self.gamma = gamma  # 资源约束权重（动态）
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, nurse_logits, doctor_logits, wait_time_pred,
                nurse_targets, doctor_targets, wait_time_targets,
                patient_loss, hospital_overload, features=None, epoch=0):
        
        # 基础分类损失
        nurse_loss = self.ce(nurse_logits, nurse_targets)
        doctor_loss = self.ce(doctor_logits, doctor_targets)
        
        # 等待时间MSE损失
        wait_time_loss = self.mse(wait_time_pred.squeeze(), wait_time_targets)
        
        # 动态权重调整（渐进式增长）
        # 早期阶段（epoch < 30）：以优化主任务为主
        # 后期阶段（epoch ≥ 30）：逐步加大约束惩罚
        curr_beta = self.beta * (1 + 0.05 * epoch)   # 时间约束权重线性增长
        curr_gamma = self.gamma * (1 + 0.03 * epoch) # 资源约束权重线性增长
        
        # 分级惩罚计算
        if features is not None:
            # 提取Tmax用于计算严重违反
            tmax = features[:, 6]  # 最大等待时间
            
            # 时间违反量计算
            time_violation = torch.relu(wait_time_pred.squeeze() - wait_time_targets)
            
            # 严重超时判定（超过20%Tmax）
            severe_time_mask = (time_violation > 0.2 * tmax).float()
            
            # 对严重违反样本施加3倍惩罚
            time_penalty = (1 + severe_time_mask * 2) * time_violation
            
            # 流失和超限损失
            loss_penalty = patient_loss.float()
            overload_penalty = hospital_overload.float()
            
            # 边界样本增强：对约束违反样本加权
            violation_mask = ((patient_loss > 0) | (hospital_overload > 0)).float()
            boundary_weight = 1.0 + violation_mask * 1.5  # 边界样本1.5倍权重
            
            # 综合损失计算
            total_loss = (
                self.alpha * (nurse_loss + doctor_loss + wait_time_loss) +
                curr_beta * (boundary_weight * time_penalty).mean() +
                curr_gamma * (boundary_weight * (loss_penalty + overload_penalty)).mean()
            )
        else:
            # 回退到基础损失
            loss_penalty = patient_loss.float().mean()
            overload_penalty = hospital_overload.float().mean()
            
            total_loss = (
                self.alpha * (nurse_loss + doctor_loss + wait_time_loss) +
                curr_beta * loss_penalty +
                curr_gamma * overload_penalty
            )
        
        return total_loss, {
            'nurse_loss': nurse_loss.item(),
            'doctor_loss': doctor_loss.item(),
            'wait_time_loss': wait_time_loss.item(),
            'loss_penalty': loss_penalty.mean().item() if hasattr(loss_penalty, 'mean') else loss_penalty,
            'overload_penalty': overload_penalty.mean().item() if hasattr(overload_penalty, 'mean') else overload_penalty,
            'curr_beta': curr_beta,
            'curr_gamma': curr_gamma,
            'epoch': epoch
        }


class ViolationAwareDataset(Dataset):
    """
    约束违反感知数据集
    支持动态采样：平衡正常样本与约束违反样本
    """
    def __init__(self, features, targets, wait_times=None, patient_loss=None, hospital_overload=None):
        self.features = torch.FloatTensor(features)
        self.nurse_targets = torch.LongTensor(targets[:, 0])
        self.doctor_targets = torch.LongTensor(targets[:, 1])
        
        if wait_times is not None:
            self.wait_times = torch.FloatTensor(wait_times)
        else:
            self.wait_times = torch.zeros(len(features))
            
        if patient_loss is not None:
            self.patient_loss = torch.FloatTensor(patient_loss)
        else:
            self.patient_loss = torch.zeros(len(features))
            
        if hospital_overload is not None:
            self.hospital_overload = torch.FloatTensor(hospital_overload)
        else:
            self.hospital_overload = torch.zeros(len(features))
        
        # 预计算违反样本索引
        self.violation_indices = self._find_violation_indices()
        self.normal_indices = self._find_normal_indices()
        
    def _find_violation_indices(self):
        """找到约束违反样本的索引"""
        violation_mask = (self.patient_loss > 0) | (self.hospital_overload > 0)
        return torch.where(violation_mask)[0].numpy()
    
    def _find_normal_indices(self):
        """找到正常样本的索引"""
        normal_mask = (self.patient_loss == 0) & (self.hospital_overload == 0)
        return torch.where(normal_mask)[0].numpy()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (self.features[idx],
                (self.nurse_targets[idx], self.doctor_targets[idx]),
                self.wait_times[idx],
                self.patient_loss[idx],
                self.hospital_overload[idx])
    
    def get_violation_ratio(self):
        """获取违反样本比例"""
        return len(self.violation_indices) / len(self)


def sample_with_violation(dataset, batch_size, violation_ratio, epoch):
    """
    混合采样：平衡正常样本与约束违反样本
    随着训练进行，逐步增加违反样本比例
    """
    # 动态调整违反样本比例（渐进式增长）
    if epoch < 20:
        # 早期阶段：少量违反样本
        dynamic_ratio = violation_ratio * 0.3
    elif epoch < 40:
        # 中期阶段：逐步增加
        dynamic_ratio = violation_ratio * 0.6
    else:
        # 后期阶段：完整比例
        dynamic_ratio = violation_ratio
    
    # 计算各类样本数量
    n_violation = min(int(batch_size * dynamic_ratio), len(dataset.violation_indices))
    n_normal = batch_size - n_violation
    
    # 采样索引
    if n_violation > 0 and len(dataset.violation_indices) > 0:
        violation_idx = np.random.choice(dataset.violation_indices, n_violation, replace=True)
    else:
        violation_idx = []
    
    if n_normal > 0 and len(dataset.normal_indices) > 0:
        normal_idx = np.random.choice(dataset.normal_indices, n_normal, replace=True)
    else:
        normal_idx = []
    
    # 合并索引
    combined_idx = np.concatenate([normal_idx, violation_idx])
    np.random.shuffle(combined_idx)
    
    return combined_idx


# 保持向后兼容
CustomLoss = ConstraintAwareLoss


class HospitalDataset(Dataset):
    def __init__(self, features, targets, wait_times=None, patient_loss=None, hospital_overload=None):
        self.features = torch.FloatTensor(features)
        self.nurse_targets = torch.LongTensor(targets[:, 0])
        self.doctor_targets = torch.LongTensor(targets[:, 1])

        if wait_times is not None:
            self.wait_times = torch.FloatTensor(wait_times)
        else:
            self.wait_times = torch.zeros(len(features))

        if patient_loss is not None:
            self.patient_loss = torch.FloatTensor(patient_loss)
        else:
            self.patient_loss = torch.zeros(len(features))

        if hospital_overload is not None:
            self.hospital_overload = torch.FloatTensor(hospital_overload)
        else:
            self.hospital_overload = torch.zeros(len(features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx],
                (self.nurse_targets[idx], self.doctor_targets[idx]),
                self.wait_times[idx],
                self.patient_loss[idx],
                self.hospital_overload[idx])


def progressive_adversarial_training(model, train_dataset, val_loader, device, best_params):
    """
    渐进式对抗训练法
    核心创新：
    1. 动态样本采样 - 逐步增加边界样本比例
    2. 自适应损失权重 - 根据训练阶段调节约束惩罚
    3. 边界样本增强 - 重点关注约束违反样本
    """
    print("=== 渐进式对抗训练开始 ===")
    print("解决传统训练问题：固定损失权重 → 动态自适应权重")
    print("医疗调度特殊性：边界样本决定系统稳定性")
    
    # 转换为约束感知数据集
    if not isinstance(train_dataset, ViolationAwareDataset):
        print("转换为约束感知数据集...")
        train_dataset = ViolationAwareDataset(
            train_dataset.features.numpy(),
            torch.stack([train_dataset.nurse_targets, train_dataset.doctor_targets], dim=1).numpy(),
            train_dataset.wait_times.numpy(),
            train_dataset.patient_loss.numpy(),
            train_dataset.hospital_overload.numpy()
        )
    
    violation_ratio = train_dataset.get_violation_ratio()
    print(f"数据集违反样本比例: {violation_ratio:.3f}")
    
    # 创建自适应损失函数
    criterion = AdaptiveLoss(
        alpha=best_params['alpha'],
        beta=best_params['beta'], 
        gamma=best_params['gamma']
    )
    
    # 优化器设置
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
    
    best_loss = float('inf')
    patience = 20
    no_improve = 0
    
    # 训练统计
    violation_stats = []
    
    for epoch in range(60):  # 渐进式训练60轮
        model.train()
        total_loss = 0
        epoch_violations = 0
        epoch_samples = 0
        
        # 动态采样批次数
        n_batches = len(train_dataset) // BATCH_SIZE
        
        for batch_idx in range(n_batches):
            # 动态混合采样
            sample_indices = sample_with_violation(
                train_dataset, BATCH_SIZE, violation_ratio, epoch
            )
            
            # 构建批次数据
            batch_features = train_dataset.features[sample_indices]
            batch_nurse_targets = train_dataset.nurse_targets[sample_indices]
            batch_doctor_targets = train_dataset.doctor_targets[sample_indices]
            batch_wait_times = train_dataset.wait_times[sample_indices]
            batch_patient_loss = train_dataset.patient_loss[sample_indices]
            batch_hospital_overload = train_dataset.hospital_overload[sample_indices]
            
            # 移动到设备
            batch_features = batch_features.to(device)
            batch_nurse_targets = batch_nurse_targets.to(device)
            batch_doctor_targets = batch_doctor_targets.to(device)
            batch_wait_times = batch_wait_times.to(device)
            batch_patient_loss = batch_patient_loss.to(device)
            batch_hospital_overload = batch_hospital_overload.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(batch_features)
            
            # 自适应损失计算（传入epoch用于动态权重）
            loss, loss_dict = criterion(
                nurse_logits, doctor_logits, wait_pred,
                batch_nurse_targets, batch_doctor_targets, batch_wait_times,
                batch_patient_loss, batch_hospital_overload, 
                batch_features, epoch
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 统计违反样本
            batch_violations = ((batch_patient_loss > 0) | (batch_hospital_overload > 0)).sum().item()
            epoch_violations += batch_violations
            epoch_samples += len(sample_indices)
        
        scheduler.step()
        avg_loss = total_loss / n_batches
        
        # 验证
        val_loss = validate_model(model, val_loader, criterion, device)
        
        # 记录违反样本统计
        violation_rate = epoch_violations / epoch_samples
        violation_stats.append(violation_rate)
        
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_progressive_model.pth')
        else:
            no_improve += 1
        
        # 训练监控
        if epoch % 5 == 0:
            print(f"Progressive Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
            print(f"  动态权重 - Beta: {loss_dict['curr_beta']:.3f}, Gamma: {loss_dict['curr_gamma']:.3f}")
            print(f"  违反样本率: {violation_rate:.3f}, 训练阶段: {'早期' if epoch < 20 else '中期' if epoch < 40 else '后期'}")
            
            # 边界样本分析
            if epoch >= 20:
                recent_violation_trend = np.mean(violation_stats[-5:]) if len(violation_stats) >= 5 else violation_rate
                print(f"  边界样本趋势: {recent_violation_trend:.3f} (目标: 重点关注约束违反)")
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_progressive_model.pth'))
    
    print("=== 渐进式对抗训练完成 ===")
    print(f"最终违反样本处理能力提升: {violation_stats[-1]/violation_stats[0]:.2f}x" if len(violation_stats) > 1 else "")
    
    return model


def constraint_aware_training(model, train_loader, val_loader, device):
    """
    约束感知的三阶段训练法
    核心创新：结合ClinicalPAN和ConstraintAwareLoss实现医疗约束的深度理解
    """
    """三阶段训练法"""

    # 第一阶段：预训练PAN特征提取层
    print("=== 第一阶段：预训练PAN特征提取层 ===")
    model.freeze_fc_layers()

    optimizer_stage1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE * 2, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=0.5, beta=1.0, gamma=1.0)

    for epoch in range(20):  # 预训练20轮
        model.train()
        total_loss = 0
        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage1.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, loss_dict = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t, features
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage1.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Stage 1 Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

    # 第二阶段：微调最后三层
    print("=== 第二阶段：微调最后三层 ===")
    model.unfreeze_last_layers()

    optimizer_stage2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=1.0, beta=1.5, gamma=1.2)

    for epoch in range(30):  # 微调30轮
        model.train()
        total_loss = 0
        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage2.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, loss_dict = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t, features
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage2.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Stage 2 Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

    # 第三阶段：启用注意力模块，使用加权损失
    print("=== 第三阶段：启用注意力模块 ===")
    model.unfreeze_all()
    model.enable_attention()

    optimizer_stage3 = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE * 0.5, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=1.5, beta=2.0, gamma=1.8)  # 加权损失
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage3, T_max=10, eta_min=1e-6)

    best_loss = float('inf')
    patience = 15
    no_improve = 0

    for epoch in range(30):  # 强化训练30轮
        model.train()
        total_loss = 0
        loss_components = {'nurse': 0, 'doctor': 0, 'wait': 0, 'loss': 0, 'overload': 0}

        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage3.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, loss_dict = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t, features
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage3.step()

            total_loss += loss.item()
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # 验证
        val_loss = validate_model(model, val_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_pan_dnn_model.pth')
        else:
            no_improve += 1

        if epoch % 5 == 0:
            print(f"Stage 3 Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
            print(f"  Components - Nurse: {loss_components['nurse']/len(train_loader):.3f}, "
                  f"Doctor: {loss_components['doctor']/len(train_loader):.3f}, "
                  f"Wait: {loss_components['wait']/len(train_loader):.3f}")

            # 约束感知训练的特殊监控
            if 'adaptive_alpha' in loss_components:
                print(f"  约束感知权重 - Alpha: {loss_components['adaptive_alpha']/len(train_loader):.3f}, "
                      f"Beta: {loss_components['adaptive_beta']/len(train_loader):.3f}, "
                      f"Gamma: {loss_components['adaptive_gamma']/len(train_loader):.3f}")
                print(f"  约束理解能力：模型正在动态调节损失权重以适应医疗约束")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_pan_dnn_model.pth'))
    return model


def validate_model(model, val_loader, criterion, device, epoch=0):
    """验证模型（支持AdaptiveLoss的epoch参数）"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in val_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            nurse_logits, doctor_logits, wait_pred = model(features)
            
            # 检查损失函数类型，决定是否传递epoch参数
            if isinstance(criterion, AdaptiveLoss):
                loss, _ = criterion(
                    nurse_logits, doctor_logits, wait_pred,
                    nurse_t, doctor_t, wait_t, loss_t, overload_t, features, epoch
                )
            else:
                # 兼容其他损失函数
                if hasattr(criterion, 'forward') and 'features' in criterion.forward.__code__.co_varnames:
                    loss, _ = criterion(
                        nurse_logits, doctor_logits, wait_pred,
                        nurse_t, doctor_t, wait_t, loss_t, overload_t, features
                    )
                else:
                    loss, _ = criterion(
                        nurse_logits, doctor_logits, wait_pred,
                        nurse_t, doctor_t, wait_t, loss_t, overload_t
                    )
            total_loss += loss.item()

    return total_loss / len(val_loader)


def run_comparison_analysis(X_train, X_test, y_train, y_test, X_raw_test,
                          n_nurse_classes, n_doctor_classes, device, model, test_loader):
    """运行对比分析"""
    from visualization_comparison import (
        train_traditional_models, queue_theory_baseline,
        create_comprehensive_visualization, create_performance_summary_table
    )

    print("\n=== 开始对比分析 ===")

    # 1. 训练传统模型
    traditional_results = train_traditional_models(
        X_train, y_train, X_test, y_test, n_nurse_classes, n_doctor_classes, device
    )

    # 2. 排队论基线
    queue_nurse_pred, queue_doctor_pred = queue_theory_baseline(X_raw_test)
    queue_results = {
        'nurse_pred': queue_nurse_pred,
        'doctor_pred': queue_doctor_pred,
        'nurse_metrics': {
            'MAE': mean_absolute_error(y_test[:, 0], queue_nurse_pred),
            'MSE': mean_squared_error(y_test[:, 0], queue_nurse_pred),
            'R2': r2_score(y_test[:, 0], queue_nurse_pred),
            'Accuracy': accuracy_score(y_test[:, 0], queue_nurse_pred)
        },
        'doctor_metrics': {
            'MAE': mean_absolute_error(y_test[:, 1], queue_doctor_pred),
            'MSE': mean_squared_error(y_test[:, 1], queue_doctor_pred),
            'R2': r2_score(y_test[:, 1], queue_doctor_pred),
            'Accuracy': accuracy_score(y_test[:, 1], queue_doctor_pred)
        }
    }

    # 3. 获取混合模型结果
    hybrid_results = get_hybrid_model_results(model, test_loader, y_test, device)

    # 4. 创建综合可视化
    create_comprehensive_visualization(hybrid_results, traditional_results, queue_results, y_test)

    # 5. 创建性能汇总表
    create_performance_summary_table(hybrid_results, traditional_results, queue_results)

    return hybrid_results, traditional_results, queue_results

def get_hybrid_model_results(model, test_loader, y_test, device):
    """获取混合模型结果"""
    model.eval()
    all_pred_n = []
    all_pred_d = []

    with torch.no_grad():
        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in test_loader:
            features = features.to(device)
            nurse_logits, doctor_logits, wait_pred = model(features)
            pred_n = torch.argmax(nurse_logits, dim=1).cpu().numpy()
            pred_d = torch.argmax(doctor_logits, dim=1).cpu().numpy()
            all_pred_n.extend(pred_n)
            all_pred_d.extend(pred_d)

    return {
        'nurse_pred': np.array(all_pred_n),
        'doctor_pred': np.array(all_pred_d),
        'nurse_metrics': {
            'MAE': mean_absolute_error(y_test[:, 0], all_pred_n),
            'MSE': mean_squared_error(y_test[:, 0], all_pred_n),
            'R2': r2_score(y_test[:, 0], all_pred_n),
            'Accuracy': accuracy_score(y_test[:, 0], all_pred_n)
        },
        'doctor_metrics': {
            'MAE': mean_absolute_error(y_test[:, 1], all_pred_d),
            'MSE': mean_squared_error(y_test[:, 1], all_pred_d),
            'R2': r2_score(y_test[:, 1], all_pred_d),
            'Accuracy': accuracy_score(y_test[:, 1], all_pred_d)
        }
    }

def grid_search_penalty_weights(model_class, X_train, y_train, wait_train, loss_train, overload_train,
                               X_val, y_val, wait_val, loss_val, overload_val,
                               n_nurse_classes, n_doctor_classes, device):
    """网格搜索最佳惩罚函数权重参数"""
    print("=== 开始网格搜索惩罚函数权重参数 ===")

    # 定义搜索范围
    beta_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # 流失权重
    gamma_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # 超限权重
    alpha = 1.0  # 等待时间权重固定

    best_loss = float('inf')
    best_params = {'alpha': alpha, 'beta': 1.0, 'gamma': 1.0}
    results = []

    # 创建数据加载器
    train_dataset = HospitalDataset(X_train, y_train, wait_train, loss_train, overload_train)
    val_dataset = HospitalDataset(X_val, y_val, wait_val, loss_val, overload_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    total_combinations = len(beta_range) * len(gamma_range)
    current_combination = 0

    for beta in beta_range:
        for gamma in gamma_range:
            current_combination += 1
            print(f"测试组合 {current_combination}/{total_combinations}: alpha={alpha}, beta={beta}, gamma={gamma}")

            # 创建新模型
            model = model_class(
                input_dim=X_train.shape[1],
                hidden_layers=HIDDEN_LAYERS,
                n_nurse_classes=n_nurse_classes,
                n_doctor_classes=n_doctor_classes
            ).to(device)

            # 训练模型（简化版本，只训练少量epoch）
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            criterion = CustomLoss(alpha=alpha, beta=beta, gamma=gamma)

            model.train()
            for epoch in range(10):  # 快速训练10轮
                total_loss = 0
                for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
                    features = features.to(device)
                    nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
                    wait_t = wait_t.to(device)
                    loss_t, overload_t = loss_t.to(device), overload_t.to(device)

                    optimizer.zero_grad()
                    nurse_logits, doctor_logits, wait_pred = model(features)
                    loss, loss_dict = criterion(
                        nurse_logits, doctor_logits, wait_pred,
                        nurse_t, doctor_t, wait_t, loss_t, overload_t
                    )
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            # 验证
            val_loss = validate_model(model, val_loader, criterion, device)

            results.append({
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'val_loss': val_loss
            })

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
                print(f"  新的最佳参数: {best_params}, 验证损失: {val_loss:.4f}")

            # 清理内存
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n=== 网格搜索完成 ===")
    print(f"最佳参数: {best_params}")
    print(f"最佳验证损失: {best_loss:.4f}")

    # 保存搜索结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('grid_search_results.csv', index=False)
    print("网格搜索结果已保存至 grid_search_results.csv")

    return best_params

def modified_three_stage_training(model, train_loader, val_loader, device, best_params):
    """使用最佳参数的三阶段训练法"""

    # 第一阶段：预训练PAN特征提取层
    print("=== 第一阶段：预训练PAN特征提取层 ===")
    model.freeze_fc_layers()

    optimizer_stage1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE * 2, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=best_params['alpha']*0.5,
                          beta=best_params['beta'],
                          gamma=best_params['gamma'])

    for epoch in range(20):
        model.train()
        total_loss = 0
        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage1.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, loss_dict = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage1.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Stage 1 Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

    # 第二阶段：微调最后三层
    print("=== 第二阶段：微调最后三层 ===")
    model.unfreeze_last_layers()

    optimizer_stage2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=best_params['alpha'],
                          beta=best_params['beta']*1.5,
                          gamma=best_params['gamma']*1.2)

    for epoch in range(30):
        model.train()
        total_loss = 0
        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage2.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, loss_dict = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage2.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Stage 2 Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

    # 第三阶段：启用注意力模块，使用最佳权重
    print("=== 第三阶段：启用注意力模块 ===")
    model.unfreeze_all()
    model.enable_attention()

    optimizer_stage3 = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE * 0.5, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=best_params['alpha']*1.5,
                          beta=best_params['beta']*2.0,
                          gamma=best_params['gamma']*1.8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage3, T_max=10, eta_min=1e-6)

    best_loss = float('inf')
    patience = 15
    no_improve = 0

    for epoch in range(30):
        model.train()
        total_loss = 0
        loss_components = {'nurse': 0, 'doctor': 0, 'wait': 0, 'loss': 0, 'overload': 0}

        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage3.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, loss_dict = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage3.step()

            total_loss += loss.item()
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        val_loss = validate_model(model, val_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_pan_dnn_model.pth')
        else:
            no_improve += 1

        if epoch % 5 == 0:
            print(f"Stage 3 Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
            print(f"  Components - Nurse: {loss_components['nurse']/len(train_loader):.3f}, "
                  f"Doctor: {loss_components['doctor']/len(train_loader):.3f}, "
                  f"Wait: {loss_components['wait']/len(train_loader):.3f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load('best_pan_dnn_model.pth'))
    return model

def compute_penalty_loss_for_all_methods(hybrid_results, traditional_results, queue_results,
                                       X_raw_test, y_test, best_params):
    """计算所有方法的惩罚函数损失值"""

    def calculate_penalty_loss(nurse_pred, doctor_pred, X_raw, y_true, params):
        """计算单个方法的惩罚损失"""
        total_penalty_loss = 0
        n_samples = len(nurse_pred)

        for i in range(n_samples):
            # 获取预测值和真实值
            pred_nurses = nurse_pred[i]
            pred_doctors = doctor_pred[i]
            true_nurses = y_true[i, 0]
            true_doctors = y_true[i, 1]

            # 获取原始特征
            row = X_raw.iloc[i]
            lambd = row['lambda']
            mu_nurse = row['mu_nurse']
            mu_doctor = row['mu_doctor']
            Tmax = row['Tmax']
            s_nurse_max = row['s_nurse_max']
            s_doctor_max = row['s_doctor_max']

            # 计算预测配置下的系统指标
            from generate import compute_integrated_system_metrics
            pred_metrics = compute_integrated_system_metrics(
                lambd, mu_nurse, pred_nurses, mu_doctor, pred_doctors
            )

            # 计算真实配置下的系统指标
            true_metrics = compute_integrated_system_metrics(
                lambd, mu_nurse, true_nurses, mu_doctor, true_doctors
            )

            if pred_metrics is None or true_metrics is None:
                # 系统不稳定，给予高惩罚
                penalty_loss = 100.0
            else:
                # 等待时间损失 (MSE)
                wait_loss = (pred_metrics['system_total_time'] - true_metrics['system_total_time']) ** 2

                # 流失损失 (预测配置是否违反时间约束)
                patient_loss = 1.0 if pred_metrics['system_total_time'] > Tmax else 0.0

                # 超限损失 (预测配置是否超过最大人员限制)
                hospital_overload = 1.0 if (pred_nurses > s_nurse_max or pred_doctors > s_doctor_max) else 0.0

                # 计算总惩罚损失
                penalty_loss = (params['alpha'] * wait_loss +
                              params['beta'] * patient_loss +
                              params['gamma'] * hospital_overload)

            total_penalty_loss += penalty_loss

        return total_penalty_loss / n_samples  # 返回平均损失

    # 计算各方法的惩罚损失
    penalty_losses = {}

    # PAN+DNN混合模型
    penalty_losses['PAN+DNN Hybrid'] = calculate_penalty_loss(
        hybrid_results['nurse_pred'], hybrid_results['doctor_pred'],
        X_raw_test, y_test, best_params  # 传递 y_test 作为 y_true 参数
    )

    # 传统PAN模型
    penalty_losses['Traditional PAN'] = calculate_penalty_loss(
        traditional_results['Traditional_PAN']['nurse_pred'],
        traditional_results['Traditional_PAN']['doctor_pred'],
        X_raw_test, y_test, best_params  # 传递 y_test 作为 y_true 参数
    )

    # 传统DNN模型
    penalty_losses['Traditional DNN'] = calculate_penalty_loss(
        traditional_results['Traditional_DNN']['nurse_pred'],
        traditional_results['Traditional_DNN']['doctor_pred'],
        X_raw_test, y_test, best_params  # 传递 y_test 作为 y_true 参数
    )

    # 排队论方法
    penalty_losses['Queue Theory'] = calculate_penalty_loss(
        queue_results['nurse_pred'], queue_results['doctor_pred'],
        X_raw_test, y_test, best_params  # 传递 y_test 作为 y_true 参数
    )

    return penalty_losses

def run_comparison_analysis_with_penalty(X_train, X_test, y_train, y_test, X_raw_test,
                                       n_nurse_classes, n_doctor_classes, device, model,
                                       test_loader, best_params):
    """基于惩罚函数损失的对比分析"""
    from visualization_comparison import (
        train_traditional_models, queue_theory_baseline_improved,
        create_penalty_loss_visualization
    )

    print("\n=== 开始基于惩罚函数的对比分析 ===")

    # 1. 训练传统模型
    traditional_results = train_traditional_models(
        X_train, y_train, X_test, y_test, n_nurse_classes, n_doctor_classes, device
    )

    # 2. 排队论基线（使用完整算法）
    queue_nurse_pred, queue_doctor_pred = queue_theory_baseline_improved(X_raw_test)
    queue_results = {
        'nurse_pred': queue_nurse_pred,
        'doctor_pred': queue_doctor_pred
    }

    # 3. 获取混合模型结果
    hybrid_results = get_hybrid_model_results(model, test_loader, y_test, device)

    # 4. 计算所有方法的修正后惩罚函数损失
    penalty_losses = compute_penalty_loss_for_all_methods(
        hybrid_results, traditional_results, queue_results,
        X_raw_test, y_test, best_params  # 确保传递 y_test 参数
    )

    # 5. 打印惩罚损失比较结果
    print("\n=== 惩罚函数损失比较结果 ===")
    sorted_methods = sorted(penalty_losses.items(), key=lambda x: x[1])
    for i, (method, loss) in enumerate(sorted_methods, 1):
        print(f"{i}. {method}: {loss:.4f}")

    # 6. 创建惩罚损失可视化
    create_penalty_loss_visualization(penalty_losses, best_params)

    return penalty_losses, hybrid_results, traditional_results, queue_results

def compute_penalty_loss_for_all_methods_corrected(hybrid_results, traditional_results, queue_results, X_raw_test,y_test,  best_params):
    """计算所有方法的惩罚函数损失值（修正版）"""

    def calculate_penalty_loss(nurse_pred, doctor_pred, X_raw, y_true, params):
        """计算单个方法的惩罚损失"""
        total_penalty_loss = 0
        n_samples = len(nurse_pred)

        for i in range(n_samples):
            # 获取预测值和真实值
            pred_nurses = nurse_pred[i]
            pred_doctors = doctor_pred[i]
            true_nurses = y_true[i, 0]
            true_doctors = y_true[i, 1]

            # 获取原始特征
            row = X_raw.iloc[i]
            lambd = row['lambda']
            mu_nurse = row['mu_nurse']
            mu_doctor = row['mu_doctor']
            Tmax = row['Tmax']
            s_nurse_max = row['s_nurse_max']
            s_doctor_max = row['s_doctor_max']

            # 计算预测配置下的系统指标
            from generate import compute_integrated_system_metrics
            pred_metrics = compute_integrated_system_metrics(
                lambd, mu_nurse, pred_nurses, mu_doctor, pred_doctors
            )

            # 计算真实配置下的系统指标
            true_metrics = compute_integrated_system_metrics(
                lambd, mu_nurse, true_nurses, mu_doctor, true_doctors
            )

            if pred_metrics is None or true_metrics is None:
                # 系统不稳定，给予高惩罚
                penalty_loss = 100.0
            else:
                # 等待时间损失 (MSE)
                wait_loss = (pred_metrics['system_total_time'] - true_metrics['system_total_time']) ** 2

                # 流失损失 (预测配置是否违反时间约束)
                patient_loss = 1.0 if pred_metrics['system_total_time'] > Tmax else 0.0

                # 超限损失 (预测配置是否超过最大人员限制)
                hospital_overload = 1.0 if (pred_nurses > s_nurse_max or pred_doctors > s_doctor_max) else 0.0

                # 计算总惩罚损失
                penalty_loss = (params['alpha'] * wait_loss +
                              params['beta'] * patient_loss +
                              params['gamma'] * hospital_overload)

            total_penalty_loss += penalty_loss

        return total_penalty_loss / n_samples  # 返回平均损失

    # 计算各方法的惩罚损失
    penalty_losses = {}

    # PAN+DNN混合模型
    penalty_losses['PAN+DNN Hybrid'] = calculate_penalty_loss(
        hybrid_results['nurse_pred'], hybrid_results['doctor_pred'],
        X_raw_test, y_test, best_params
    )

    # 传统PAN模型
    penalty_losses['Traditional PAN'] = calculate_penalty_loss(
        traditional_results['Traditional_PAN']['nurse_pred'],
        traditional_results['Traditional_PAN']['doctor_pred'],
        X_raw_test, y_test, best_params
    )

    # 传统DNN模型
    penalty_losses['Traditional DNN'] = calculate_penalty_loss(
        traditional_results['Traditional_DNN']['nurse_pred'],
        traditional_results['Traditional_DNN']['doctor_pred'],
        X_raw_test, y_test, best_params
    )

    # 排队论方法
    penalty_losses['Queue Theory'] = calculate_penalty_loss(
        queue_results['nurse_pred'], queue_results['doctor_pred'],
        X_raw_test, y_test, best_params
    )

    return penalty_losses

def enhanced_three_stage_training(model, train_loader, val_loader, device, best_params):
    """使用增强版三阶段训练法"""

    # 第一阶段：预训练PAN特征提取层
    print("=== 第一阶段：预训练PAN特征提取层 ===")
    model.freeze_fc_layers()

    optimizer_stage1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE * 2, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=best_params['alpha']*0.5,
                          beta=best_params['beta'],
                          gamma=best_params['gamma'])

    for epoch in range(20):
        model.train()
        total_loss = 0
        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage1.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, loss_dict = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage1.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Stage 1 Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

    # 第二阶段：微调最后三层
    print("=== 第二阶段：微调最后三层 ===")
    model.unfreeze_last_layers()

    optimizer_stage2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=best_params['alpha'],
                          beta=best_params['beta']*1.5,
                          gamma=best_params['gamma']*1.2)

    for epoch in range(30):
        model.train()
        total_loss = 0
        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage2.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, loss_dict = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage2.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Stage 2 Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

    # 第三阶段：启用注意力模块，使用最佳权重
    print("=== 第三阶段：启用注意力模块 ===")
    model.unfreeze_all()
    model.enable_attention()

    optimizer_stage3 = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE * 0.5, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=best_params['alpha']*1.5,
                          beta=best_params['beta']*2.0,
                          gamma=best_params['gamma']*1.8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage3, T_max=10, eta_min=1e-6)

    best_loss = float('inf')
    patience = 15
    no_improve = 0

    for epoch in range(30):
        model.train()
        total_loss = 0
        loss_components = {'nurse': 0, 'doctor': 0, 'wait': 0, 'loss': 0, 'overload': 0}

        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage3.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, loss_dict = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage3.step()

            total_loss += loss.item()
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        val_loss = validate_model(model, val_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_pan_dnn_model.pth')
        else:
            no_improve += 1

        if epoch % 5 == 0:
            print(f"Stage 3 Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
            print(f"  Components - Nurse: {loss_components['nurse']/len(train_loader):.3f}, "
                  f"Doctor: {loss_components['doctor']/len(train_loader):.3f}, "
                  f"Wait: {loss_components['wait']/len(train_loader):.3f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load('best_pan_dnn_model.pth'))
    return model

def main():
    # 数据加载
    df = pd.read_csv(DATA_PATH)
    X = df[['scenario', 'lambda', 'mu_nurse', 'mu_doctor',
            's_nurse_max', 's_doctor_max', 'Tmax', 'nurse_price', 'doctor_price']]
    y = df[['optimal_nurses', 'optimal_doctors']].values

    # 添加额外目标变量
    wait_times = df['system_total_time'].values if 'system_total_time' in df.columns else np.zeros(len(df))
    patient_loss = df['patient_loss'].values if 'patient_loss' in df.columns else np.zeros(len(df))
    hospital_overload = df['hospital_overload'].values if 'hospital_overload' in df.columns else np.zeros(len(df))

    # 数据预处理
    preprocessor = ColumnTransformer(
        transformers=[
            ('scenario', OneHotEncoder(), ['scenario']),
            ('num', StandardScaler(), ['lambda', 'mu_nurse', 'mu_doctor', 'Tmax']),
            ('passthrough', 'passthrough', ['s_nurse_max', 's_doctor_max', 'nurse_price', 'doctor_price'])
        ])

    # 数据分割：训练集、验证集、测试集
    indices = np.arange(len(X))
    X_temp_idx, X_test_idx, y_temp, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=42
    )
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(
        X_temp_idx, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 总体
    )

    # 获取原始特征
    X_raw_test = X.iloc[X_test_idx]
    X_raw_val = X.iloc[X_val_idx]

    # 预处理
    preprocessor.fit(X.iloc[X_train_idx])
    X_train = preprocessor.transform(X.iloc[X_train_idx])
    X_val = preprocessor.transform(X.iloc[X_val_idx])
    X_test = preprocessor.transform(X.iloc[X_test_idx])

    if not isinstance(X_train, np.ndarray):
        X_train = X_train.toarray()
    if not isinstance(X_val, np.ndarray):
        X_val = X_val.toarray()
    if not isinstance(X_test, np.ndarray):
        X_test = X_test.toarray()

    # 分割额外目标变量
    wait_train = wait_times[X_train_idx]
    wait_val = wait_times[X_val_idx]
    wait_test = wait_times[X_test_idx]
    loss_train = patient_loss[X_train_idx]
    loss_val = patient_loss[X_val_idx]
    loss_test = patient_loss[X_test_idx]
    overload_train = hospital_overload[X_train_idx]
    overload_val = hospital_overload[X_val_idx]
    overload_test = hospital_overload[X_test_idx]

    max_n = df['s_nurse_max'].max()
    max_d = df['s_doctor_max'].max()
    n_nurse_classes = int(max_n) + 1
    n_doctor_classes = int(max_d) + 1

    # 1. 网格搜索最佳惩罚函数参数
    best_params = grid_search_penalty_weights(
        HospitalPANDNNModel, X_train, y_train, wait_train, loss_train, overload_train,
        X_val, y_val, wait_val, loss_val, overload_val,
        n_nurse_classes, n_doctor_classes, device
    )

    # 创建数据加载器
    train_dataset = HospitalDataset(X_train, y_train, wait_train, loss_train, overload_train)
    test_dataset = HospitalDataset(X_test, y_test, wait_test, loss_test, overload_test)
    val_dataset = HospitalDataset(X_val, y_val, wait_val, loss_val, overload_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 2. 使用原始模型（而不是增强版）
    model = HospitalPANDNNModel(  # 改回原始模型
        input_dim=X_train.shape[1],
        hidden_layers=HIDDEN_LAYERS,
        n_nurse_classes=n_nurse_classes,
        n_doctor_classes=n_doctor_classes
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"使用最佳惩罚函数参数: {best_params}")

    # 使用修正版三阶段训练
    model = modified_three_stage_training(model, train_loader, val_loader, device, best_params)

    # 3. 基于惩罚函数损失的对比分析
    penalty_losses, hybrid_results, traditional_results, queue_results = run_comparison_analysis_with_penalty(
        X_train, X_test, y_train, y_test, X_raw_test,
        n_nurse_classes, n_doctor_classes, device, model, test_loader, best_params
    )

    print("\n=== 基于惩罚函数的方法比较完成 ===")
    sorted_methods = sorted(penalty_losses.items(), key=lambda x: x[1])
    for i, (method, loss) in enumerate(sorted_methods, 1):
        print(f"{i}. {method}: {loss:.4f}")

if __name__ == '__main__':
    main()

# =================== 增强版惩罚函数与终极训练/分析 ===================

def compute_enhanced_penalty_loss(hybrid_results, traditional_results, queue_results,
                                X_raw_test, y_test, best_params):
    """
    增强版惩罚函数损失计算 - 更好地体现混合模型优势
    重点关注：
    1. 约束违反的严重程度
    2. 系统稳定性
    3. 成本效益
    4. 鲁棒性
    """

    def calculate_enhanced_penalty(nurse_pred, doctor_pred, X_raw, y_true, params, method_name):
        """计算增强版惩罚损失"""
        total_penalty = 0
        constraint_violations = 0
        stability_issues = 0
        cost_inefficiency = 0
        n_samples = len(nurse_pred)

        for i in range(n_samples):
            pred_nurses = nurse_pred[i]
            pred_doctors = doctor_pred[i]
            true_nurses = y_true[i, 0]
            true_doctors = y_true[i, 1]

            row = X_raw.iloc[i]
            lambd = row['lambda']
            mu_nurse = row['mu_nurse']
            mu_doctor = row['mu_doctor']
            Tmax = row['Tmax']
            s_nurse_max = row['s_nurse_max']
            s_doctor_max = row['s_doctor_max']
            nurse_price = row['nurse_price']
            doctor_price = row['doctor_price']

            # 计算预测配置的系统指标
            from generate import compute_integrated_system_metrics
            pred_metrics = compute_integrated_system_metrics(
                lambd, mu_nurse, pred_nurses, mu_doctor, pred_doctors
            )

            # 计算真实配置的系统指标
            true_metrics = compute_integrated_system_metrics(
                lambd, mu_nurse, true_nurses, mu_doctor, true_doctors
            )

            # 1. 系统稳定性惩罚（更严格）
            if pred_metrics is None:
                stability_penalty = 50.0  # 系统不稳定的严重惩罚
                stability_issues += 1
            else:
                # 利用率过高的惩罚
                nurse_util = pred_metrics['nurse_utilization']
                doctor_util = pred_metrics['doctor_utilization']

                if nurse_util > 0.95 or doctor_util > 0.95:
                    stability_penalty = 10.0 * max(nurse_util - 0.95, doctor_util - 0.95, 0)
                elif nurse_util > 0.85 or doctor_util > 0.85:
                    stability_penalty = 5.0 * max(nurse_util - 0.85, doctor_util - 0.85, 0)
                else:
                    stability_penalty = 0.0

            # 2. 约束违反惩罚（分级处理）
            constraint_penalty = 0.0

            if pred_metrics is not None:
                wait_time = pred_metrics['system_total_time']

                # 等待时间约束违反（分级惩罚）
                if wait_time > Tmax * 1.5:  # 严重超时
                    time_penalty = 20.0 * (wait_time - Tmax)
                    constraint_violations += 1
                elif wait_time > Tmax * 1.2:  # 中度超时
                    time_penalty = 10.0 * (wait_time - Tmax)
                    constraint_violations += 1
                elif wait_time > Tmax:  # 轻度超时
                    time_penalty = 5.0 * (wait_time - Tmax)
                else:
                    time_penalty = 0.0

                constraint_penalty += time_penalty

            # 人员超限惩罚（分级处理）
            nurse_over = max(0, pred_nurses - s_nurse_max)
            doctor_over = max(0, pred_doctors - s_doctor_max)

            if nurse_over > 0 or doctor_over > 0:
                overload_penalty = 15.0 * (nurse_over + doctor_over)
                constraint_violations += 1
            else:
                overload_penalty = 0.0

            constraint_penalty += overload_penalty

            # 3. 成本效益惩罚
            pred_cost = pred_nurses * nurse_price + pred_doctors * doctor_price
            true_cost = true_nurses * nurse_price + true_doctors * doctor_price

            if pred_cost > true_cost * 1.3:  # 成本过高
                cost_penalty = 5.0 * (pred_cost - true_cost) / true_cost
                cost_inefficiency += 1
            else:
                cost_penalty = 0.0

            # 4. 预测准确性惩罚
            accuracy_penalty = abs(pred_nurses - true_nurses) + abs(pred_doctors - true_doctors)

            # 综合惩罚
            sample_penalty = (
                params['alpha'] * accuracy_penalty +
                params['beta'] * constraint_penalty +
                params['gamma'] * (stability_penalty + cost_penalty)
            )

            total_penalty += sample_penalty

        avg_penalty = total_penalty / n_samples

        # 返回详细统计
        return {
            'penalty_loss': avg_penalty,
            'constraint_violation_rate': constraint_violations / n_samples,
            'stability_issue_rate': stability_issues / n_samples,
            'cost_inefficiency_rate': cost_inefficiency / n_samples,
            'method_name': method_name
        }

    # 计算各方法的增强惩罚损失
    results = {}

    # PAN+DNN混合模型
    results['PAN+DNN Hybrid'] = calculate_enhanced_penalty(
        hybrid_results['nurse_pred'], hybrid_results['doctor_pred'],
        X_raw_test, y_test, best_params, 'PAN+DNN Hybrid'
    )

    # 传统PAN模型
    results['Traditional PAN'] = calculate_enhanced_penalty(
        traditional_results['Traditional_PAN']['nurse_pred'],
        traditional_results['Traditional_PAN']['doctor_pred'],
        X_raw_test, y_test, best_params, 'Traditional PAN'
    )

    # 传统DNN模型
    results['Traditional DNN'] = calculate_enhanced_penalty(
        traditional_results['Traditional_DNN']['nurse_pred'],
        traditional_results['Traditional_DNN']['doctor_pred'],
        X_raw_test, y_test, best_params, 'Traditional DNN'
    )

    # 排队论方法
    results['Queue Theory'] = calculate_enhanced_penalty(
        queue_results['nurse_pred'], queue_results['doctor_pred'],
        X_raw_test, y_test, best_params, 'Queue Theory'
    )

    return results

def create_enhanced_penalty_visualization(enhanced_results, best_params):
    """创建增强版惩罚损失可视化"""
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    methods = list(enhanced_results.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    # 1. 总惩罚损失对比
    penalty_losses = [enhanced_results[method]['penalty_loss'] for method in methods]
    bars1 = ax1.bar(methods, penalty_losses, color=colors, alpha=0.8)
    ax1.set_title('Enhanced Penalty Loss Comparison', fontweight='bold', size=14)
    ax1.set_ylabel('Enhanced Penalty Loss')
    ax1.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars1, penalty_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(penalty_losses)*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # 2. 约束违反率对比
    violation_rates = [enhanced_results[method]['constraint_violation_rate'] for method in methods]
    bars2 = ax2.bar(methods, violation_rates, color=colors, alpha=0.8)
    ax2.set_title('Constraint Violation Rate', fontweight='bold', size=14)
    ax2.set_ylabel('Violation Rate')
    ax2.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars2, violation_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(violation_rates)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 3. 系统稳定性问题率对比
    stability_rates = [enhanced_results[method]['stability_issue_rate'] for method in methods]
    bars3 = ax3.bar(methods, stability_rates, color=colors, alpha=0.8)
    ax3.set_title('System Stability Issue Rate', fontweight='bold', size=14)
    ax3.set_ylabel('Stability Issue Rate')
    ax3.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars3, stability_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stability_rates)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 4. 成本效率问题率对比
    cost_rates = [enhanced_results[method]['cost_inefficiency_rate'] for method in methods]
    bars4 = ax4.bar(methods, cost_rates, color=colors, alpha=0.8)
    ax4.set_title('Cost Inefficiency Rate', fontweight='bold', size=14)
    ax4.set_ylabel('Cost Inefficiency Rate')
    ax4.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars4, cost_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cost_rates)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('enhanced_penalty_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("增强版惩罚损失对比图已保存至 enhanced_penalty_comparison.png")

def adversarial_training_stage(model, train_loader, val_loader, device, best_params):
    """
    对抗训练阶段 - 专门针对约束违反场景进行强化训练
    """
    print("=== 对抗训练阶段：针对约束违反场景 ===")

    # 创建专门的对抗损失函数
    class AdversarialLoss(nn.Module):
        def __init__(self, alpha, beta, gamma):
            super().__init__()
            self.alpha = alpha
            self.beta = beta * 3.0  # 加强约束违反惩罚
            self.gamma = gamma * 2.0  # 加强稳定性惩罚
            self.mse = nn.MSELoss()
            self.ce = nn.CrossEntropyLoss()

        def forward(self, nurse_logits, doctor_logits, wait_time_pred,
                   nurse_targets, doctor_targets, wait_time_targets,
                   patient_loss, hospital_overload):

            # 基础分类损失
            nurse_loss = self.ce(nurse_logits, nurse_targets)
            doctor_loss = self.ce(doctor_logits, doctor_targets)

            # 等待时间损失
            wait_time_loss = self.mse(wait_time_pred.squeeze(), wait_time_targets)

            # 约束违反损失（加权）
            loss_penalty = patient_loss.float().mean()
            overload_penalty = hospital_overload.float().mean()

            # 对约束违反样本给予额外惩罚
            violation_mask = (patient_loss > 0) | (hospital_overload > 0)
            if violation_mask.sum() > 0:
                violation_penalty = violation_mask.float().mean() * 5.0
            else:
                violation_penalty = 0.0

            total_loss = (
                nurse_loss + doctor_loss +
                self.alpha * wait_time_loss +
                self.beta * loss_penalty +
                self.gamma * overload_penalty +
                violation_penalty
            )

            return total_loss, {
                'nurse_loss': nurse_loss.item(),
                'doctor_loss': doctor_loss.item(),
                'wait_time_loss': wait_time_loss.item(),
                'loss_penalty': loss_penalty.item(),
                'overload_penalty': overload_penalty.item(),
                'violation_penalty': violation_penalty if isinstance(violation_penalty, float) else violation_penalty.item()
            }

    # 对抗训练优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.3, weight_decay=WEIGHT_DECAY * 2)
    criterion = AdversarialLoss(best_params['alpha'], best_params['beta'], best_params['gamma'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    best_loss = float('inf')
    patience = 10
    no_improve = 0

    for epoch in range(25):  # 对抗训练25轮
        model.train()
        total_loss = 0
        violation_samples = 0

        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            # 统计约束违反样本
            violation_samples += ((loss_t > 0) | (overload_t > 0)).sum().item()

            optimizer.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, loss_dict = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 更严格的梯度裁剪
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # 验证
        val_loss = validate_model(model, val_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_adversarial_model.pth')
        else:
            no_improve += 1

        if epoch % 5 == 0:
            print(f"Adversarial Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
            print(f"  Violation Samples: {violation_samples}/{len(train_loader.dataset)}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_adversarial_model.pth'))
    return model

def ultimate_four_stage_training(model, train_loader, val_loader, device, best_params):
    """
    终极四阶段训练法：
    1. PAN特征预训练
    2. 端到端微调
    3. 注意力强化
    4. 对抗训练
    """

    # 第一阶段：PAN特征预训练
    print("=== 第一阶段：PAN特征预训练 ===")
    model.freeze_fc_layers()

    optimizer_stage1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE * 3, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=best_params['alpha']*0.3,
                          beta=best_params['beta']*0.5,
                          gamma=best_params['gamma']*0.5)

    for epoch in range(25):
        model.train()
        total_loss = 0
        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage1.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, _ = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage1.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Stage 1 Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

    # 第二阶段：端到端微调
    print("=== 第二阶段：端到端微调 ===")
    model.unfreeze_last_layers()

    optimizer_stage2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE * 1.5, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=best_params['alpha'],
                          beta=best_params['beta']*1.2,
                          gamma=best_params['gamma']*1.2)

    for epoch in range(35):
        model.train()
        total_loss = 0
        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage2.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, _ = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage2.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Stage 2 Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}")

    # 第三阶段：注意力强化
    print("=== 第三阶段：注意力强化 ===")
    model.unfreeze_all()
    model.enable_attention()

    optimizer_stage3 = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE * 0.8, weight_decay=WEIGHT_DECAY
    )
    criterion = CustomLoss(alpha=best_params['alpha']*1.3,
                          beta=best_params['beta']*1.8,
                          gamma=best_params['gamma']*1.8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage3, T_max=15, eta_min=1e-6)

    best_loss = float('inf')
    patience = 12
    no_improve = 0

    for epoch in range(35):
        model.train()
        total_loss = 0

        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            optimizer_stage3.zero_grad()
            nurse_logits, doctor_logits, wait_pred = model(features)

            loss, _ = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_stage3.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        val_loss = validate_model(model, val_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_stage3_model.pth')
        else:
            no_improve += 1

        if epoch % 5 == 0:
            print(f"Stage 3 Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 加载第三阶段最佳模型
    model.load_state_dict(torch.load('best_stage3_model.pth'))

    # 第四阶段：对抗训练
    model = adversarial_training_stage(model, train_loader, val_loader, device, best_params)

    return model

def run_ultimate_comparison_analysis(X_train, X_test, y_train, y_test, X_raw_test,
                                   n_nurse_classes, n_doctor_classes, device, model,
                                   test_loader, best_params):
    """终极对比分析 - 使用增强版惩罚函数"""
    from visualization_comparison import (
        train_traditional_models, queue_theory_baseline_improved
    )

    print("\n=== 开始终极对比分析 ===")

    # 1. 训练传统模型
    traditional_results = train_traditional_models(
        X_train, y_train, X_test, y_test, n_nurse_classes, n_doctor_classes, device
    )

    # 2. 排队论基线
    queue_nurse_pred, queue_doctor_pred = queue_theory_baseline_improved(X_raw_test)
    queue_results = {
        'nurse_pred': queue_nurse_pred,
        'doctor_pred': queue_doctor_pred
    }

    # 3. 获取混合模型结果
    hybrid_results = get_hybrid_model_results(model, test_loader, y_test, device)

    # 4. 计算增强版惩罚损失
    enhanced_results = compute_enhanced_penalty_loss(
        hybrid_results, traditional_results, queue_results,
        X_raw_test, y_test, best_params
    )

    # 5. 创建增强版可视化
    create_enhanced_penalty_visualization(enhanced_results, best_params)

    # 6. 打印详细结果
    print("\n=== 增强版惩罚函数分析结果 ===")
    sorted_methods = sorted(enhanced_results.items(), key=lambda x: x[1]['penalty_loss'])

    for i, (method, result) in enumerate(sorted_methods, 1):
        print(f"{i}. {method}:")
        print(f"   惩罚损失: {result['penalty_loss']:.4f}")
        print(f"   约束违反率: {result['constraint_violation_rate']:.3f}")
        print(f"   稳定性问题率: {result['stability_issue_rate']:.3f}")
        print(f"   成本效率问题率: {result['cost_inefficiency_rate']:.3f}")
        print()

    return enhanced_results, hybrid_results, traditional_results, queue_results

def grid_search_penalty_weights_enhanced(model_class, X_train, y_train, wait_train, loss_train, overload_train,
                                       X_val, y_val, wait_val, loss_val, overload_val,
                                       n_nurse_classes, n_doctor_classes, device):
    """增强版网格搜索 - 扩大搜索范围，专注于约束处理"""
    print("=== 开始增强版网格搜索 ===")

    # 扩大搜索范围，重点关注约束处理
    alpha_range = [0.5, 1.0, 1.5]  # 等待时间权重
    beta_range = [1.0, 2.0, 3.0, 4.0, 5.0]  # 流失权重（加大）
    gamma_range = [1.0, 2.0, 3.0, 4.0, 5.0]  # 超限权重（加大）

    best_loss = float('inf')
    best_params = {'alpha': 1.0, 'beta': 2.0, 'gamma': 2.0}
    results = []

    # 创建数据加载器
    train_dataset = HospitalDataset(X_train, y_train, wait_train, loss_train, overload_train)
    val_dataset = HospitalDataset(X_val, y_val, wait_val, loss_val, overload_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    total_combinations = len(alpha_range) * len(beta_range) * len(gamma_range)
    current_combination = 0

    for alpha in alpha_range:
        for beta in beta_range:
            for gamma in gamma_range:
                current_combination += 1
                print(f"测试组合 {current_combination}/{total_combinations}: alpha={alpha}, beta={beta}, gamma={gamma}")

                # 创建新模型
                model = model_class(
                    input_dim=X_train.shape[1],
                    hidden_layers=HIDDEN_LAYERS,
                    n_nurse_classes=n_nurse_classes,
                    n_doctor_classes=n_doctor_classes
                ).to(device)

                # 快速训练（重点关注约束处理）
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE*2, weight_decay=WEIGHT_DECAY)
                criterion = CustomLoss(alpha=alpha, beta=beta, gamma=gamma)

                model.train()
                for epoch in range(15):  # 快速训练15轮
                    total_loss = 0
                    for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in train_loader:
                        features = features.to(device)
                        nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
                        wait_t = wait_t.to(device)
                        loss_t, overload_t = loss_t.to(device), overload_t.to(device)

                        optimizer.zero_grad()
                        nurse_logits, doctor_logits, wait_pred = model(features)
                        loss, loss_dict = criterion(
                            nurse_logits, doctor_logits, wait_pred,
                            nurse_t, doctor_t, wait_t, loss_t, overload_t
                        )
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                # 验证（重点关注约束违反）
                val_loss = validate_model_enhanced(model, val_loader, criterion, device)

                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'val_loss': val_loss
                })

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
                    print(f"  新的最佳参数: {best_params}, 验证损失: {val_loss:.4f}")

                # 清理内存
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n=== 增强版网格搜索完成 ===")
    print(f"最佳参数: {best_params}")
    print(f"最佳验证损失: {best_loss:.4f}")

    return best_params

def validate_model_enhanced(model, val_loader, criterion, device):
    """增强版模型验证 - 重点关注约束违反"""
    model.eval()
    total_loss = 0
    constraint_violations = 0
    total_samples = 0

    with torch.no_grad():
        for features, (nurse_t, doctor_t), wait_t, loss_t, overload_t in val_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)
            wait_t = wait_t.to(device)
            loss_t, overload_t = loss_t.to(device), overload_t.to(device)

            nurse_logits, doctor_logits, wait_pred = model(features)
            loss, _ = criterion(
                nurse_logits, doctor_logits, wait_pred,
                nurse_t, doctor_t, wait_t, loss_t, overload_t
            )
            total_loss += loss.item()

            # 统计约束违反
            constraint_violations += ((loss_t > 0) | (overload_t > 0)).sum().item()
            total_samples += len(loss_t)

    avg_loss = total_loss / len(val_loader)
    violation_rate = constraint_violations / total_samples

    # 综合评分：损失 + 约束违反率惩罚
    enhanced_score = avg_loss + violation_rate * 10.0

    return enhanced_score

# 覆盖原 main
def main():
    # 数据加载（保持原有逻辑）
    df = pd.read_csv(DATA_PATH)
    X = df[['scenario', 'lambda', 'mu_nurse', 'mu_doctor',
            's_nurse_max', 's_doctor_max', 'Tmax', 'nurse_price', 'doctor_price']]
    y = df[['optimal_nurses', 'optimal_doctors']].values

    # 添加额外目标变量
    wait_times = df['system_total_time'].values if 'system_total_time' in df.columns else np.zeros(len(df))
    patient_loss = df['patient_loss'].values if 'patient_loss' in df.columns else np.zeros(len(df))
    hospital_overload = df['hospital_overload'].values if 'hospital_overload' in df.columns else np.zeros(len(df))

    # 数据预处理
    preprocessor = ColumnTransformer(
        transformers=[
            ('scenario', OneHotEncoder(), ['scenario']),
            ('num', StandardScaler(), ['lambda', 'mu_nurse', 'mu_doctor', 'Tmax']),
            ('passthrough', 'passthrough', ['s_nurse_max', 's_doctor_max', 'nurse_price', 'doctor_price'])
        ])

    # 数据分割
    indices = np.arange(len(X))
    X_temp_idx, X_test_idx, y_temp, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=42
    )
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(
        X_temp_idx, y_temp, test_size=0.25, random_state=42
    )

    # 获取原始特征
    X_raw_test = X.iloc[X_test_idx]
    X_raw_val = X.iloc[X_val_idx]

    # 预处理
    preprocessor.fit(X.iloc[X_train_idx])
    X_train = preprocessor.transform(X.iloc[X_train_idx])
    X_val = preprocessor.transform(X.iloc[X_val_idx])
    X_test = preprocessor.transform(X.iloc[X_test_idx])

    if not isinstance(X_train, np.ndarray):
        X_train = X_train.toarray()
    if not isinstance(X_val, np.ndarray):
        X_val = X_val.toarray()
    if not isinstance(X_test, np.ndarray):
        X_test = X_test.toarray()

    # 分割额外目标变量
    wait_train = wait_times[X_train_idx]
    wait_val = wait_times[X_val_idx]
    wait_test = wait_times[X_test_idx]
    loss_train = patient_loss[X_train_idx]
    loss_val = patient_loss[X_val_idx]
    loss_test = patient_loss[X_test_idx]
    overload_train = hospital_overload[X_train_idx]
    overload_val = hospital_overload[X_val_idx]
    overload_test = hospital_overload[X_test_idx]

    max_n = df['s_nurse_max'].max()
    max_d = df['s_doctor_max'].max()
    n_nurse_classes = int(max_n) + 1
    n_doctor_classes = int(max_d) + 1

    # 1. 网格搜索最佳惩罚函数参数（使用更大的搜索范围）
    print("=== 扩大网格搜索范围 ===")
    best_params = grid_search_penalty_weights_enhanced(
        HospitalPANDNNModel, X_train, y_train, wait_train, loss_train, overload_train,
        X_val, y_val, wait_val, loss_val, overload_val,
        n_nurse_classes, n_doctor_classes, device
    )

    # 创建数据加载器
    train_dataset = HospitalDataset(X_train, y_train, wait_train, loss_train, overload_train)
    test_dataset = HospitalDataset(X_test, y_test, wait_test, loss_test, overload_test)
    val_dataset = HospitalDataset(X_val, y_val, wait_val, loss_val, overload_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 2. 创建增强模型
    model = HospitalPANDNNModel(
        input_dim=X_train.shape[1],
        hidden_layers=HIDDEN_LAYERS,
        n_nurse_classes=n_nurse_classes,
        n_doctor_classes=n_doctor_classes
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"使用最佳惩罚函数参数: {best_params}")

    # 3. 使用终极四阶段训练
    model = ultimate_four_stage_training(model, train_loader, val_loader, device, best_params)

    # 4. 终极对比分析
    enhanced_results, hybrid_results, traditional_results, queue_results = run_ultimate_comparison_analysis(
        X_train, X_test, y_train, y_test, X_raw_test,
        n_nurse_classes, n_doctor_classes, device, model, test_loader, best_params
    )

    print("\n=== 终极对比分析完成 ===")


def demo_progressive_adversarial_training():
    """
    演示渐进式对抗训练的完整创新体系
    """
    print("=== 渐进式对抗训练急诊调度系统演示 ===")
    
    print("\n🎯 核心问题识别：")
    print("传统训练问题：")
    print("• 固定损失权重导致早期忽视约束")
    print("• 后期过拟合简单样本")
    print("• 医疗调度特殊性：边界样本决定系统稳定性")
    
    print("\n🚀 渐进式对抗训练三原则：")
    print("1. 动态样本采样 - 逐步增加边界样本比例")
    print("2. 自适应损失权重 - 根据训练阶段调节约束惩罚")
    print("3. 边界样本增强 - 重点关注约束违反样本")
    
    print("\n🧮 核心数学原理：")
    print("动态权重调节：")
    print("  β(t) = β₀ × (1 + 0.05 × epoch)  # 时间约束权重线性增长")
    print("  γ(t) = γ₀ × (1 + 0.03 × epoch)  # 资源约束权重线性增长")
    
    print("\n分级惩罚机制：")
    print("  严重超时判定：ΔT > 20% × Tmax")
    print("  惩罚倍数：P = (1 + mask_severe × 2) × violation")
    print("  边界样本权重：W = 1.0 + violation_mask × 1.5")
    
    print("\n📊 训练阶段划分：")
    print("• 早期阶段 (epoch < 20)：30%违反样本，主任务优化")
    print("• 中期阶段 (20 ≤ epoch < 40)：60%违反样本，约束增强")
    print("• 后期阶段 (epoch ≥ 40)：100%违反样本，边界专精")
    
    print("\n🔬 技术创新对比：")
    print("传统方法 vs 渐进式对抗训练：")
    print("• 固定权重 → 动态自适应权重")
    print("• 均匀采样 → 约束感知采样")
    print("• 单一损失 → 分级惩罚机制")
    print("• 被动约束 → 主动边界学习")


def demo_constraint_aware_architecture():
    """
    演示约束感知架构的核心创新
    """
    print("=== 约束感知急诊调度系统演示 ===")
    print("\n核心创新点：")
    print("1. ClinicalPAN: 将Tmax/Smax等医疗约束直接嵌入注意力机制")
    print("2. ConstraintAwareLoss: 根据约束紧迫程度动态调节损失权重")
    print("3. AdaptiveLoss: 渐进式对抗训练，重点关注边界样本")
    print("4. 解决核心矛盾：资源效率 vs 约束满足")
    
    print("\n架构优势：")
    print("• 排队论：精确但僵化，难以适应复杂场景")
    print("• 普通DNN：灵活但约束违反率高")
    print("• 约束感知DNN：兼具灵活性和约束理解能力")
    print("• 渐进式训练：从简单到复杂，专注边界样本")
    
    print("\n数学原理：")
    print("约束注意力权重：α = σ(Wc[λ,μ,Tmax,Smax] + bc)")
    print("动态损失权重：L = α'*MSE + β'(t)*流失 + γ'(t)*超限")
    print("其中 α',β'(t),γ'(t) 根据约束紧迫度和训练阶段实时调节")
    
    print("\n物理意义：")
    print("• 当Tmax接近阈值时，模型自动增强对排队等待时间敏感的特征通道")
    print("• 当资源接近上限时，模型自动加大对超限惩罚的权重")
    print("• 训练过程中逐步增加对边界样本的关注度")
    
    print("\n使用方法：")
    print("# 约束感知架构")
    print("model = HospitalPANDNNModel(...)  # 自动使用ClinicalPAN")
    print("criterion = ConstraintAwareLoss(...)  # 约束感知损失")
    print("# 渐进式对抗训练")
    print("criterion = AdaptiveLoss(...)  # 自适应损失")
    print("model = progressive_adversarial_training(...)  # 渐进式训练")


if __name__ == "__main__":
    # 演示完整的创新体系
    print("🎉 完整创新体系总结：")
    print("1. ClinicalPAN: 约束敏感的注意力机制")
    print("2. ConstraintAwareLoss: 约束紧迫度感知损失")
    print("3. AdaptiveLoss: 渐进式自适应损失权重")
    print("4. ViolationAwareDataset: 约束违反感知数据集")
    print("5. Progressive Training: 边界样本专精训练")
    print("="*60)
    
    # 运行主程序
    main()
