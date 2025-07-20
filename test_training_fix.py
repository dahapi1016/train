#!/usr/bin/env python3
"""
简化的训练测试脚本 - 验证改进效果
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class SimpleHospitalDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.nurse_targets = torch.LongTensor(targets[:, 0])
        self.doctor_targets = torch.LongTensor(targets[:, 1])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], (self.nurse_targets[idx], self.doctor_targets[idx])

class SimpleModel(nn.Module):
    def __init__(self, input_dim, n_nurse_classes, n_doctor_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.nurse_head = nn.Linear(64, n_nurse_classes)
        self.doctor_head = nn.Linear(64, n_doctor_classes)

    def forward(self, x):
        shared_out = self.shared(x)
        nurse_logits = self.nurse_head(shared_out)
        doctor_logits = self.doctor_head(shared_out)
        return nurse_logits, doctor_logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def preprocess_data():
    """预处理数据"""
    print("=== 数据预处理 ===")

    # 读取数据
    df = pd.read_csv('emergency_hospital_data.csv')
    print(f"原始数据量: {len(df)}")

    # 准备特征和目标
    feature_columns = ['lambda', 'mu_nurse', 'mu_doctor', 'Tmax',
                      's_nurse_max', 's_doctor_max', 'nurse_price', 'doctor_price']
    X = df[feature_columns].values
    y = df[['optimal_nurses', 'optimal_doctors']].values

    # 分析类别分布
    nurse_counts = pd.Series(y[:, 0]).value_counts()
    doctor_counts = pd.Series(y[:, 1]).value_counts()

    print(f"护士类别数: {len(nurse_counts)}, 不平衡比例: {nurse_counts.max()/nurse_counts.min():.1f}:1")
    print(f"医生类别数: {len(doctor_counts)}, 不平衡比例: {doctor_counts.max()/doctor_counts.min():.1f}:1")

    # 移除少样本类别
    min_samples = 5
    valid_nurse_classes = nurse_counts[nurse_counts >= min_samples].index.tolist()
    valid_doctor_classes = doctor_counts[doctor_counts >= min_samples].index.tolist()

    # 过滤数据
    mask = []
    for i in range(len(y)):
        if y[i, 0] in valid_nurse_classes and y[i, 1] in valid_doctor_classes:
            mask.append(True)
        else:
            mask.append(False)

    mask = np.array(mask)
    X_filtered = X[mask]
    y_filtered = y[mask]

    # 重新映射标签
    nurse_mapping = {old: new for new, old in enumerate(sorted(valid_nurse_classes))}
    doctor_mapping = {old: new for new, old in enumerate(sorted(valid_doctor_classes))}

    for i in range(len(y_filtered)):
        y_filtered[i, 0] = nurse_mapping[y_filtered[i, 0]]
        y_filtered[i, 1] = doctor_mapping[y_filtered[i, 1]]

    n_nurse_classes = len(valid_nurse_classes)
    n_doctor_classes = len(valid_doctor_classes)

    print(f"过滤后数据量: {len(X_filtered)}")
    print(f"新的护士类别数: {n_nurse_classes}")
    print(f"新的医生类别数: {n_doctor_classes}")

    return X_filtered, y_filtered, n_nurse_classes, n_doctor_classes

def compute_class_weights(train_loader, target_type, n_classes):
    """计算类别权重"""
    class_counts = torch.zeros(n_classes)

    for _, (nurse_t, doctor_t) in train_loader:
        targets = nurse_t if target_type == 'nurse' else doctor_t
        for i in range(n_classes):
            class_counts[i] += (targets == i).sum().item()

    class_counts = torch.clamp(class_counts, min=1)
    total_samples = class_counts.sum()
    weights = torch.sqrt(total_samples / (n_classes * class_counts))

    # 限制权重比例
    max_weight = weights.max()
    min_weight = weights.min()
    if max_weight / min_weight > 10:
        weights = torch.clamp(weights, max=min_weight * 10)

    weights = weights / weights.sum() * n_classes
    return weights

def train_model(model, train_loader, val_loader, device, n_nurse_classes, n_doctor_classes):
    """训练模型"""
    print("=== 开始训练 ===")

    # 计算类别权重
    nurse_weights = compute_class_weights(train_loader, 'nurse', n_nurse_classes)
    doctor_weights = compute_class_weights(train_loader, 'doctor', n_doctor_classes)

    print(f"护士权重范围: {nurse_weights.min():.3f} - {nurse_weights.max():.3f}")
    print(f"医生权重范围: {doctor_weights.min():.3f} - {doctor_weights.max():.3f}")

    # 使用Focal Loss
    nurse_criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=nurse_weights.to(device))
    doctor_criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=doctor_weights.to(device))

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_acc = 0.0
    patience = 15
    no_improve = 0

    for epoch in range(50):
        # 训练
        model.train()
        total_loss = 0
        correct_nurse = 0
        correct_doctor = 0
        total_samples = 0

        for features, (nurse_t, doctor_t) in train_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)

            optimizer.zero_grad()
            nurse_logits, doctor_logits = model(features)

            nurse_loss = nurse_criterion(nurse_logits, nurse_t)
            doctor_loss = doctor_criterion(doctor_logits, doctor_t)
            loss = nurse_loss + doctor_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

            # 计算准确率
            nurse_pred = torch.argmax(nurse_logits, dim=1)
            doctor_pred = torch.argmax(doctor_logits, dim=1)
            correct_nurse += (nurse_pred == nurse_t).sum().item()
            correct_doctor += (doctor_pred == doctor_t).sum().item()
            total_samples += nurse_t.size(0)

        # 验证
        model.eval()
        val_correct_nurse = 0
        val_correct_doctor = 0
        val_total = 0

        with torch.no_grad():
            for features, (nurse_t, doctor_t) in val_loader:
                features = features.to(device)
                nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)

                nurse_logits, doctor_logits = model(features)
                nurse_pred = torch.argmax(nurse_logits, dim=1)
                doctor_pred = torch.argmax(doctor_logits, dim=1)

                val_correct_nurse += (nurse_pred == nurse_t).sum().item()
                val_correct_doctor += (doctor_pred == doctor_t).sum().item()
                val_total += nurse_t.size(0)

        # 计算准确率
        train_acc_nurse = correct_nurse / total_samples
        train_acc_doctor = correct_doctor / total_samples
        val_acc_nurse = val_correct_nurse / val_total
        val_acc_doctor = val_correct_doctor / val_total

        avg_val_acc = (val_acc_nurse + val_acc_doctor) / 2
        scheduler.step(avg_val_acc)

        # 早停
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            no_improve = 0
            torch.save(model.state_dict(), 'best_simple_model.pth')
        else:
            no_improve += 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train Acc - Nurse: {train_acc_nurse:.3f}, Doctor: {train_acc_doctor:.3f}")
            print(f"  Val Acc - Nurse: {val_acc_nurse:.3f}, Doctor: {val_acc_doctor:.3f}")
            print(f"  Avg Val Acc: {avg_val_acc:.3f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_simple_model.pth'))
    return model, best_acc

def main():
    print("=== 简化训练测试 ===")

    # 预处理数据
    X, y, n_nurse_classes, n_doctor_classes = preprocess_data()

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # 创建数据加载器
    train_dataset = SimpleHospitalDataset(X_train, y_train)
    val_dataset = SimpleHospitalDataset(X_val, y_val)
    test_dataset = SimpleHospitalDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(X_train.shape[1], n_nurse_classes, n_doctor_classes).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"使用设备: {device}")

    # 训练模型
    model, best_acc = train_model(model, train_loader, val_loader, device, n_nurse_classes, n_doctor_classes)

    # 最终测试
    model.eval()
    test_correct_nurse = 0
    test_correct_doctor = 0
    test_total = 0

    with torch.no_grad():
        for features, (nurse_t, doctor_t) in test_loader:
            features = features.to(device)
            nurse_t, doctor_t = nurse_t.to(device), doctor_t.to(device)

            nurse_logits, doctor_logits = model(features)
            nurse_pred = torch.argmax(nurse_logits, dim=1)
            doctor_pred = torch.argmax(doctor_logits, dim=1)

            test_correct_nurse += (nurse_pred == nurse_t).sum().item()
            test_correct_doctor += (doctor_pred == doctor_t).sum().item()
            test_total += nurse_t.size(0)

    test_acc_nurse = test_correct_nurse / test_total
    test_acc_doctor = test_correct_doctor / test_total
    avg_test_acc = (test_acc_nurse + test_acc_doctor) / 2

    print(f"\n=== 最终测试结果 ===")
    print(f"护士预测准确率: {test_acc_nurse:.3f}")
    print(f"医生预测准确率: {test_acc_doctor:.3f}")
    print(f"平均准确率: {avg_test_acc:.3f}")

    print(f"\n=== 改进效果 ===")
    print(f"原始准确率: 0.333")
    print(f"改进后准确率: {avg_test_acc:.3f}")
    print(f"提升幅度: {(avg_test_acc - 0.333) / 0.333 * 100:.1f}%")

    if avg_test_acc > 0.5:
        print("✅ 准确率显著提升！改进方案有效")
    elif avg_test_acc > 0.4:
        print("✅ 准确率有所提升，改进方案部分有效")
    else:
        print("❌ 准确率提升有限，需要进一步优化")

if __name__ == "__main__":
    main()

