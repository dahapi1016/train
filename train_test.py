import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import DataLoader, Dataset

# 常量定义
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_LAYERS = [128, 64, 32]
DROPOUT_RATE = 0.2

# 快速测试参数
QUICK_TEST_PARAMS = {'alpha': 0.5, 'beta': 1.0, 'gamma': 1.0}

# 数据集类定义
class HospitalDataset(Dataset):
    def __init__(self, features, targets, wait_times, patient_loss, hospital_overload):
        self.features = torch.FloatTensor(features)
        self.nurse_targets = torch.LongTensor(targets[:, 0])
        self.doctor_targets = torch.LongTensor(targets[:, 1])
        self.wait_times = torch.FloatTensor(wait_times)
        self.patient_loss = torch.FloatTensor(patient_loss)
        self.hospital_overload = torch.FloatTensor(hospital_overload)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            self.features[idx],
            (self.nurse_targets[idx], self.doctor_targets[idx]),
            self.wait_times[idx],
            self.patient_loss[idx],
            self.hospital_overload[idx]
        )

# 简化的PAN层
class SimplePANLayer(nn.Module):
    def __init__(self, input_dim):
        super(SimplePANLayer, self).__init__()
        self.input_dim = input_dim
        
        # 简化的特征变换
        self.transform = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 简化的注意力机制
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 特征变换
        transformed = self.transform(x)
        
        # 注意力权重
        attention_weights = self.attention(x)
        
        # 加权特征
        weighted_features = transformed * attention_weights
        
        # 残差连接
        return x + weighted_features[:, :self.input_dim]

# 简化的混合模型
class SimpleHospitalModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, n_nurse_classes, n_doctor_classes):
        super().__init__()
        self.input_dim = input_dim
        
        # 简化的PAN层
        self.pan_layer = SimplePANLayer(input_dim)
        
        # DNN主干网络
        layers = []
        prev_size = input_dim
        for layer_size in hidden_layers:
            layers += [
                nn.Linear(prev_size, layer_size),
                nn.BatchNorm1d(layer_size),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE)
            ]
            prev_size = layer_size
        self.shared_layers = nn.Sequential(*layers)
        
        # 任务特定头部
        self.nurse_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_nurse_classes)
        )
        self.doctor_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_doctor_classes)
        )
        
        # 等待时间预测头
        self.wait_time_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # PAN特征提取
        pan_out = self.pan_layer(x)
        
        # DNN主干网络
        shared_out = self.shared_layers(pan_out)
        
        # 任务预测
        nurse_logits = self.nurse_head(shared_out)
        doctor_logits = self.doctor_head(shared_out)
        wait_time_pred = self.wait_time_head(shared_out)
        
        return nurse_logits, doctor_logits, wait_time_pred

# 简化的损失函数
class SimpleLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, nurse_logits, doctor_logits, wait_time_pred,
               nurse_targets, doctor_targets, wait_time_targets,
               patient_loss, hospital_overload):
        
        # 基础损失
        nurse_loss = self.ce(nurse_logits, nurse_targets)
        doctor_loss = self.ce(doctor_logits, doctor_targets)
        wait_time_loss = self.mse(wait_time_pred.squeeze(), wait_time_targets)
        
        # 约束损失
        loss_penalty = patient_loss.float().mean()
        overload_penalty = hospital_overload.float().mean()
        
        total_loss = (
            self.alpha * (nurse_loss + doctor_loss + wait_time_loss) +
            self.beta * loss_penalty +
            self.gamma * overload_penalty
        )
        
        return total_loss, {
            'nurse_loss': nurse_loss.item(),
            'doctor_loss': doctor_loss.item(),
            'wait_time_loss': wait_time_loss.item(),
            'loss_penalty': loss_penalty.item(),
            'overload_penalty': overload_penalty.item()
        }

def quick_train_model(model, train_loader, val_loader, device, params, epochs=30):
    """快速训练模型"""
    print(f"=== 快速训练开始 (epochs={epochs}) ===")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = SimpleLoss(alpha=params['alpha'], beta=params['beta'], gamma=params['gamma'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    best_loss = float('inf')
    patience = 10
    no_improve = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
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
                val_loss += loss.item()
        
        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_quick_model.pth')
        else:
            no_improve += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_quick_model.pth'))
    print("=== 快速训练完成 ===")
    return model

def get_model_results(model, test_loader, y_test, device):
    """获取模型结果"""
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

def quick_comparison_analysis(hybrid_results, X_raw_test, y_test, params):
    """快速对比分析"""
    print("\n=== 快速对比分析 ===")
    
    # 简化的排队论基线
    def simple_queue_baseline(X_raw):
        nurse_pred = []
        doctor_pred = []
        
        for _, row in X_raw.iterrows():
            # 简单的启发式规则
            lambda_val = row['lambda']
            mu_nurse = row['mu_nurse']
            mu_doctor = row['mu_doctor']
            
            # 基于利用率的简单配置
            nurses = max(1, int(lambda_val / mu_nurse) + 1)
            doctors = max(1, int(lambda_val / mu_doctor) + 1)
            
            # 限制在最大值内
            nurses = min(nurses, row['s_nurse_max'])
            doctors = min(doctors, row['s_doctor_max'])
            
            nurse_pred.append(nurses)
            doctor_pred.append(doctors)
        
        return np.array(nurse_pred), np.array(doctor_pred)
    
    # 获取排队论结果
    queue_nurse_pred, queue_doctor_pred = simple_queue_baseline(X_raw_test)
    
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
    
    # 打印对比结果
    print("\n护士预测结果对比:")
    print(f"PAN+DNN Hybrid - MAE: {hybrid_results['nurse_metrics']['MAE']:.3f}, Accuracy: {hybrid_results['nurse_metrics']['Accuracy']:.3f}")
    print(f"Queue Theory   - MAE: {queue_results['nurse_metrics']['MAE']:.3f}, Accuracy: {queue_results['nurse_metrics']['Accuracy']:.3f}")
    
    print("\n医生预测结果对比:")
    print(f"PAN+DNN Hybrid - MAE: {hybrid_results['doctor_metrics']['MAE']:.3f}, Accuracy: {hybrid_results['doctor_metrics']['Accuracy']:.3f}")
    print(f"Queue Theory   - MAE: {queue_results['doctor_metrics']['MAE']:.3f}, Accuracy: {queue_results['doctor_metrics']['Accuracy']:.3f}")
    
    # 简化的约束违反分析
    def calculate_violations(nurse_pred, doctor_pred, X_raw):
        violations = 0
        for i in range(len(nurse_pred)):
            row = X_raw.iloc[i]
            if (nurse_pred[i] > row['s_nurse_max'] or doctor_pred[i] > row['s_doctor_max']):
                violations += 1
        return violations / len(nurse_pred)
    
    hybrid_violations = calculate_violations(hybrid_results['nurse_pred'], hybrid_results['doctor_pred'], X_raw_test)
    queue_violations = calculate_violations(queue_results['nurse_pred'], queue_results['doctor_pred'], X_raw_test)
    
    print(f"\n约束违反率对比:")
    print(f"PAN+DNN Hybrid: {hybrid_violations:.3f}")
    print(f"Queue Theory:   {queue_violations:.3f}")
    
    return hybrid_results, queue_results

def main_quick_test():
    """快速测试主函数"""
    print("=== 快速测试模式 ===")
    print(f"使用固定参数: {QUICK_TEST_PARAMS}")
    
    # 数据加载
    try:
        df = pd.read_csv('queue_theory_challenging_data.csv')
        print("加载现有的对抗性数据集")
    except FileNotFoundError:
        print("生成新的对抗性数据集...")
        from generate_adversarial import generate_queue_theory_challenging_data
        df = generate_queue_theory_challenging_data(n_samples=2000)  # 减少样本数量加快测试
        df.to_csv('queue_theory_challenging_data.csv', index=False)
        print("对抗性数据集生成完成")
    
    # 为了快速测试，只使用部分数据
    df = df.sample(n=min(2000, len(df)), random_state=42)
    print(f"使用 {len(df)} 个样本进行快速测试")
    
    # 数据预处理
    feature_columns = ['scenario', 'lambda', 'mu_nurse', 'mu_doctor',
                      's_nurse_max', 's_doctor_max', 'Tmax', 'nurse_price', 'doctor_price']
    
    # 添加额外特征（如果存在）
    if 'cv_nurse' in df.columns:
        feature_columns.extend(['cv_nurse', 'cv_doctor'])
    if 'correlation' in df.columns:
        feature_columns.append('correlation')
    
    X = df[feature_columns]
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
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_dataset = HospitalDataset(X_train, y_train, wait_train, loss_train, overload_train)
    test_dataset = HospitalDataset(X_test, y_test, wait_test, loss_test, overload_test)
    val_dataset = HospitalDataset(X_val, y_val, wait_val, loss_val, overload_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 创建模型
    model = SimpleHospitalModel(
        input_dim=X_train.shape[1],
        hidden_layers=HIDDEN_LAYERS,
        n_nurse_classes=n_nurse_classes,
        n_doctor_classes=n_doctor_classes
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 快速训练
    model = quick_train_model(model, train_loader, val_loader, device, QUICK_TEST_PARAMS, epochs=20)
    
    # 获取结果
    hybrid_results = get_model_results(model, test_loader, y_test, device)
    
    # 快速对比分析
    hybrid_results, queue_results = quick_comparison_analysis(hybrid_results, X_raw_test, y_test, QUICK_TEST_PARAMS)
    
    print("\n=== 快速测试完成 ===")
    return hybrid_results, queue_results

if __name__ == "__main__":
    # 运行快速测试
    hybrid_results, queue_results = main_quick_test()

