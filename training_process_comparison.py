import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

warnings.filterwarnings('ignore')

# 设置中文字体和样式
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
sns.set_style("whitegrid")

# 如果上面的设置不行，尝试这个
try:
    import platform
    system = platform.system()
    if system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica', 'DejaVu Sans']
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Liberation Sans']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['figure.figsize'] = (12, 8)

class TraditionalPANModel(nn.Module):
    """传统PAN模型"""
    def __init__(self, input_dim, n_nurse_classes, n_doctor_classes):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        self.nurse_head = nn.Linear(input_dim, n_nurse_classes)
        self.doctor_head = nn.Linear(input_dim, n_doctor_classes)

    def forward(self, x):
        attention_weights = self.attention(x)
        attended_x = x * attention_weights
        nurse_logits = self.nurse_head(attended_x)
        doctor_logits = self.doctor_head(attended_x)
        return nurse_logits, doctor_logits

class TraditionalDNNModel(nn.Module):
    """传统DNN模型"""
    def __init__(self, input_dim, hidden_layers, n_nurse_classes, n_doctor_classes):
        super().__init__()
        layers = []
        prev_size = input_dim
        for layer_size in hidden_layers:
            layers += [
                nn.Linear(prev_size, layer_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ]
            prev_size = layer_size

        self.shared_layers = nn.Sequential(*layers)
        self.nurse_head = nn.Linear(prev_size, n_nurse_classes)
        self.doctor_head = nn.Linear(prev_size, n_doctor_classes)

    def forward(self, x):
        shared_out = self.shared_layers(x)
        nurse_logits = self.nurse_head(shared_out)
        doctor_logits = self.doctor_head(shared_out)
        return nurse_logits, doctor_logits

class SimpleHybridModel(nn.Module):
    """简化混合模型"""
    def __init__(self, input_dim, n_nurse_classes, n_doctor_classes):
        super().__init__()
        # 简化的PAN层
        self.pan_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # DNN层
        self.dnn_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 输出头
        self.nurse_head = nn.Linear(64, n_nurse_classes)
        self.doctor_head = nn.Linear(64, n_doctor_classes)

    def forward(self, x):
        # PAN处理
        attention_weights = self.pan_attention(x)
        attended_x = x * attention_weights
        
        # DNN处理
        dnn_out = self.dnn_layers(attended_x)
        
        # 输出
        nurse_logits = self.nurse_head(dnn_out)
        doctor_logits = self.doctor_head(dnn_out)
        return nurse_logits, doctor_logits

def train_model_with_tracking(model, X_train, y_train, X_val, y_val, device, model_name, epochs=80):
    """训练模型并跟踪训练过程"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 数据转换
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # 训练跟踪
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        nurse_logits, doctor_logits = model(X_train_tensor)
        
        loss = (criterion(nurse_logits, y_train_tensor[:, 0]) +
                criterion(doctor_logits, y_train_tensor[:, 1]))
        
        loss.backward()
        optimizer.step()
        
        # 计算训练准确率
        with torch.no_grad():
            train_nurse_pred = torch.argmax(nurse_logits, dim=1)
            train_doctor_pred = torch.argmax(doctor_logits, dim=1)
            train_acc = (accuracy_score(y_train_tensor[:, 0].cpu(), train_nurse_pred.cpu()) +
                        accuracy_score(y_train_tensor[:, 1].cpu(), train_doctor_pred.cpu())) / 2
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_nurse_logits, val_doctor_logits = model(X_val_tensor)
            val_loss = (criterion(val_nurse_logits, y_val_tensor[:, 0]) +
                       criterion(val_doctor_logits, y_val_tensor[:, 1]))
            
            val_nurse_pred = torch.argmax(val_nurse_logits, dim=1)
            val_doctor_pred = torch.argmax(val_doctor_logits, dim=1)
            val_acc = (accuracy_score(y_val_tensor[:, 0].cpu(), val_nurse_pred.cpu()) +
                      accuracy_score(y_val_tensor[:, 1].cpu(), val_doctor_pred.cpu())) / 2
        
        # 记录
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        if epoch % 10 == 0:
            print(f"{model_name} Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
            print(f"  Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'model': model
    }

def create_training_process_comparison():
    """创建训练过程对比图"""
    print("开始训练过程对比分析...")
    
    # 加载数据
    try:
        df = pd.read_csv('emergency_hospital_data.csv')
    except:
        print("使用增强数据集...")
        df = pd.read_csv('emergency_hospital_data_enhanced.csv')
    
    # 数据预处理
    feature_columns = ['lambda', 'mu_nurse', 'mu_doctor', 's_nurse_max', 's_doctor_max', 
                      'Tmax', 'nurse_price', 'doctor_price']
    target_columns = ['optimal_nurses', 'optimal_doctors']
    
    X = df[feature_columns].values
    y = df[target_columns].values
    
    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 将连续目标转换为分类问题
    from sklearn.preprocessing import LabelEncoder
    y_nurse_encoded = LabelEncoder().fit_transform(y[:, 0])
    y_doctor_encoded = LabelEncoder().fit_transform(y[:, 1])
    y_encoded = np.column_stack([y_nurse_encoded, y_doctor_encoded])
    
    # 划分数据集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取类别数
    n_nurse_classes = len(np.unique(y_encoded[:, 0]))
    n_doctor_classes = len(np.unique(y_encoded[:, 1]))
    
    # 训练不同模型
    models = {
        'Traditional_PAN': TraditionalPANModel(X_train.shape[1], n_nurse_classes, n_doctor_classes),
        'Traditional_DNN': TraditionalDNNModel(X_train.shape[1], [128, 64], n_nurse_classes, n_doctor_classes),
        'Hybrid_PAN_DNN': SimpleHybridModel(X_train.shape[1], n_nurse_classes, n_doctor_classes)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n开始训练 {model_name}...")
        model = model.to(device)
        results[model_name] = train_model_with_tracking(
            model, X_train, y_train, X_val, y_val, device, model_name, epochs=80
        )
    
    # 创建可视化
    create_training_curves_comparison(results)
    create_convergence_analysis(results)
    create_model_complexity_comparison(results)
    create_performance_summary_table(results, X_test, y_test, device)
    
    return results

def create_training_curves_comparison(results):
    """创建训练曲线对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('不同模型架构训练过程对比', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    model_names = list(results.keys())
    
    # 训练损失对比
    ax1 = axes[0, 0]
    for i, model_name in enumerate(model_names):
        ax1.plot(results[model_name]['train_losses'], 
                color=colors[i], linewidth=2, label=model_name, alpha=0.8)
    ax1.set_title('训练损失对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 验证损失对比
    ax2 = axes[0, 1]
    for i, model_name in enumerate(model_names):
        ax2.plot(results[model_name]['val_losses'], 
                color=colors[i], linewidth=2, label=model_name, alpha=0.8)
    ax2.set_title('验证损失对比', fontsize=14, fontweight='bold')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('损失值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 训练准确率对比
    ax3 = axes[1, 0]
    for i, model_name in enumerate(model_names):
        ax3.plot(results[model_name]['train_accuracies'], 
                color=colors[i], linewidth=2, label=model_name, alpha=0.8)
    ax3.set_title('训练准确率对比', fontsize=14, fontweight='bold')
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('准确率')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 验证准确率对比
    ax4 = axes[1, 1]
    for i, model_name in enumerate(model_names):
        ax4.plot(results[model_name]['val_accuracies'], 
                color=colors[i], linewidth=2, label=model_name, alpha=0.8)
    ax4.set_title('验证准确率对比', fontsize=14, fontweight='bold')
    ax4.set_xlabel('训练轮次')
    ax4.set_ylabel('准确率')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_convergence_analysis(results):
    """创建收敛性分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('模型收敛性分析', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    model_names = list(results.keys())
    
    # 收敛速度分析
    ax1 = axes[0]
    convergence_epochs = []
    for i, model_name in enumerate(model_names):
        val_losses = results[model_name]['val_losses']
        # 找到损失稳定时的轮次（连续5轮变化小于1%）
        stable_epoch = len(val_losses)
        for j in range(5, len(val_losses)):
            recent_losses = val_losses[j-5:j]
            if max(recent_losses) - min(recent_losses) < 0.01 * min(recent_losses):
                stable_epoch = j
                break
        convergence_epochs.append(stable_epoch)
        
        ax1.bar(model_name, stable_epoch, color=colors[i], alpha=0.8)
    
    ax1.set_title('收敛速度对比（达到稳定状态所需轮次）', fontsize=14, fontweight='bold')
    ax1.set_ylabel('收敛轮次')
    ax1.grid(True, alpha=0.3)
    
    # 最终性能对比
    ax2 = axes[1]
    final_accuracies = [results[model_name]['val_accuracies'][-1] for model_name in model_names]
    
    bars = ax2.bar(model_names, final_accuracies, color=colors, alpha=0.8)
    ax2.set_title('最终验证准确率对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('准确率')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_complexity_comparison(results):
    """创建模型复杂度对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('模型复杂度与性能关系分析', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    model_names = list(results.keys())
    
    # 计算模型参数数量
    param_counts = []
    for model_name in model_names:
        model = results[model_name]['model']
        param_count = sum(p.numel() for p in model.parameters())
        param_counts.append(param_count)
    
    # 最终准确率
    final_accuracies = [results[model_name]['val_accuracies'][-1] for model_name in model_names]
    
    # 参数数量对比
    ax1 = axes[0]
    bars = ax1.bar(model_names, param_counts, color=colors, alpha=0.8)
    ax1.set_title('模型参数数量对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('参数数量')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(param_counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 效率对比（准确率/参数数量）
    ax2 = axes[1]
    efficiencies = [acc / (count/1000) for acc, count in zip(final_accuracies, param_counts)]
    
    bars = ax2.bar(model_names, efficiencies, color=colors, alpha=0.8)
    ax2.set_title('模型效率对比（准确率/千参数）', fontsize=14, fontweight='bold')
    ax2.set_ylabel('效率指标')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(efficiencies)*0.01,
                f'{eff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_complexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_summary_table(results, X_test, y_test, device):
    """创建性能总结表"""
    # 计算测试集性能
    test_metrics = {}
    
    for model_name in results.keys():
        model = results[model_name]['model']
        model.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_test_tensor = torch.LongTensor(y_test).to(device)
            
            nurse_logits, doctor_logits = model(X_test_tensor)
            nurse_pred = torch.argmax(nurse_logits, dim=1).cpu().numpy()
            doctor_pred = torch.argmax(doctor_logits, dim=1).cpu().numpy()
            
            # 计算指标
            nurse_mae = mean_absolute_error(y_test[:, 0], nurse_pred)
            doctor_mae = mean_absolute_error(y_test[:, 1], doctor_pred)
            nurse_acc = accuracy_score(y_test[:, 0], nurse_pred)
            doctor_acc = accuracy_score(y_test[:, 1], doctor_pred)
            
            test_metrics[model_name] = {
                'nurse_mae': nurse_mae,
                'doctor_mae': doctor_mae,
                'nurse_acc': nurse_acc,
                'doctor_acc': doctor_acc,
                'avg_acc': (nurse_acc + doctor_acc) / 2
            }
    
    # 创建总结表
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    table_data = []
    headers = ['模型', '护士MAE', '医生MAE', '护士准确率', '医生准确率', '平均准确率', '收敛轮次', '参数数量']
    
    for model_name in results.keys():
        metrics = test_metrics[model_name]
        convergence_epoch = len(results[model_name]['val_losses'])
        param_count = sum(p.numel() for p in results[model_name]['model'].parameters())
        
        table_data.append([
            model_name,
            f"{metrics['nurse_mae']:.3f}",
            f"{metrics['doctor_mae']:.3f}",
            f"{metrics['nurse_acc']:.3f}",
            f"{metrics['doctor_acc']:.3f}",
            f"{metrics['avg_acc']:.3f}",
            convergence_epoch,
            f"{param_count:,}"
        ])
    
    # 创建表格
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.1, 0.1, 0.12, 0.12, 0.12, 0.1, 0.15])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置标题行样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置数据行样式
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i == 1:  # 混合模型行
                table[(i, j)].set_facecolor('#FFE66D')
            else:
                table[(i, j)].set_facecolor('#F7F7F7')
    
    plt.title('模型性能总结对比表', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('performance_summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = create_training_process_comparison()
    print("训练过程对比分析完成！") 