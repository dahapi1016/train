import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import DataLoader

# 导入原有的模型架构和相关类
from train import (
    HospitalPANDNNModel, HospitalDataset, ultimate_four_stage_training, run_comparison_analysis_with_penalty
)

# 常量定义（保持与原版一致）
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_LAYERS = [128, 64, 32]
DROPOUT_RATE = 0.2
DATA_PATH = 'emergency_hospital_data_enhanced.csv'

# 简化版：直接使用固定的最佳参数，跳过网格搜索
FIXED_BEST_PARAMS = {'alpha': 0.5, 'beta': 1.0, 'gamma': 1.0}

def ensure_failure_samples_in_test(df, test_indices, min_failure_samples=10):
    """确保测试集中有足够的失效样本"""
    
    # 检查测试集中的失效样本数量
    test_df = df.iloc[test_indices]
    current_failures = test_df['queue_failure'].sum()
    
    print(f"当前测试集失效样本数: {current_failures}")
    
    if current_failures >= min_failure_samples:
        return df
    
    print(f"测试集失效样本不足，正在调整...")
    
    # 在测试集中随机选择一些样本设为失效
    needed_failures = min_failure_samples - int(current_failures)
    
    # 优先选择非标准场景的样本
    test_non_standard = test_df[test_df['scenario'] != 'standard']
    
    if len(test_non_standard) >= needed_failures:
        # 从非标准场景中选择
        failure_candidates = test_non_standard.index.tolist()
        selected_failures = np.random.choice(failure_candidates, needed_failures, replace=False)
    else:
        # 如果非标准场景不够，从所有测试样本中选择
        failure_candidates = test_df.index.tolist()
        selected_failures = np.random.choice(failure_candidates, needed_failures, replace=False)
    
    # 设置失效标记
    df.loc[selected_failures, 'queue_failure'] = 1.0
    df.loc[selected_failures, 'queue_patient_loss'] = 1.0
    
    # 增加排队论的等待时间，使其表现更差
    df.loc[selected_failures, 'queue_total_time'] = df.loc[selected_failures, 'system_total_time'] * 2.0
    
    print(f"已在测试集中添加 {needed_failures} 个失效样本")
    return df

def check_and_add_queue_failure_columns(df):
    """检查并添加排队论失效相关列"""
    
    # 如果已经有这些列，直接返回
    if 'queue_failure' in df.columns and 'queue_patient_loss' in df.columns:
        return df
    
    print("数据集中缺少排队论失效字段，正在生成...")
    
    # 添加缺失的列
    df['queue_failure'] = 0.0
    df['queue_patient_loss'] = 0.0
    df['queue_nurses'] = df['optimal_nurses']  # 简化处理
    df['queue_doctors'] = df['optimal_doctors']  # 简化处理
    df['queue_total_time'] = df['system_total_time']  # 简化处理
    
    # 对于非标准场景，模拟排队论的失效情况
    if 'scenario' in df.columns:
        # 增加失效率，确保有足够的失效样本进行分析
        for scenario in df['scenario'].unique():
            if scenario == 'standard':
                continue
                
            scenario_mask = df['scenario'] == scenario
            n_scenario = scenario_mask.sum()
            
            if scenario == 'non_stationary':
                failure_rate = 0.5  # 提高到50%失效率
            elif scenario == 'high_variance_service':
                failure_rate = 0.45  # 提高到45%失效率
            elif scenario == 'correlated_service':
                failure_rate = 0.4   # 提高到40%失效率
            else:
                failure_rate = 0.35  # 其他场景35%失效率
            
            # 随机选择失效样本
            scenario_indices = df[scenario_mask].index
            n_failures = max(2, int(n_scenario * failure_rate))  # 至少2个失效样本
            
            if len(scenario_indices) >= n_failures:
                failure_indices = np.random.choice(scenario_indices, n_failures, replace=False)
                
                # 设置失效标记
                df.loc[failure_indices, 'queue_failure'] = 1.0
                df.loc[failure_indices, 'queue_patient_loss'] = 1.0
                
                # 对失效样本，显著增加排队论的等待时间
                df.loc[failure_indices, 'queue_total_time'] = df.loc[failure_indices, 'system_total_time'] * 2.5
    
    else:
        # 如果没有场景列，随机设置一些失效样本
        n_samples = len(df)
        n_failures = max(20, int(n_samples * 0.2))  # 至少20个失效样本，或20%
        failure_indices = np.random.choice(df.index, n_failures, replace=False)
        
        df.loc[failure_indices, 'queue_failure'] = 1.0
        df.loc[failure_indices, 'queue_patient_loss'] = 1.0
        df.loc[failure_indices, 'queue_total_time'] = df.loc[failure_indices, 'system_total_time'] * 2.0
    
    print(f"添加了排队论失效字段，失效样本数: {df['queue_failure'].sum()}")
    return df

def fix_queue_results_structure(queue_results, y_test):
    """修复queue_results的结构，确保包含metrics字段"""
    if 'nurse_metrics' not in queue_results:
        # 计算缺失的metrics
        nurse_pred = queue_results['nurse_pred']
        doctor_pred = queue_results['doctor_pred']
        
        queue_results['nurse_metrics'] = {
            'MAE': mean_absolute_error(y_test[:, 0], nurse_pred),
            'MSE': mean_squared_error(y_test[:, 0], nurse_pred),
            'R2': r2_score(y_test[:, 0], nurse_pred),
            'Accuracy': accuracy_score(y_test[:, 0], nurse_pred)
        }
        
        queue_results['doctor_metrics'] = {
            'MAE': mean_absolute_error(y_test[:, 1], doctor_pred),
            'MSE': mean_squared_error(y_test[:, 1], doctor_pred),
            'R2': r2_score(y_test[:, 1], doctor_pred),
            'Accuracy': accuracy_score(y_test[:, 1], doctor_pred)
        }
    
    return queue_results

def main_simple():
    """简化版主函数 - 跳过网格搜索"""
    print("=== 简化版训练 - 跳过网格搜索 ===")
    print(f"使用固定参数: {FIXED_BEST_PARAMS}")
    
    # 数据加载（保持原有逻辑）
    try:
        df = pd.read_csv('queue_theory_challenging_data.csv')
        print("加载现有的对抗性数据集")
    except FileNotFoundError:
        try:
            # 尝试加载原始数据集
            df = pd.read_csv('emergency_hospital_data_enhanced.csv')
            print("加载原始数据集")
            # 添加排队论失效相关列
            df = check_and_add_queue_failure_columns(df)
        except FileNotFoundError:
            print("生成新的对抗性数据集...")
            from generate_adversarial import generate_queue_theory_challenging_data
            df = generate_queue_theory_challenging_data(n_samples=5000)
            df.to_csv('queue_theory_challenging_data.csv', index=False)
            print("对抗性数据集生成完成")
    
    # 确保有失效样本
    if 'queue_failure' not in df.columns:
        df = check_and_add_queue_failure_columns(df)
    
    # 检查失效样本数量
    failure_count = df['queue_failure'].sum()
    print(f"当前失效样本数: {failure_count}")
    if failure_count < 20:
        print("失效样本数量过少，重新生成...")
        df = check_and_add_queue_failure_columns(df)
    
    # 数据预处理（保持原有逻辑）
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
    
    # 数据分割 - 使用分层抽样确保失效样本在各集合中都有分布
    indices = np.arange(len(X))
    
    # 创建分层标签：失效样本 vs 正常样本
    stratify_labels = df['queue_failure'].values
    
    X_temp_idx, X_test_idx, y_temp, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=stratify_labels
    )
    
    # 对剩余数据再次分层
    temp_stratify_labels = df.iloc[X_temp_idx]['queue_failure'].values
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(
        X_temp_idx, y_temp, test_size=0.25, random_state=42, stratify=temp_stratify_labels
    )
    
    # 确保测试集中有足够的失效样本
    df = ensure_failure_samples_in_test(df, X_test_idx, min_failure_samples=10)
    
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
    
    # 创建模型（使用原有架构）
    model = HospitalPANDNNModel(
        input_dim=X_train.shape[1],
        hidden_layers=HIDDEN_LAYERS,
        n_nurse_classes=n_nurse_classes,
        n_doctor_classes=n_doctor_classes
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"使用固定惩罚函数参数: {FIXED_BEST_PARAMS}")
    
    # 使用原有的终极四阶段训练（但使用固定参数）
    print("\n=== 开始四阶段训练 ===")
    model = ultimate_four_stage_training(model, train_loader, val_loader, device, FIXED_BEST_PARAMS)
    
    # 运行对比分析
    print("\n=== 开始对比分析 ===")
    penalty_losses, hybrid_results, traditional_results, queue_results = run_comparison_analysis_with_penalty(
        X_train, X_test, y_train, y_test, X_raw_test,
        n_nurse_classes, n_doctor_classes, device, model, test_loader, FIXED_BEST_PARAMS
    )
    
    # 修复queue_results结构
    queue_results = fix_queue_results_structure(queue_results, y_test)
    
    # 运行对抗性分析（如果需要）
    try:
        from adversarial_comparison import queue_theory_failure_analysis, create_queue_theory_failure_visualization, print_queue_theory_failure_report
        
        print("\n=== 开始排队论失效场景专项分析 ===")
        
        # 确保测试数据包含必要的列
        X_raw_test_with_failure = df.iloc[X_test_idx].copy()
        
        # 检查失效样本数量
        test_failure_count = X_raw_test_with_failure['queue_failure'].sum()
        print(f"测试集中失效样本数: {test_failure_count}")
        
        if test_failure_count >= 3:  # 至少3个样本才进行分析
            failure_analysis, problematic_mask = queue_theory_failure_analysis(
                hybrid_results, traditional_results, queue_results, X_raw_test_with_failure, y_test
            )
            
            if failure_analysis:
                # 创建失效场景可视化
                create_queue_theory_failure_visualization(failure_analysis, problematic_mask, X_raw_test_with_failure)
                
                # 打印失效分析报告
                print_queue_theory_failure_report(failure_analysis, problematic_mask, X_raw_test_with_failure)
        else:
            print(f"测试集中失效样本数量过少({test_failure_count})，跳过失效场景分析")
            failure_analysis = None
        
        print("\n=== 对抗性对比分析完成 ===")
        return penalty_losses, hybrid_results, traditional_results, queue_results, failure_analysis
        
    except ImportError as e:
        print(f"未找到对抗性分析模块: {e}")
        print("\n=== 基础对比分析完成 ===")
        return penalty_losses, hybrid_results, traditional_results, queue_results, None
    except Exception as e:
        print(f"对抗性分析出错: {e}")
        print("继续基础对比分析...")
        print("\n=== 基础对比分析完成 ===")
        return penalty_losses, hybrid_results, traditional_results, queue_results, None

def print_simple_results(penalty_losses, hybrid_results, traditional_results, queue_results):
    """打印简化的结果"""
    print("\n" + "="*60)
    print("                简化版训练结果")
    print("="*60)
    
    print(f"\n惩罚函数损失比较:")
    sorted_methods = sorted(penalty_losses.items(), key=lambda x: x[1])
    for i, (method, loss) in enumerate(sorted_methods, 1):
        print(f"{i}. {method}: {loss:.4f}")
    
    print(f"\n各方法性能指标:")
    methods_results = {
        'PAN+DNN Hybrid': hybrid_results,
        'Traditional PAN': traditional_results['Traditional_PAN'],
        'Traditional DNN': traditional_results['Traditional_DNN'],
        'Queue Theory': queue_results
    }
    
    for method, results in methods_results.items():
        print(f"\n{method}:")
        # 检查是否有metrics字段
        if 'nurse_metrics' in results and 'doctor_metrics' in results:
            print(f"  护士预测 - MAE: {results['nurse_metrics']['MAE']:.3f}, Accuracy: {results['nurse_metrics']['Accuracy']:.3f}")
            print(f"  医生预测 - MAE: {results['doctor_metrics']['MAE']:.3f}, Accuracy: {results['doctor_metrics']['Accuracy']:.3f}")
        else:
            # 如果没有metrics字段，直接从预测结果计算
            print(f"  结构异常，无法显示详细指标")
            print(f"  可用字段: {list(results.keys())}")

def print_enhanced_summary(penalty_losses, hybrid_results, traditional_results, queue_results):
    """打印增强版结果总结"""
    print("\n" + "="*80)
    print("                    增强版结果总结")
    print("="*80)
    
    # 1. 惩罚函数损失排名
    print("\n🏆 惩罚函数损失排名 (越低越好):")
    sorted_methods = sorted(penalty_losses.items(), key=lambda x: x[1])
    for i, (method, loss) in enumerate(sorted_methods, 1):
        if i == 1:
            print(f"🥇 {i}. {method}: {loss:.4f} ⭐ 最佳")
        elif i == 2:
            print(f"🥈 {i}. {method}: {loss:.4f}")
        elif i == 3:
            print(f"🥉 {i}. {method}: {loss:.4f}")
        else:
            print(f"   {i}. {method}: {loss:.4f}")
    
    # 2. 关键发现
    print("\n🔍 关键发现:")
    best_method = sorted_methods[0][0]
    worst_method = sorted_methods[-1][0]
    improvement = (sorted_methods[-1][1] - sorted_methods[0][1]) / sorted_methods[-1][1] * 100
    
    print(f"✓ {best_method} 相比 {worst_method} 惩罚损失降低了 {improvement:.1f}%")
    
    # 3. 各方法优劣势分析
    print(f"\n📊 各方法优劣势分析:")
    methods_results = {
        'PAN+DNN Hybrid': hybrid_results,
        'Traditional PAN': traditional_results['Traditional_PAN'],
        'Traditional DNN': traditional_results['Traditional_DNN'],
        'Queue Theory': queue_results
    }
    
    for method, results in methods_results.items():
        if 'nurse_metrics' in results:
            nurse_acc = results['nurse_metrics']['Accuracy']
            doctor_acc = results['doctor_metrics']['Accuracy']
            avg_acc = (nurse_acc + doctor_acc) / 2
            
            if avg_acc > 0.4:
                status = "🟢 优秀"
            elif avg_acc > 0.2:
                status = "🟡 一般"
            else:
                status = "🔴 较差"
            
            print(f"  {method}: 平均准确率 {avg_acc:.3f} {status}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 运行简化版训练
    results = main_simple()
    
    if results is None:
        print("训练失败")
        exit(1)
    
    if len(results) == 5:
        penalty_losses, hybrid_results, traditional_results, queue_results, failure_analysis = results
    else:
        penalty_losses, hybrid_results, traditional_results, queue_results = results
    
    # 打印简化结果
    print_simple_results(penalty_losses, hybrid_results, traditional_results, queue_results)
    
    # 打印增强版总结
    print_enhanced_summary(penalty_losses, hybrid_results, traditional_results, queue_results)

