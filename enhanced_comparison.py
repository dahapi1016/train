import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score


def scenario_based_performance_analysis(hybrid_results, traditional_results, queue_results, X_raw_test, y_test):
    """基于场景的性能分析"""
    
    # 按场景分组分析
    scenarios = X_raw_test['scenario'].unique()
    scenario_performance = {}
    
    for scenario in scenarios:
        mask = X_raw_test['scenario'] == scenario
        if mask.sum() == 0:
            continue
            
        scenario_data = {
            'scenario': scenario,
            'sample_count': mask.sum(),
            'methods': {}
        }
        
        # 分析各方法在该场景下的表现
        methods = {
            'PAN+DNN Hybrid': hybrid_results,
            'Traditional PAN': traditional_results['Traditional_PAN'],
            'Traditional DNN': traditional_results['Traditional_DNN'],
            'Queue Theory': queue_results
        }
        
        for method_name, results in methods.items():
            nurse_pred_scenario = results['nurse_pred'][mask]
            doctor_pred_scenario = results['doctor_pred'][mask]
            nurse_true_scenario = y_test[mask, 0]
            doctor_true_scenario = y_test[mask, 1]
            
            # 计算场景特定指标
            scenario_data['methods'][method_name] = {
                'nurse_mae': mean_absolute_error(nurse_true_scenario, nurse_pred_scenario),
                'doctor_mae': mean_absolute_error(doctor_true_scenario, doctor_pred_scenario),
                'nurse_accuracy': accuracy_score(nurse_true_scenario, nurse_pred_scenario),
                'doctor_accuracy': accuracy_score(doctor_true_scenario, doctor_pred_scenario),
                'constraint_violation_rate': calculate_constraint_violations(
                    nurse_pred_scenario, doctor_pred_scenario, X_raw_test[mask]
                )
            }
        
        scenario_performance[scenario] = scenario_data
    
    return scenario_performance

def calculate_constraint_violations(nurse_pred, doctor_pred, X_raw_scenario):
    """计算约束违反率"""
    violations = 0
    total = len(nurse_pred)
    
    for i in range(total):
        # 检查资源约束
        if (nurse_pred[i] > X_raw_scenario.iloc[i]['s_nurse_max'] or 
            doctor_pred[i] > X_raw_scenario.iloc[i]['s_doctor_max']):
            violations += 1
            continue
            
        # 检查时间约束（需要重新计算系统指标）
        from generate import compute_integrated_system_metrics
        metrics = compute_integrated_system_metrics(
            X_raw_scenario.iloc[i]['lambda'],
            X_raw_scenario.iloc[i]['mu_nurse'], nurse_pred[i],
            X_raw_scenario.iloc[i]['mu_doctor'], doctor_pred[i]
        )
        
        if metrics and metrics['system_total_time'] > X_raw_scenario.iloc[i]['Tmax']:
            violations += 1
    
    return violations / total if total > 0 else 0

def create_scenario_comparison_visualization(scenario_performance):
    """创建场景对比可视化"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    scenarios = list(scenario_performance.keys())
    methods = ['PAN+DNN Hybrid', 'Traditional PAN', 'Traditional DNN', 'Queue Theory']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # 1. 护士预测MAE对比
    mae_data = {method: [] for method in methods}
    for scenario in scenarios:
        for method in methods:
            mae_data[method].append(
                scenario_performance[scenario]['methods'][method]['nurse_mae']
            )
    
    x = np.arange(len(scenarios))
    width = 0.2
    for i, method in enumerate(methods):
        ax1.bar(x + i*width, mae_data[method], width, label=method, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('场景类型')
    ax1.set_ylabel('护士预测MAE')
    ax1.set_title('不同场景下护士预测误差对比', fontweight='bold', size=14)
    ax1.set_xticks(x + width*1.5)
    ax1.set_xticklabels(scenarios, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 医生预测MAE对比
    mae_data_doctor = {method: [] for method in methods}
    for scenario in scenarios:
        for method in methods:
            mae_data_doctor[method].append(
                scenario_performance[scenario]['methods'][method]['doctor_mae']
            )
    
    for i, method in enumerate(methods):
        ax2.bar(x + i*width, mae_data_doctor[method], width, label=method, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('场景类型')
    ax2.set_ylabel('医生预测MAE')
    ax2.set_title('不同场景下医生预测误差对比', fontweight='bold', size=14)
    ax2.set_xticks(x + width*1.5)
    ax2.set_xticklabels(scenarios, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 约束违反率对比
    violation_data = {method: [] for method in methods}
    for scenario in scenarios:
        for method in methods:
            violation_data[method].append(
                scenario_performance[scenario]['methods'][method]['constraint_violation_rate']
            )
    
    for i, method in enumerate(methods):
        ax3.bar(x + i*width, violation_data[method], width, label=method, color=colors[i], alpha=0.8)
    
    ax3.set_xlabel('场景类型')
    ax3.set_ylabel('约束违反率')
    ax3.set_title('不同场景下约束违反率对比', fontweight='bold', size=14)
    ax3.set_xticks(x + width*1.5)
    ax3.set_xticklabels(scenarios, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 综合性能雷达图（非泊松场景）
    non_poisson_scenarios = [s for s in scenarios if s != 'standard']
    if non_poisson_scenarios:
        # 计算非泊松场景的平均性能
        avg_performance = {method: {'mae': 0, 'accuracy': 0, 'violation': 0} for method in methods}
        
        for scenario in non_poisson_scenarios:
            for method in methods:
                perf = scenario_performance[scenario]['methods'][method]
                avg_performance[method]['mae'] += (perf['nurse_mae'] + perf['doctor_mae']) / 2
                avg_performance[method]['accuracy'] += (perf['nurse_accuracy'] + perf['doctor_accuracy']) / 2
                avg_performance[method]['violation'] += perf['constraint_violation_rate']
        
        # 归一化
        for method in methods:
            avg_performance[method]['mae'] /= len(non_poisson_scenarios)
            avg_performance[method]['accuracy'] /= len(non_poisson_scenarios)
            avg_performance[method]['violation'] /= len(non_poisson_scenarios)
        
        # 创建雷达图
        categories = ['预测精度\n(1-MAE)', '分类准确率', '约束满足率\n(1-违反率)']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        
        for i, method in enumerate(methods):
            values = [
                1 - min(avg_performance[method]['mae'] / 5, 0.8),  # 预测精度
                avg_performance[method]['accuracy'],  # 准确率
                1 - avg_performance[method]['violation']  # 约束满足率
            ]
            values += values[:1]
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax4.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_title('非泊松场景综合性能对比', fontweight='bold', size=14, pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax4.grid(True)
        ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('scenario_based_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_adaptability_analysis(scenario_performance):
    """创建适应性分析"""
    
    # 计算各方法在不同场景下的性能稳定性
    methods = ['PAN+DNN Hybrid', 'Traditional PAN', 'Traditional DNN', 'Queue Theory']
    scenarios = list(scenario_performance.keys())
    
    stability_metrics = {}
    
    for method in methods:
        mae_values = []
        violation_rates = []
        
        for scenario in scenarios:
            perf = scenario_performance[scenario]['methods'][method]
            avg_mae = (perf['nurse_mae'] + perf['doctor_mae']) / 2
            mae_values.append(avg_mae)
            violation_rates.append(perf['constraint_violation_rate'])
        
        # 计算稳定性指标
        stability_metrics[method] = {
            'mae_std': np.std(mae_values),  # MAE标准差（越小越稳定）
            'mae_cv': np.std(mae_values) / np.mean(mae_values) if np.mean(mae_values) > 0 else 0,  # 变异系数
            'violation_std': np.std(violation_rates),  # 违反率标准差
            'max_violation': max(violation_rates),  # 最大违反率
            'adaptability_score': calculate_adaptability_score(mae_values, violation_rates)
        }
    
    # 可视化适应性分析
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 性能稳定性对比
    methods_list = list(stability_metrics.keys())
    mae_stds = [stability_metrics[m]['mae_std'] for m in methods_list]
    violation_stds = [stability_metrics[m]['violation_std'] for m in methods_list]
    
    x = np.arange(len(methods_list))
    width = 0.35
    
    ax1.bar(x - width/2, mae_stds, width, label='预测误差稳定性', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, violation_stds, width, label='约束违反稳定性', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('方法')
    ax1.set_ylabel('标准差（越小越稳定）')
    ax1.set_title('不同方法的性能稳定性对比', fontweight='bold', size=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_list, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 适应性得分对比
    adaptability_scores = [stability_metrics[m]['adaptability_score'] for m in methods_list]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    bars = ax2.bar(methods_list, adaptability_scores, color=colors, alpha=0.8)
    ax2.set_xlabel('方法')
    ax2.set_ylabel('适应性得分（越高越好）')
    ax2.set_title('不同方法的适应性得分对比', fontweight='bold', size=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, adaptability_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('adaptability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stability_metrics

def calculate_adaptability_score(mae_values, violation_rates):
    """计算适应性得分"""
    # 适应性得分 = 1 / (1 + 性能波动惩罚 + 约束违反惩罚)
    mae_penalty = np.std(mae_values) / np.mean(mae_values) if np.mean(mae_values) > 0 else 0
    violation_penalty = np.mean(violation_rates) + np.std(violation_rates)
    
    return 1 / (1 + mae_penalty + violation_penalty * 2)  # 约束违反权重更高

def print_comprehensive_analysis_results(scenario_performance, stability_metrics):
    """打印综合分析结果"""
    
    print("\n" + "="*80)
    print("                    综合性能分析报告")
    print("="*80)
    
    # 1. 场景表现总结
    print("\n1. 各场景表现总结:")
    print("-" * 50)
    
    scenarios = list(scenario_performance.keys())
    methods = ['PAN+DNN Hybrid', 'Traditional PAN', 'Traditional DNN', 'Queue Theory']
    
    for scenario in scenarios:
        print(f"\n【{scenario.upper()}场景】")
        print(f"样本数量: {scenario_performance[scenario]['sample_count']}")
        
        # 找出该场景下表现最好的方法
        best_method = min(methods, key=lambda m: 
            scenario_performance[scenario]['methods'][m]['constraint_violation_rate'])
        
        print(f"最佳方法: {best_method}")
        print("各方法约束违反率:")
        for method in methods:
            violation_rate = scenario_performance[scenario]['methods'][method]['constraint_violation_rate']
            print(f"  {method}: {violation_rate:.3f}")
    
    # 2. 适应性排名
    print("\n2. 方法适应性排名:")
    print("-" * 50)
    
    sorted_methods = sorted(stability_metrics.items(), 
                          key=lambda x: x[1]['adaptability_score'], reverse=True)
    
    for i, (method, metrics) in enumerate(sorted_methods, 1):
        print(f"{i}. {method}")
        print(f"   适应性得分: {metrics['adaptability_score']:.4f}")
        print(f"   性能稳定性: {metrics['mae_cv']:.4f} (变异系数)")
        print(f"   最大约束违反率: {metrics['max_violation']:.3f}")
        print()
    
    # 3. 核心发现
    print("3. 核心发现:")
    print("-" * 50)
    
    # 找出PAN+DNN相对于排队论的优势场景
    hybrid_better_scenarios = []
    for scenario in scenarios:
        if scenario != 'standard':  # 非标准场景
            hybrid_violation = scenario_performance[scenario]['methods']['PAN+DNN Hybrid']['constraint_violation_rate']
            queue_violation = scenario_performance[scenario]['methods']['Queue Theory']['constraint_violation_rate']
            
            if hybrid_violation < queue_violation:
                improvement = (queue_violation - hybrid_violation) / queue_violation * 100
                hybrid_better_scenarios.append((scenario, improvement))
    
    if hybrid_better_scenarios:
        print("✓ PAN+DNN混合模型在以下非标准场景中显著优于排队论:")
        for scenario, improvement in hybrid_better_scenarios:
            print(f"  - {scenario}场景: 约束违反率降低 {improvement:.1f}%")
    
    # 计算总体优势
    non_poisson_scenarios = [s for s in scenarios if s != 'standard']
    if non_poisson_scenarios:
        avg_hybrid_violation = np.mean([
            scenario_performance[s]['methods']['PAN+DNN Hybrid']['constraint_violation_rate']
            for s in non_poisson_scenarios
        ])
        avg_queue_violation = np.mean([
            scenario_performance[s]['methods']['Queue Theory']['constraint_violation_rate']
            for s in non_poisson_scenarios
        ])
        
        if avg_hybrid_violation < avg_queue_violation:
            overall_improvement = (avg_queue_violation - avg_hybrid_violation) / avg_queue_violation * 100
            print(f"\n✓ 在所有非泊松场景中，PAN+DNN平均约束违反率比排队论低 {overall_improvement:.1f}%")
    
    print(f"\n✓ PAN+DNN适应性得分: {stability_metrics['PAN+DNN Hybrid']['adaptability_score']:.4f}")
    print(f"✓ 排队论适应性得分: {stability_metrics['Queue Theory']['adaptability_score']:.4f}")
    
    print("\n" + "="*80)

