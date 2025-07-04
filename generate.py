import math
import random

import numpy as np
import pandas as pd


def generate_non_poisson_arrivals(base_lambda, scenario_type, size=1000):
    """
    生成非泊松到达过程数据
    Args:
        base_lambda: 基础到达率
        scenario_type: 场景类型
        size: 生成数据量
    """
    if scenario_type == 'burst':
        # 突发场景：伽马分布模拟突发到达
        shape = 0.5  # 形状参数小于1，产生突发效应
        scale = base_lambda / shape
        arrivals = np.random.gamma(shape, scale, size)
        return np.clip(arrivals, base_lambda * 0.1, base_lambda * 5)
    
    elif scenario_type == 'seasonal':
        # 季节性场景：正弦波叠加泊松过程
        t = np.linspace(0, 24, size)  # 24小时周期
        seasonal_factor = 1 + 0.8 * np.sin(2 * np.pi * t / 24)  # 80%的季节性变化
        arrivals = base_lambda * seasonal_factor + np.random.normal(0, base_lambda * 0.1, size)
        return np.clip(arrivals, base_lambda * 0.2, base_lambda * 3)
    
    elif scenario_type == 'heavy_tail':
        # 重尾分布：帕累托分布模拟极端事件
        alpha = 1.5  # 形状参数，控制尾部厚度
        arrivals = (np.random.pareto(alpha, size) + 1) * base_lambda * 0.5
        return np.clip(arrivals, base_lambda * 0.1, base_lambda * 10)
    
    elif scenario_type == 'bimodal':
        # 双峰分布：模拟早晚高峰
        peak1 = np.random.normal(base_lambda * 0.7, base_lambda * 0.1, size // 2)
        peak2 = np.random.normal(base_lambda * 1.8, base_lambda * 0.2, size - size // 2)
        arrivals = np.concatenate([peak1, peak2])
        np.random.shuffle(arrivals)
        return np.clip(arrivals, base_lambda * 0.1, base_lambda * 3)
    
    else:
        # 标准泊松过程
        return np.random.poisson(base_lambda, size)

def generate_enhanced_hospital_data(n_samples=10000, non_poisson_ratio=0.2):
    """
    生成增强版医院数据集
    Args:
        n_samples: 总样本数
        non_poisson_ratio: 非泊松数据比例
    """
    data = []
    
    # 计算各类数据的数量
    n_non_poisson = int(n_samples * non_poisson_ratio)
    n_poisson = n_samples - n_non_poisson
    
    # 非泊松场景类型分布
    non_poisson_scenarios = ['burst', 'seasonal', 'heavy_tail', 'bimodal']
    scenario_counts = [n_non_poisson // 4] * 4
    scenario_counts[0] += n_non_poisson % 4  # 余数分配给第一个场景
    
    print(f"生成数据分布:")
    print(f"- 标准泊松场景: {n_poisson} ({(1-non_poisson_ratio)*100:.1f}%)")
    for i, scenario in enumerate(non_poisson_scenarios):
        print(f"- {scenario}场景: {scenario_counts[i]} ({scenario_counts[i]/n_samples*100:.1f}%)")
    
    # 生成标准泊松数据
    for i in range(n_poisson):
        scenario = 'standard'
        lambda_base = random.uniform(5, 50)
        
        # 基础参数
        mu_nurse = random.uniform(2, 8)
        mu_doctor = random.uniform(1, 4)
        s_nurse_max = random.randint(3, 20)
        s_doctor_max = random.randint(2, 15)
        Tmax = random.uniform(10, 60)
        nurse_price = random.uniform(50, 150)
        doctor_price = random.uniform(200, 500)
        
        # 使用标准泊松到达率
        lambda_actual = lambda_base
        
        # 计算最优配置
        optimal_nurses, optimal_doctors = find_optimal_staffing(
            lambda_actual, mu_nurse, mu_doctor, s_nurse_max, s_doctor_max,
            Tmax, nurse_price, doctor_price
        )
        
        # 计算系统指标
        metrics = compute_integrated_system_metrics(
            lambda_actual, mu_nurse, optimal_nurses, mu_doctor, optimal_doctors
        )
        
        if metrics is not None:
            data.append({
                'scenario': scenario,
                'lambda': lambda_actual,
                'mu_nurse': mu_nurse,
                'mu_doctor': mu_doctor,
                's_nurse_max': s_nurse_max,
                's_doctor_max': s_doctor_max,
                'Tmax': Tmax,
                'nurse_price': nurse_price,
                'doctor_price': doctor_price,
                'optimal_nurses': optimal_nurses,
                'optimal_doctors': optimal_doctors,
                'system_total_time': metrics['system_total_time'],
                'patient_loss': 1.0 if metrics['system_total_time'] > Tmax else 0.0,
                'hospital_overload': 1.0 if (optimal_nurses > s_nurse_max or optimal_doctors > s_doctor_max) else 0.0,
                'arrival_pattern': 'poisson'
            })
    
    # 生成非泊松数据
    for scenario_idx, scenario in enumerate(non_poisson_scenarios):
        for i in range(scenario_counts[scenario_idx]):
            lambda_base = random.uniform(5, 50)
            
            # 基础参数
            mu_nurse = random.uniform(2, 8)
            mu_doctor = random.uniform(1, 4)
            s_nurse_max = random.randint(3, 20)
            s_doctor_max = random.randint(2, 15)
            Tmax = random.uniform(10, 60)
            nurse_price = random.uniform(50, 150)
            doctor_price = random.uniform(200, 500)
            
            # 生成非泊松到达率
            lambda_samples = generate_non_poisson_arrivals(lambda_base, scenario, size=100)
            lambda_actual = np.mean(lambda_samples)  # 使用平均值作为代表
            lambda_variance = np.var(lambda_samples)  # 记录方差信息
            
            # 对于非泊松场景，排队论的最优解可能不适用
            # 我们需要考虑到达过程的不确定性
            optimal_nurses, optimal_doctors = find_robust_staffing(
                lambda_actual, lambda_variance, mu_nurse, mu_doctor, 
                s_nurse_max, s_doctor_max, Tmax, nurse_price, doctor_price, scenario
            )
            
            # 计算系统指标（使用平均到达率）
            metrics = compute_integrated_system_metrics(
                lambda_actual, mu_nurse, optimal_nurses, mu_doctor, optimal_doctors
            )
            
            if metrics is not None:
                # 对于非泊松场景，系统性能会有额外的不确定性
                uncertainty_penalty = calculate_uncertainty_penalty(lambda_variance, scenario)
                adjusted_total_time = metrics['system_total_time'] * (1 + uncertainty_penalty)
                
                data.append({
                    'scenario': scenario,
                    'lambda': lambda_actual,
                    'lambda_variance': lambda_variance,
                    'mu_nurse': mu_nurse,
                    'mu_doctor': mu_doctor,
                    's_nurse_max': s_nurse_max,
                    's_doctor_max': s_doctor_max,
                    'Tmax': Tmax,
                    'nurse_price': nurse_price,
                    'doctor_price': doctor_price,
                    'optimal_nurses': optimal_nurses,
                    'optimal_doctors': optimal_doctors,
                    'system_total_time': adjusted_total_time,
                    'patient_loss': 1.0 if adjusted_total_time > Tmax else 0.0,
                    'hospital_overload': 1.0 if (optimal_nurses > s_nurse_max or optimal_doctors > s_doctor_max) else 0.0,
                    'arrival_pattern': 'non_poisson',
                    'uncertainty_penalty': uncertainty_penalty
                })
    
    return pd.DataFrame(data)

def find_robust_staffing(lambda_mean, lambda_var, mu_nurse, mu_doctor, 
                        s_nurse_max, s_doctor_max, Tmax, nurse_price, doctor_price, scenario):
    """
    针对非泊松场景的鲁棒人员配置算法
    考虑到达过程的不确定性，采用更保守的配置策略
    """
    # 计算不确定性调整因子
    cv = np.sqrt(lambda_var) / lambda_mean if lambda_mean > 0 else 0  # 变异系数
    
    # 不同场景的调整策略
    if scenario == 'burst':
        # 突发场景：需要更多缓冲
        safety_factor = 1.3 + 0.5 * cv
    elif scenario == 'seasonal':
        # 季节性场景：需要适应峰值
        safety_factor = 1.2 + 0.3 * cv
    elif scenario == 'heavy_tail':
        # 重尾场景：需要应对极端事件
        safety_factor = 1.5 + 0.7 * cv
    elif scenario == 'bimodal':
        # 双峰场景：需要平衡两个峰值
        safety_factor = 1.25 + 0.4 * cv
    else:
        safety_factor = 1.0
    
    # 调整后的到达率
    adjusted_lambda = lambda_mean * safety_factor
    
    # 使用调整后的到达率计算配置
    return find_optimal_staffing(
        adjusted_lambda, mu_nurse, mu_doctor, s_nurse_max, s_doctor_max,
        Tmax, nurse_price, doctor_price
    )

def calculate_uncertainty_penalty(lambda_var, scenario):
    """计算不确定性惩罚因子"""
    base_penalty = {
        'burst': 0.3,
        'seasonal': 0.15,
        'heavy_tail': 0.5,
        'bimodal': 0.2
    }
    
    # 基础惩罚 + 方差相关的额外惩罚
    return base_penalty.get(scenario, 0) + min(lambda_var / 100, 0.3)

# 保持原有的核心函数
def compute_p0(lambd, mu, s):
    """计算系统空闲概率 P0"""
    if s * mu <= lambd:
        return None
    rho = lambd / (s * mu)
    sum_part = 0.0
    for k in range(s):
        sum_part += ((lambd / mu) ** k) / math.factorial(k)
    term = ((lambd / mu) ** s) / (math.factorial(s) * (1 - rho))
    return 1 / (sum_part + term)

def compute_metrics(lambd, mu, s):
    """计算排队指标"""
    P0 = compute_p0(lambd, mu, s)
    if P0 is None:
        return None, None
    rho = lambd / (s * mu)
    numerator = P0 * ((lambd / mu) ** s) * mu
    denominator = math.factorial(s) * (s * mu - lambd) * (1 - rho)
    Wq = numerator / denominator
    W_total = Wq + 1 / mu
    return Wq, W_total

def find_optimal_staffing(lambd, mu_nurse, mu_doctor, s_nurse_max, s_doctor_max, 
                         Tmax, nurse_price, doctor_price):
    """寻找最优人员配置"""
    best_cost = float('inf')
    best_nurses = 1
    best_doctors = 1
    
    for nurses in range(1, s_nurse_max + 1):
        for doctors in range(1, s_doctor_max + 1):
            metrics = compute_integrated_system_metrics(lambd, mu_nurse, nurses, mu_doctor, doctors)
            if metrics and metrics['system_total_time'] <= Tmax:
                cost = nurses * nurse_price + doctors * doctor_price
                if cost < best_cost:
                    best_cost = cost
                    best_nurses = nurses
                    best_doctors = doctors
    
    return best_nurses, best_doctors

def compute_integrated_system_metrics(lambd, mu_nurse, s_nurse, mu_doctor, s_doctor):
    """计算综合系统指标"""
    # 护士子系统
    nurse_metrics = compute_metrics(lambd, mu_nurse, s_nurse)
    if nurse_metrics[0] is None:
        return None
    
    # 医生子系统  
    doctor_metrics = compute_metrics(lambd, mu_doctor, s_doctor)
    if doctor_metrics[0] is None:
        return None
    
    # 系统总时间（串联系统）
    total_time = nurse_metrics[1] + doctor_metrics[1]
    
    return {
        'nurse_wait_time': nurse_metrics[0],
        'doctor_wait_time': doctor_metrics[0],
        'system_total_time': total_time,
        'nurse_utilization': lambd / (s_nurse * mu_nurse),
        'doctor_utilization': lambd / (s_doctor * mu_doctor)
    }

if __name__ == "__main__":
    # 生成增强版数据集
    print("开始生成增强版医院数据集...")
    df = generate_enhanced_hospital_data(n_samples=10000, non_poisson_ratio=0.2)
    
    # 保存数据
    df.to_csv('emergency_hospital_data_enhanced.csv', index=False)
    print(f"数据生成完成！共生成 {len(df)} 条记录")
    
    # 数据统计
    print("\n数据集统计:")
    print(df['scenario'].value_counts())
    print(f"\n非泊松数据比例: {(df['arrival_pattern'] == 'non_poisson').mean():.2%}")
    print(f"约束违反率: {df['patient_loss'].mean():.2%}")
    print(f"资源超限率: {df['hospital_overload'].mean():.2%}")
