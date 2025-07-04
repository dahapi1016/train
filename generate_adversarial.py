import math
import random

import pandas as pd


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

def compute_integrated_system_metrics(lambd, mu_nurse, s_nurse, mu_doctor, s_doctor):
    """计算综合系统指标"""
    nurse_metrics = compute_metrics(lambd, mu_nurse, s_nurse)
    if nurse_metrics[0] is None:
        return None
    
    doctor_metrics = compute_metrics(lambd, mu_doctor, s_doctor)
    if doctor_metrics[0] is None:
        return None
    
    total_time = nurse_metrics[1] + doctor_metrics[1]
    
    return {
        'nurse_wait_time': nurse_metrics[0],
        'doctor_wait_time': doctor_metrics[0],
        'system_total_time': total_time,
        'nurse_utilization': lambd / (s_nurse * mu_nurse),
        'doctor_utilization': lambd / (s_doctor * mu_doctor)
    }

def find_optimal_staffing_queue_theory(lambd, mu_nurse, mu_doctor, s_nurse_max, s_doctor_max, Tmax, nurse_price, doctor_price):
    """标准排队论最优配置算法"""
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

def find_optimal_staffing_robust(lambd, mu_nurse, mu_doctor, s_nurse_max, s_doctor_max, Tmax, nurse_price, doctor_price):
    """鲁棒最优配置算法（考虑不确定性）"""
    best_cost = float('inf')
    best_nurses = 1
    best_doctors = 1
    
    # 增加安全边际
    safety_factor = 0.85  # 要求系统时间不超过Tmax的85%
    adjusted_Tmax = Tmax * safety_factor
    
    for nurses in range(1, s_nurse_max + 1):
        for doctors in range(1, s_doctor_max + 1):
            metrics = compute_integrated_system_metrics(lambd, mu_nurse, nurses, mu_doctor, doctors)
            if metrics and metrics['system_total_time'] <= adjusted_Tmax:
                cost = nurses * nurse_price + doctors * doctor_price
                if cost < best_cost:
                    best_cost = cost
                    best_nurses = nurses
                    best_doctors = doctors
    
    return best_nurses, best_doctors

def generate_queue_theory_challenging_data(n_samples=5000):
    """
    生成专门挑战排队论局限性的数据集
    重点生成排队论假设不成立的场景
    """
    data = []
    
    # 场景1: 非平稳到达过程 (40%)
    n_non_stationary = int(n_samples * 0.4)
    for i in range(n_non_stationary):
        scenario = 'non_stationary'
        
        # 时变到达率：λ(t) = λ_base * (1 + 0.8*sin(2πt/24))
        lambda_base = random.uniform(8, 25)
        lambda_peak = lambda_base * 1.8  # 峰值是基础值的1.8倍
        lambda_valley = lambda_base * 0.4  # 谷值是基础值的0.4倍
        
        # 排队论用平均值，但实际需要考虑峰值
        lambda_avg = (lambda_peak + lambda_valley) / 2
        lambda_effective = lambda_peak  # 实际需要按峰值配置
        
        mu_nurse = random.uniform(3, 7)
        mu_doctor = random.uniform(1.5, 3.5)
        s_nurse_max = random.randint(5, 15)
        s_doctor_max = random.randint(3, 10)
        Tmax = random.uniform(15, 45)
        nurse_price = random.uniform(60, 120)
        doctor_price = random.uniform(250, 450)
        
        # 排队论解（基于平均到达率，会低估需求）
        queue_nurses, queue_doctors = find_optimal_staffing_queue_theory(
            lambda_avg, mu_nurse, mu_doctor, s_nurse_max, s_doctor_max, Tmax, nurse_price, doctor_price
        )
        
        # 实际最优解（考虑峰值需求）
        optimal_nurses, optimal_doctors = find_optimal_staffing_robust(
            lambda_effective, mu_nurse, mu_doctor, s_nurse_max, s_doctor_max, Tmax, nurse_price, doctor_price
        )
        
        # 计算排队论解的实际性能（用峰值到达率）
        queue_metrics = compute_integrated_system_metrics(lambda_effective, mu_nurse, queue_nurses, mu_doctor, queue_doctors)
        optimal_metrics = compute_integrated_system_metrics(lambda_effective, mu_nurse, optimal_nurses, mu_doctor, optimal_doctors)
        
        if queue_metrics and optimal_metrics:
            data.append({
                'scenario': scenario,
                'lambda': lambda_avg,  # 输入特征用平均值
                'lambda_peak': lambda_peak,  # 记录峰值用于评估
                'mu_nurse': mu_nurse,
                'mu_doctor': mu_doctor,
                's_nurse_max': s_nurse_max,
                's_doctor_max': s_doctor_max,
                'Tmax': Tmax,
                'nurse_price': nurse_price,
                'doctor_price': doctor_price,
                'optimal_nurses': optimal_nurses,
                'optimal_doctors': optimal_doctors,
                'queue_nurses': queue_nurses,  # 排队论解
                'queue_doctors': queue_doctors,
                'system_total_time': optimal_metrics['system_total_time'],
                'queue_total_time': queue_metrics['system_total_time'],  # 排队论解的实际性能
                'patient_loss': 1.0 if optimal_metrics['system_total_time'] > Tmax else 0.0,
                'queue_patient_loss': 1.0 if queue_metrics['system_total_time'] > Tmax else 0.0,
                'hospital_overload': 1.0 if (optimal_nurses > s_nurse_max or optimal_doctors > s_doctor_max) else 0.0,
                'queue_failure': 1.0 if queue_metrics['system_total_time'] > Tmax * 1.2 else 0.0  # 排队论严重失效
            })
    
    # 场景2: 服务时间高变异性 (30%)
    n_high_variance = int(n_samples * 0.3)
    for i in range(n_high_variance):
        scenario = 'high_variance_service'
        
        lambda_val = random.uniform(10, 30)
        
        # 高变异性服务时间（CV > 1）
        mu_nurse_mean = random.uniform(3, 6)
        mu_doctor_mean = random.uniform(1.5, 3)
        
        # 实际服务率需要考虑变异性的影响
        cv_nurse = random.uniform(1.5, 3.0)  # 变异系数 > 1
        cv_doctor = random.uniform(1.2, 2.5)
        
        # 排队论用平均服务率
        mu_nurse_queue = mu_nurse_mean
        mu_doctor_queue = mu_doctor_mean
        
        # 实际有效服务率（考虑变异性影响）
        mu_nurse_effective = mu_nurse_mean / (1 + cv_nurse * 0.3)  # 变异性降低有效服务率
        mu_doctor_effective = mu_doctor_mean / (1 + cv_doctor * 0.3)
        
        s_nurse_max = random.randint(4, 12)
        s_doctor_max = random.randint(3, 8)
        Tmax = random.uniform(20, 50)
        nurse_price = random.uniform(70, 130)
        doctor_price = random.uniform(280, 480)
        
        # 排队论解
        queue_nurses, queue_doctors = find_optimal_staffing_queue_theory(
            lambda_val, mu_nurse_queue, mu_doctor_queue, s_nurse_max, s_doctor_max, Tmax, nurse_price, doctor_price
        )
        
        # 实际最优解
        optimal_nurses, optimal_doctors = find_optimal_staffing_robust(
            lambda_val, mu_nurse_effective, mu_doctor_effective, s_nurse_max, s_doctor_max, Tmax, nurse_price, doctor_price
        )
        
        # 评估性能
        queue_metrics = compute_integrated_system_metrics(lambda_val, mu_nurse_effective, queue_nurses, mu_doctor_effective, queue_doctors)
        optimal_metrics = compute_integrated_system_metrics(lambda_val, mu_nurse_effective, optimal_nurses, mu_doctor_effective, optimal_doctors)
        
        if queue_metrics and optimal_metrics:
            data.append({
                'scenario': scenario,
                'lambda': lambda_val,
                'mu_nurse': mu_nurse_effective,
                'mu_doctor': mu_doctor_effective,
                'cv_nurse': cv_nurse,
                'cv_doctor': cv_doctor,
                's_nurse_max': s_nurse_max,
                's_doctor_max': s_doctor_max,
                'Tmax': Tmax,
                'nurse_price': nurse_price,
                'doctor_price': doctor_price,
                'optimal_nurses': optimal_nurses,
                'optimal_doctors': optimal_doctors,
                'queue_nurses': queue_nurses,
                'queue_doctors': queue_doctors,
                'system_total_time': optimal_metrics['system_total_time'],
                'queue_total_time': queue_metrics['system_total_time'],
                'patient_loss': 1.0 if optimal_metrics['system_total_time'] > Tmax else 0.0,
                'queue_patient_loss': 1.0 if queue_metrics['system_total_time'] > Tmax else 0.0,
                'hospital_overload': 1.0 if (optimal_nurses > s_nurse_max or optimal_doctors > s_doctor_max) else 0.0,
                'queue_failure': 1.0 if queue_metrics['system_total_time'] > Tmax * 1.2 else 0.0
            })
    
    # 场景3: 相关性服务过程 (20%)
    n_correlated = int(n_samples * 0.2)
    for i in range(n_correlated):
        scenario = 'correlated_service'
        
        lambda_val = random.uniform(12, 28)
        mu_nurse = random.uniform(4, 8)
        mu_doctor = random.uniform(2, 4)
        
        # 护士和医生服务时间相关（排队论假设独立）
        correlation = random.uniform(0.3, 0.7)  # 正相关
        
        s_nurse_max = random.randint(4, 10)
        s_doctor_max = random.randint(3, 8)
        Tmax = random.uniform(18, 40)
        nurse_price = random.uniform(65, 125)
        doctor_price = random.uniform(270, 470)
        
        # 排队论解（假设独立）
        queue_nurses, queue_doctors = find_optimal_staffing_queue_theory(
            lambda_val, mu_nurse, mu_doctor, s_nurse_max, s_doctor_max, Tmax, nurse_price, doctor_price
        )
        
        # 考虑相关性的最优解
        correlation_penalty = 1 + correlation * 0.4
        adjusted_Tmax = Tmax / correlation_penalty
        
        optimal_nurses, optimal_doctors = find_optimal_staffing_queue_theory(
            lambda_val, mu_nurse, mu_doctor, s_nurse_max, s_doctor_max, adjusted_Tmax, nurse_price, doctor_price
        )
        
        # 计算性能
        base_queue_metrics = compute_integrated_system_metrics(lambda_val, mu_nurse, queue_nurses, mu_doctor, queue_doctors)
        base_optimal_metrics = compute_integrated_system_metrics(lambda_val, mu_nurse, optimal_nurses, mu_doctor, optimal_doctors)
        
        if base_queue_metrics and base_optimal_metrics:
            # 应用相关性影响
            queue_total_time = base_queue_metrics['system_total_time'] * correlation_penalty
            optimal_total_time = base_optimal_metrics['system_total_time'] * correlation_penalty
            
            data.append({
                'scenario': scenario,
                'lambda': lambda_val,
                'mu_nurse': mu_nurse,
                'mu_doctor': mu_doctor,
                'correlation': correlation,
                's_nurse_max': s_nurse_max,
                's_doctor_max': s_doctor_max,
                'Tmax': Tmax,
                'nurse_price': nurse_price,
                'doctor_price': doctor_price,
                'optimal_nurses': optimal_nurses,
                'optimal_doctors': optimal_doctors,
                'queue_nurses': queue_nurses,
                'queue_doctors': queue_doctors,
                'system_total_time': optimal_total_time,
                'queue_total_time': queue_total_time,
                'patient_loss': 1.0 if optimal_total_time > Tmax else 0.0,
                'queue_patient_loss': 1.0 if queue_total_time > Tmax else 0.0,
                'hospital_overload': 1.0 if (optimal_nurses > s_nurse_max or optimal_doctors > s_doctor_max) else 0.0,
                'queue_failure': 1.0 if queue_total_time > Tmax * 1.2 else 0.0
            })
    
    # 场景4: 标准场景作为对照 (10%)
    n_standard = n_samples - n_non_stationary - n_high_variance - n_correlated
    for i in range(n_standard):
        scenario = 'standard'
        
        lambda_val = random.uniform(8, 25)
        mu_nurse = random.uniform(3, 7)
        mu_doctor = random.uniform(1.5, 3.5)
        s_nurse_max = random.randint(4, 12)
        s_doctor_max = random.randint(3, 8)
        Tmax = random.uniform(20, 50)
        nurse_price = random.uniform(60, 130)
        doctor_price = random.uniform(250, 480)
        
        # 标准场景下排队论和最优解应该接近
        optimal_nurses, optimal_doctors = find_optimal_staffing_queue_theory(
            lambda_val, mu_nurse, mu_doctor, s_nurse_max, s_doctor_max, Tmax, nurse_price, doctor_price
        )
        
        metrics = compute_integrated_system_metrics(lambda_val, mu_nurse, optimal_nurses, mu_doctor, optimal_doctors)
        
        if metrics:
            data.append({
                'scenario': scenario,
                'lambda': lambda_val,
                'mu_nurse': mu_nurse,
                'mu_doctor': mu_doctor,
                's_nurse_max': s_nurse_max,
                's_doctor_max': s_doctor_max,
                'Tmax': Tmax,
                'nurse_price': nurse_price,
                'doctor_price': doctor_price,
                'optimal_nurses': optimal_nurses,
                'optimal_doctors': optimal_doctors,
                'queue_nurses': optimal_nurses,  # 标准场景下相同
                'queue_doctors': optimal_doctors,
                'system_total_time': metrics['system_total_time'],
                'queue_total_time': metrics['system_total_time'],
                'patient_loss': 1.0 if metrics['system_total_time'] > Tmax else 0.0,
                'queue_patient_loss': 1.0 if metrics['system_total_time'] > Tmax else 0.0,
                'hospital_overload': 1.0 if (optimal_nurses > s_nurse_max or optimal_doctors > s_doctor_max) else 0.0,
                'queue_failure': 0.0  # 标准场景下排队论不失效
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("生成挑战排队论的对抗数据集...")
    df = generate_queue_theory_challenging_data(n_samples=5000)
    
    df.to_csv('queue_theory_challenging_data.csv', index=False)
    print(f"数据生成完成！共生成 {len(df)} 条记录")
    
    # 统计排队论失效情况
    print("\n排队论失效统计:")
    print(f"总样本数: {len(df)}")
    print(f"排队论严重失效样本: {df['queue_failure'].sum()} ({df['queue_failure'].mean():.2%})")
    print(f"排队论约束违反样本: {df['queue_patient_loss'].sum()} ({df['queue_patient_loss'].mean():.2%})")
    
    print("\n各场景统计:")
    for scenario in df['scenario'].unique():
        scenario_data = df[df['scenario'] == scenario]
        failure_rate = scenario_data['queue_failure'].mean()
        violation_rate = scenario_data['queue_patient_loss'].mean()
        print(f"{scenario}: 失效率 {failure_rate:.2%}, 违反率 {violation_rate:.2%}")

