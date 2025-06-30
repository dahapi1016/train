import math
import random

import pandas as pd


# 首先我们要做的是围绕这个文章进行优化，-。
# 第一步是在数据生成部分通过仿真生成数据。要生成病人到达率，医生和护士的数量，以及医护人员的服务率。然后生成数据上面有约束，第一个是病人最大等待时间，一个是最大医护人员数量，前者违背标记为病人流失，后者违背标记为医院超限。
# 第二步是进行计算，文章里面有公式，可以通过那些公式计算出来其他指标。
# 第三步是数据预处理，清洗数据，对流失率和超限率超过一定限度的数据剔除（看看是否有参考文献进行参考限度的数目），然后对剩余的数据提取关键特征（到达率，排队时间，利用率，服务率…），进行归一化方便后续训练
# 第四步是使用PAN+DNN训练，PAN+DNN混合架构
# •	创新点实现三阶段训练法（需在实验部分说明）：
# 1.	预训练阶段：仅训练PAN的特征提取层（冻结FC层）
# 2.	微调阶段：解冻最后三层，加入医院历史数据
# 3.	强化阶段：启用注意力模块，使用加权损失
# 创新点实现可以用别的你觉得合理的方式但简单的PAN是不够的
# 损失函数是L=a*MSE（Wq）+b流失+c超限，优化器用Adam去算。
# 然后要对比这种方法和传统的PAN，DNN和排队之间效能的对比 可视化

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
    """
    计算排队指标
    返回: (Wq: 排队等待时间, W_total: 总等待时间)
    """
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
    """
    计算集成系统指标（护士站 + 医生站串联）
    """
    # 护士站分析 (λ_n = λ)
    Wq_nurse, W_total_nurse = compute_metrics(lambd, mu_nurse, s_nurse)
    if Wq_nurse is None:
        return None

    # 医生站分析 (λ_d = λ，因为是串联系统)
    Wq_doctor, W_total_doctor = compute_metrics(lambd, mu_doctor, s_doctor)
    if Wq_doctor is None:
        return None

    # 系统总吞吐时间
    T_total = Wq_nurse + (1/mu_nurse) + Wq_doctor + (1/mu_doctor)

    # 利用率计算
    rho_nurse = lambd / (s_nurse * mu_nurse)
    rho_doctor = lambd / (s_doctor * mu_doctor)

    return {
        'nurse_queue_time': Wq_nurse,
        'nurse_total_time': W_total_nurse,
        'doctor_queue_time': Wq_doctor,
        'doctor_total_time': W_total_doctor,
        'system_total_time': T_total,
        'nurse_utilization': rho_nurse,
        'doctor_utilization': rho_doctor
    }

def find_optimal_staffing(lambd, mu_nurse, mu_doctor, s_nurse_max, s_doctor_max,
                         Tmax, nurse_price, doctor_price):
    """寻找最优医护人员配置 - 数据生成时使用"""
    min_cost = float('inf')
    optimal_nurses = 1
    optimal_doctors = 1

    # 遍历所有可能的配置，找到满足时间约束的最低成本方案
    for s_nurse in range(1, s_nurse_max + 1):
        for s_doctor in range(1, s_doctor_max + 1):
            metrics = compute_integrated_system_metrics(
                lambd, mu_nurse, s_nurse, mu_doctor, s_doctor
            )
            if metrics is None:
                continue
            # 检查是否满足时间约束
            if metrics['system_total_time'] <= Tmax:
                cost = s_nurse * nurse_price + s_doctor * doctor_price
                if cost < min_cost:
                    min_cost = cost
                    optimal_nurses = s_nurse
                    optimal_doctors = s_doctor
    return optimal_nurses, optimal_doctors

scenarios = [
    # {   # 场景1: 小型医院非紧急（门诊）（不考虑）
    #     'name': 'small_non_emergency',
    #     's_nurse_max': 5,          # 最大护士数
    #     's_doctor_max': 3,         # 最大医生数
    #     'lambda_range': (0.5, 12), # 患者到达率 (人/小时)
    #     'mu_nurse_range': (3, 6),  # 护士服务率 (人/小时)
    #     'mu_doctor_range': (5, 8), # 医生服务率 (人/小时)
    #     'Tmax': 2.0,               # 最长总等待时间 (小时)
    #     'nurse_price': 100,        # 护士单价 (元/小时)
    #     'doctor_price': 300,       # 医生单价 (元/小时)
    #     'num_samples': 3000
    # },
    {   # 场景2: 小型医院紧急（急诊）
        'name': 'small_emergency',
        's_nurse_max': 8,
        's_doctor_max': 4,
        'lambda_range': (10, 35),
        'mu_nurse_range': (5, 10),
        'mu_doctor_range': (10, 15),
        'Tmax': 0.5,  # 30分钟
        'nurse_price': 150,  # 急诊护士成本更高
        'doctor_price': 500,
        'num_samples': 3000
    },
    # {   # 场景3: 大型医院非紧急（综合门诊）（不考虑）
    #     'name': 'large_non_emergency',
    #     's_nurse_max': 20,
    #     's_doctor_max': 10,
    #     'lambda_range': (5, 25),
    #     'mu_nurse_range': (2, 5),
    #     'mu_doctor_range': (3, 6),
    #     'Tmax': 3.0,
    #     'nurse_price': 80,
    #     'doctor_price': 250,
    #     'num_samples': 3000
    # },
    {   # 场景4: 大型医院紧急（重大事故）
        'name': 'large_emergency',
        's_nurse_max': 30,
        's_doctor_max': 20,
        'lambda_range': (50, 115),
        'mu_nurse_range': (5, 12),
        'mu_doctor_range': (6, 15),
        'Tmax': 0.75,  # 45分钟
        'nurse_price': 200,
        'doctor_price': 800,
        'num_samples': 3000
    }
]


def generate_samples_with_constraints(scenario):
    """生成带约束违反标记的样本（125%容忍度）"""
    samples = []

    # 计算125%限制
    s_nurse_125 = int(scenario['s_nurse_max'] * 1.25)
    s_doctor_125 = int(scenario['s_doctor_max'] * 1.25)
    Tmax_125 = scenario['Tmax'] * 1.25

    for _ in range(scenario['num_samples']):
        # 生成医护人员服务率
        mu_nurse = random.uniform(*scenario['mu_nurse_range'])
        mu_doctor = random.uniform(*scenario['mu_doctor_range'])

        # 生成到达率
        lambd = random.uniform(*scenario['lambda_range'])

        # 寻找最优配置
        optimal_nurses, optimal_doctors = find_optimal_staffing(
            lambd, mu_nurse, mu_doctor,
            scenario['s_nurse_max'], scenario['s_doctor_max'],
            scenario['Tmax'], scenario['nurse_price'], scenario['doctor_price']
        )

        # 随机生成医护人员数量（在125%范围内）
        s_nurse = random.randint(1, s_nurse_125)
        s_doctor = random.randint(1, s_doctor_125)

        # 计算集成系统指标
        metrics = compute_integrated_system_metrics(lambd, mu_nurse, s_nurse, mu_doctor, s_doctor)

        if metrics is None:
            # 系统不稳定，设置极大值
            total_wait_time = Tmax_125 * 2
            nurse_utilization = 1.0
            doctor_utilization = 1.0
        else:
            total_wait_time = metrics['system_total_time']
            nurse_utilization = metrics['nurse_utilization']
            doctor_utilization = metrics['doctor_utilization']

        # 检查是否在125%容忍范围内
        within_time_limit = total_wait_time <= Tmax_125
        within_staff_limit = (s_nurse <= s_nurse_125 and s_doctor <= s_doctor_125)

        # 如果超过125%限制，跳过此样本
        if not (within_time_limit and within_staff_limit):
            continue

        # 约束违反标记（100% < x <= 125% 标记为违反）
        patient_loss = 1 if total_wait_time > scenario['Tmax'] else 0
        hospital_overload = 1 if (s_nurse > scenario['s_nurse_max'] or
                                s_doctor > scenario['s_doctor_max']) else 0

        # 稳态条件检查
        stable_nurse = 1 if lambd < s_nurse * mu_nurse else 0
        stable_doctor = 1 if lambd < s_doctor * mu_doctor else 0
        system_stable = stable_nurse and stable_doctor

        # 计算成本
        total_cost = s_nurse * scenario['nurse_price'] + s_doctor * scenario['doctor_price']

        sample = {
            'scenario': scenario['name'],
            'lambda': lambd,
            'mu_nurse': mu_nurse,
            'mu_doctor': mu_doctor,
            's_nurse': s_nurse,
            's_doctor': s_doctor,
            's_nurse_max': scenario['s_nurse_max'],
            's_doctor_max': scenario['s_doctor_max'],
            'Tmax': scenario['Tmax'],
            'nurse_price': scenario['nurse_price'],
            'doctor_price': scenario['doctor_price'],
            'total_cost': total_cost,
            'patient_loss': patient_loss,
            'hospital_overload': hospital_overload,
            'system_stable': system_stable,
            'nurse_utilization': min(nurse_utilization, 1.0),
            'doctor_utilization': min(doctor_utilization, 1.0),
            'optimal_nurses': optimal_nurses,
            'optimal_doctors': optimal_doctors
        }

        # 添加详细指标
        if metrics is not None:
            sample.update({
                'nurse_queue_time': metrics['nurse_queue_time'],
                'doctor_queue_time': metrics['doctor_queue_time'],
                'system_total_time': metrics['system_total_time']
            })
        else:
            sample.update({
                'nurse_queue_time': scenario['Tmax'] * 2,
                'doctor_queue_time': scenario['Tmax'] * 2,
                'system_total_time': scenario['Tmax'] * 4
            })

        samples.append(sample)

    return samples


def clean_data(df, max_loss_rate=0.6, max_overload_rate=0.7, min_stability_rate=0.5):
    """
    数据清洗：调整急诊场景的容忍度
    急诊场景允许更高的过载率和流失率
    """
    print("开始数据清洗...")
    print(f"原始数据量: {len(df)}")

    # 按场景计算统计指标
    scenario_stats = []
    for scenario in df['scenario'].unique():
        scenario_data = df[df['scenario'] == scenario]
        loss_rate = scenario_data['patient_loss'].mean()
        overload_rate = scenario_data['hospital_overload'].mean()
        stability_rate = scenario_data['system_stable'].mean()

        scenario_stats.append({
            'scenario': scenario,
            'loss_rate': loss_rate,
            'overload_rate': overload_rate,
            'stability_rate': stability_rate,
            'sample_count': len(scenario_data)
        })

        print(f"场景 {scenario}:")
        print(f"  流失率: {loss_rate:.3f} (阈值: {max_loss_rate})")
        print(f"  过载率: {overload_rate:.3f} (阈值: {max_overload_rate})")
        print(f"  稳定率: {stability_rate:.3f} (阈值: {min_stability_rate})")

    # 筛选符合条件的场景（急诊场景放宽标准）
    valid_scenarios = []
    for stat in scenario_stats:
        if (stat['loss_rate'] <= max_loss_rate and
            stat['overload_rate'] <= max_overload_rate and
            stat['stability_rate'] >= min_stability_rate):
            valid_scenarios.append(stat['scenario'])
            print(f"✓ 保留场景: {stat['scenario']}")
        else:
            print(f"✗ 移除场景: {stat['scenario']}")
            print(f"  原因: 流失率({stat['loss_rate']:.3f}>{max_loss_rate}) 或 "
                  f"过载率({stat['overload_rate']:.3f}>{max_overload_rate}) 或 "
                  f"稳定率({stat['stability_rate']:.3f}<{min_stability_rate})")

    # 如果没有场景通过筛选，进一步放宽标准
    if not valid_scenarios:
        print("\n警告：没有场景通过筛选，自动放宽标准...")
        max_loss_rate = 0.8
        max_overload_rate = 0.8
        min_stability_rate = 0.3

        for stat in scenario_stats:
            if (stat['loss_rate'] <= max_loss_rate and
                stat['overload_rate'] <= max_overload_rate and
                stat['stability_rate'] >= min_stability_rate):
                valid_scenarios.append(stat['scenario'])
                print(f"✓ 放宽标准后保留场景: {stat['scenario']}")

    # 过滤数据（保留更多急诊数据）
    cleaned_df = df[df['scenario'].isin(valid_scenarios)].copy()

    # 进一步移除极端样本（放宽条件）
    before_filter = len(cleaned_df)
    cleaned_df = cleaned_df[
        (cleaned_df['system_stable'] == 1) &  # 只保留稳定系统
        (cleaned_df['nurse_utilization'] <= 0.99) &  # 进一步放宽利用率限制
        (cleaned_df['doctor_utilization'] <= 0.99)
    ].copy()

    print(f"极端样本过滤: {before_filter} -> {len(cleaned_df)}")
    print(f"清洗后数据量: {len(cleaned_df)}")
    print(f"数据保留率: {len(cleaned_df)/len(df)*100:.1f}%")

    return cleaned_df


def main():
    """主函数：生成急诊场景数据"""
    all_samples = []

    for scenario in scenarios:
        print(f"生成场景: {scenario['name']}")
        samples = generate_samples_with_constraints(scenario)
        all_samples.extend(samples)
        print(f"  生成样本数: {len(samples)}")

    # 转换为DataFrame
    df = pd.DataFrame(all_samples)

    # 数据清洗
    cleaned_df = clean_data(df)

    # 保存数据
    cleaned_df.to_csv('emergency_hospital_data.csv', index=False)
    print(f"数据已保存到 emergency_hospital_data.csv")

    # 打印统计信息
    print("\n=== 数据统计 ===")
    print(f"总样本数: {len(cleaned_df)}")
    print(f"流失样本数: {cleaned_df['patient_loss'].sum()}")
    print(f"过载样本数: {cleaned_df['hospital_overload'].sum()}")
    print(f"流失率: {cleaned_df['patient_loss'].mean():.3f}")
    print(f"过载率: {cleaned_df['hospital_overload'].mean():.3f}")

    return cleaned_df


if __name__ == "__main__":
    main()