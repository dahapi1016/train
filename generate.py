import math
import csv
import random

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
    P0 = compute_P0(lambd, mu, s)
    if P0 is None:
        return None, None
    rho = lambd / (s * mu)
    numerator = P0 * ((lambd / mu) ** s) * (mu / (s * mu - lambd))
    denominator = math.factorial(s) * (1 - rho) ** 2
    Wq = numerator / denominator
    W_total = Wq + 1 / mu
    return Wq, W_total

scenarios = [
    {   # 场景1: 小型医院非紧急（门诊）
        'name': 'small_non_emergency',
        's_nurse_max': 5,          # 最大护士数
        's_doctor_max': 3,         # 最大医生数
        'lambda_range': (0.5, 12), # 患者到达率 (人/小时)
        'mu_nurse_range': (3, 6),  # 护士服务率 (人/小时)
        'mu_doctor_range': (5, 8), # 医生服务率 (人/小时)
        'Tmax': 2.0,               # 最长总等待时间 (小时)
        'nurse_price': 100,        # 护士单价 (元/小时)
        'doctor_price': 300,       # 医生单价 (元/小时)
        'num_samples': 3000
    },
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
    {   # 场景3: 大型医院非紧急（综合门诊）
        'name': 'large_non_emergency',
        's_nurse_max': 20,
        's_doctor_max': 10,
        'lambda_range': (5, 25),
        'mu_nurse_range': (2, 5),
        'mu_doctor_range': (3, 6),
        'Tmax': 3.0,
        'nurse_price': 80,
        'doctor_price': 250,
        'num_samples': 3000
    },
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


def generate_samples(scenario):
    samples = []
    for _ in range(scenario['num_samples']):
        # 生成医护人员服务率
        mu_nurse = random.uniform(*scenario['mu_nurse_range'])
        mu_doctor = random.uniform(*scenario['mu_doctor_range'])

        # 计算最大允许lambda
        lambda_max = min(
            scenario['s_nurse_max'] * mu_nurse,
            scenario['s_doctor_max'] * mu_doctor
        ) - 1e-9  # 保证严格小于

        lambda_min = max(min(scenario['lambda_range']), 0.1)  # 取元组第一个元素
        lambda_max_actual = min(max(scenario['lambda_range']), lambda_max)  # 取元组第二个元素

        if lambda_min >= lambda_max_actual:
            continue

        lambd = random.uniform(lambda_min, lambda_max_actual)

        samples.append({
            'lambda': lambd,
            'mu_nurse': mu_nurse,
            'mu_doctor': mu_doctor,
            's_nurse_max': scenario['s_nurse_max'],
            's_doctor_max': scenario['s_doctor_max'],
            'Tmax': scenario['Tmax'],
            'nurse_price': scenario['nurse_price'],
            'doctor_price': scenario['doctor_price'],
            'scenario_name': scenario['name']
        })
    return samples


def find_optimal_solution(sample):
    min_cost = float('inf')
    optimal = None
    lambd = sample['lambda']
    mu_n = sample['mu_nurse']
    mu_d = sample['mu_doctor']
    max_n = sample['s_nurse_max']
    max_d = sample['s_doctor_max']
    Tmax = sample['Tmax']
    price_n = sample['nurse_price']
    price_d = sample['doctor_price']

    # 计算最小需要的医护人员数量
    min_n = math.ceil(lambd / mu_n) + 1 if mu_n > 0 else max_n
    min_d = math.ceil(lambd / mu_d) + 1 if mu_d > 0 else max_d
    min_n = max(1, min(min_n, max_n))
    min_d = max(1, min(min_d, max_d))

    # 遍历所有可能的组合
    for s_n in range(min_n, max_n + 1):
        for s_d in range(min_d, max_d + 1):
            # 计算等待时间
            _, Wn = compute_metrics(lambd, mu_n, s_n)
            _, Wd = compute_metrics(lambd, mu_d, s_d)

            if Wn is None or Wd is None:
                continue

            total_wait = Wn + Wd
            if total_wait > Tmax:
                continue

            # 计算成本
            cost = s_n * price_n + s_d * price_d
            if cost < min_cost:
                min_cost = cost
                optimal = (s_n, s_d)

    return optimal

def main():
    with open('hospital_training_data.csv', 'w', newline='') as csvfile:
        fieldnames = [
            'scenario', 'lambda', 'mu_nurse', 'mu_doctor',
            's_nurse_max', 's_doctor_max', 'Tmax',
            'nurse_price', 'doctor_price',
            'optimal_nurses', 'optimal_doctors', 'min_cost'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for scenario in scenarios:
            samples = generate_samples(scenario)
            print(f"Processing {scenario['name']} ({len(samples)} valid samples)")

            for i, sample in enumerate(samples):
                if (i + 1) % 100 == 0:
                    print(f"  Processing sample {i + 1}/{len(samples)}")

                optimal = find_optimal_solution(sample)
                if not optimal:
                    continue

                s_n, s_d = optimal
                writer.writerow({
                    'scenario': sample['scenario_name'],
                    'lambda': round(sample['lambda'], 2),
                    'mu_nurse': round(sample['mu_nurse'], 2),
                    'mu_doctor': round(sample['mu_doctor'], 2),
                    's_nurse_max': sample['s_nurse_max'],
                    's_doctor_max': sample['s_doctor_max'],
                    'Tmax': sample['Tmax'],
                    'nurse_price': sample['nurse_price'],
                    'doctor_price': sample['doctor_price'],
                    'optimal_nurses': s_n,
                    'optimal_doctors': s_d,
                    'min_cost': s_n * sample['nurse_price'] + s_d * sample['doctor_price']
                })


if __name__ == "__main__":
    main()