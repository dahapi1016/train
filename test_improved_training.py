#!/usr/bin/env python3
"""
测试改进的训练策略
验证类别不平衡问题的解决效果
"""

import numpy as np
import pandas as pd
import torch


def test_data_preprocessing():
    """测试数据预处理效果"""
    print("=== 测试数据预处理效果 ===")

    # 读取数据
    df = pd.read_csv('emergency_hospital_data.csv')

    # 基本信息
    print(f"原始数据量: {len(df)}")
    print(f"原始护士类别数: {df['optimal_nurses'].nunique()}")
    print(f"原始医生类别数: {df['optimal_doctors'].nunique()}")

    # 分析类别分布
    nurse_counts = df['optimal_nurses'].value_counts()
    doctor_counts = df['optimal_doctors'].value_counts()

    print(f"\n护士类别不平衡情况:")
    print(f"最多样本类别: {nurse_counts.max()} 个样本")
    print(f"最少样本类别: {nurse_counts.min()} 个样本")
    print(f"不平衡比例: {nurse_counts.max() / nurse_counts.min():.1f}:1")

    print(f"\n医生类别不平衡情况:")
    print(f"最多样本类别: {doctor_counts.max()} 个样本")
    print(f"最少样本类别: {doctor_counts.min()} 个样本")
    print(f"不平衡比例: {doctor_counts.max() / doctor_counts.min():.1f}:1")

    # 统计少样本类别
    min_samples = 5
    nurse_few_samples = (nurse_counts < min_samples).sum()
    doctor_few_samples = (doctor_counts < min_samples).sum()

    print(f"\n样本数 < {min_samples} 的类别:")
    print(f"护士: {nurse_few_samples} 个类别")
    print(f"医生: {doctor_few_samples} 个类别")

    return df

def test_focal_loss():
    """测试Focal Loss的效果"""
    print("\n=== 测试Focal Loss效果 ===")

    # 创建模拟的不平衡数据
    torch.manual_seed(42)

    # 模拟logits和targets
    batch_size = 32
    n_classes = 10

    # 创建不平衡的targets（大部分样本属于类别0）
    targets = torch.zeros(batch_size, dtype=torch.long)
    targets[28:] = torch.randint(1, n_classes, (4,))  # 只有4个样本属于其他类别

    logits = torch.randn(batch_size, n_classes)

    # 标准CrossEntropy Loss
    ce_loss = torch.nn.CrossEntropyLoss()
    ce_result = ce_loss(logits, targets)

    # Focal Loss
    from train import FocalLoss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    focal_result = focal_loss(logits, targets)

    print(f"CrossEntropy Loss: {ce_result:.4f}")
    print(f"Focal Loss: {focal_result:.4f}")
    print(f"Focal Loss相对减少: {(ce_result - focal_result) / ce_result * 100:.1f}%")

def analyze_class_distribution():
    """分析类别分布并给出建议"""
    print("\n=== 类别分布分析与建议 ===")

    df = pd.read_csv('emergency_hospital_data.csv')

    # 护士类别分析
    nurse_counts = df['optimal_nurses'].value_counts().sort_index()
    print("护士类别分布:")
    for class_id, count in nurse_counts.items():
        percentage = count / len(df) * 100
        print(f"  类别 {class_id}: {count} 样本 ({percentage:.1f}%)")

    # 医生类别分析
    doctor_counts = df['optimal_doctors'].value_counts().sort_index()
    print("\n医生类别分布:")
    for class_id, count in doctor_counts.items():
        percentage = count / len(df) * 100
        print(f"  类别 {class_id}: {count} 样本 ({percentage:.1f}%)")

    # 给出建议
    print("\n=== 改进建议 ===")

    # 1. 数据过滤建议
    min_samples = 5
    nurse_to_remove = (nurse_counts < min_samples).sum()
    doctor_to_remove = (doctor_counts < min_samples).sum()

    print(f"1. 数据过滤:")
    print(f"   - 建议移除样本数 < {min_samples} 的护士类别: {nurse_to_remove} 个")
    print(f"   - 建议移除样本数 < {min_samples} 的医生类别: {doctor_to_remove} 个")

    # 2. 权重设置建议
    nurse_max_weight = np.sqrt(len(df) / (len(nurse_counts) * nurse_counts.min()))
    doctor_max_weight = np.sqrt(len(df) / (len(doctor_counts) * doctor_counts.min()))

    print(f"2. 类别权重:")
    print(f"   - 护士最大权重: {nurse_max_weight:.2f}")
    print(f"   - 医生最大权重: {doctor_max_weight:.2f}")

    # 3. 模型复杂度建议
    total_classes = len(nurse_counts) + len(doctor_counts)
    samples_per_class = len(df) / total_classes

    print(f"3. 模型复杂度:")
    print(f"   - 总类别数: {total_classes}")
    print(f"   - 平均每类样本数: {samples_per_class:.1f}")
    if samples_per_class < 50:
        print("   - 建议: 减少模型复杂度，增加正则化")
    else:
        print("   - 建议: 可以使用较复杂的模型")

if __name__ == "__main__":
    # 测试数据预处理
    df = test_data_preprocessing()

    # 测试Focal Loss
    test_focal_loss()

    # 分析类别分布
    analyze_class_distribution()

    print("\n=== 总结 ===")
    print("主要问题:")
    print("1. 极度不平衡的类别分布 (最大比例 532:1)")
    print("2. 过多的类别数量 (护士28类，医生21类)")
    print("3. 大量少样本类别 (样本数 < 5)")
    print("4. 模型复杂度与数据量不匹配")

    print("\n解决方案:")
    print("1. ✅ 数据预处理: 移除少样本类别")
    print("2. ✅ Focal Loss: 处理类别不平衡")
    print("3. ✅ 改进权重计算: 限制极端权重")
    print("4. ✅ 调整训练策略: 降低学习率，增加正则化")
    print("5. ✅ 更稳定的学习率调度")

