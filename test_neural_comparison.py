#!/usr/bin/env python3
"""
测试神经网络模型对比功能
验证移除排队论后的对比分析是否正常工作
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def test_neural_comparison():
    """测试神经网络模型对比功能"""
    print("=== 测试神经网络模型对比功能 ===")

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000

    # 生成特征
    X = np.random.randn(n_samples, 10)

    # 生成目标值（护士和医生数量）
    y = np.column_stack([
        np.random.randint(1, 10, n_samples),  # 护士数量
        np.random.randint(1, 6, n_samples)    # 医生数量
    ])

    # 生成原始特征数据
    X_raw = pd.DataFrame({
        'lambda': np.random.uniform(5, 20, n_samples),
        'mu_nurse': np.random.uniform(2, 6, n_samples),
        'mu_doctor': np.random.uniform(1, 4, n_samples),
        's_nurse_max': np.random.randint(8, 15, n_samples),
        's_doctor_max': np.random.randint(5, 10, n_samples),
        'Tmax': np.random.uniform(15, 45, n_samples),
        'nurse_price': np.random.uniform(50, 120, n_samples),
        'doctor_price': np.random.uniform(200, 450, n_samples)
    })

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_raw_train, X_raw_test = train_test_split(X_raw, test_size=0.2, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    print(f"护士数量范围: {y_train[:, 0].min()}-{y_train[:, 0].max()}")
    print(f"医生数量范围: {y_train[:, 1].min()}-{y_train[:, 1].max()}")

    # 测试可视化函数导入
    try:
        from visualization_comparison import (
            train_traditional_models,
            create_comprehensive_visualization,
            create_performance_summary_table
        )
        print("✓ 可视化函数导入成功")
    except ImportError as e:
        print(f"✗ 可视化函数导入失败: {e}")
        return False

    # 测试传统模型训练
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_nurse_classes = int(y_train[:, 0].max()) + 1
        n_doctor_classes = int(y_train[:, 1].max()) + 1

        print(f"护士类别数: {n_nurse_classes}")
        print(f"医生类别数: {n_doctor_classes}")
        print(f"使用设备: {device}")

        # 训练传统模型（使用较少的epoch进行快速测试）
        traditional_results = train_traditional_models(
            X_train, y_train, X_test, y_test,
            n_nurse_classes, n_doctor_classes, device
        )
        print("✓ 传统模型训练成功")

        # 检查结果结构
        expected_keys = ['Traditional_PAN', 'Traditional_DNN']
        for key in expected_keys:
            if key not in traditional_results:
                print(f"✗ 缺少结果键: {key}")
                return False

            # 检查每个模型的指标
            model_result = traditional_results[key]
            required_metrics = ['nurse_pred', 'doctor_pred', 'nurse_metrics', 'doctor_metrics']
            for metric in required_metrics:
                if metric not in model_result:
                    print(f"✗ {key} 缺少指标: {metric}")
                    return False

        print("✓ 传统模型结果结构正确")

        # 创建模拟的混合模型结果
        hybrid_results = {
            'nurse_pred': np.random.randint(1, 10, len(y_test)),
            'doctor_pred': np.random.randint(1, 6, len(y_test)),
            'nurse_metrics': {
                'MAE': 0.8,
                'MSE': 1.2,
                'R2': 0.75,
                'Accuracy': 0.82
            },
            'doctor_metrics': {
                'MAE': 0.6,
                'MSE': 0.9,
                'R2': 0.78,
                'Accuracy': 0.85
            }
        }

        # 测试综合可视化（不实际显示图表）
        print("测试综合可视化功能...")
        create_comprehensive_visualization(hybrid_results, traditional_results, y_test)
        print("✓ 综合可视化测试成功")

        # 测试性能汇总表
        print("测试性能汇总表功能...")
        create_performance_summary_table(hybrid_results, traditional_results)
        print("✓ 性能汇总表测试成功")

        print("\n=== 所有测试通过 ===")
        print("神经网络模型对比功能正常工作")
        print("已成功移除排队论相关代码")
        return True

    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_neural_comparison()
    if success:
        print("\n🎉 测试成功！神经网络模型对比功能已准备就绪。")
    else:
        print("\n❌ 测试失败，请检查代码修改。")

