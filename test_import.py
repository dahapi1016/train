print("基础库导入成功")

try:
    from train import ConstraintAwareLoss, ViolationAwareDataset, AdaptiveLoss, HospitalDataset
    print("成功导入 ConstraintAwareLoss, ViolationAwareDataset, AdaptiveLoss, HospitalDataset")
except Exception as e:
    print(f"导入失败: {e}")
    import traceback
    traceback.print_exc()

