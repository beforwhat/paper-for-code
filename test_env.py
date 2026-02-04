import torch
import sys

print("="*50)
print(f"Python版本：{sys.version}")
print(f"PyTorch版本：{torch.__version__}")
print(f"CUDA是否可用：{torch.cuda.is_available()}")
print(f"GPU数量：{torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}：{torch.cuda.get_device_name(i)}")
    print(f"CUDA版本：{torch.version.cuda}")
    # 测试GPU计算
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"GPU矩阵乘法结果形状：{z.shape}")
print("="*50)