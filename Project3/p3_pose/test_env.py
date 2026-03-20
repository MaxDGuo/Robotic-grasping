# 导入torch库
import torch

# 1. 查看torch编译时使用的CUDA版本（核心查询命令）
print("PyTorch 对应的 CUDA 版本：", torch.version.cuda)

# 2. 额外验证：查看当前环境是否支持CUDA（GPU是否可用）
print("CUDA 是否可用：", torch.cuda.is_available())

# 3. 可选：查看显卡数量和显卡名称（进一步确认GPU环境）
if torch.cuda.is_available():
    print("显卡数量：", torch.cuda.device_count())
    print("显卡名称：", torch.cuda.get_device_name(0))
