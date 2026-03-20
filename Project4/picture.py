import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文环境用这个，中文换 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 第一步：填入从日志提取的数据 ----------------------
# 你需要把所有Epoch的数值补充完整（这里是示例，你替换成日志里的全部值）
epochs = list(range(50))  # 0-49 Epoch
train_loss = [
    0.0628, 0.1029, 0.0351, 0.0710, 0.0353, 0.0572, 0.0322, 0.1044, 0.0822, 0.0485,
    0.0867, 0.0363, 0.1538, 0.0456, 0.0590, 0.0450, 0.0474, 0.1743, 0.0247, 0.0691,
    0.0340, 0.0283, 0.1224, 0.1300, 0.1562, 0.0994, 0.0601, 0.0344, 0.0601, 0.1561,
    0.1002, 0.0482, 0.0732, 0.1220, 0.0834, 0.0174, 0.0305, 0.0823, 0.0435, 0.1007,
    0.0232, 0.0346, 0.1366, 0.0234, 0.0209, 0.0285, 0.0503, 0.0915, 0.0530, 0.0236
]
val_iou = [
    0.4831, 0.7416, 0.8539, 0.7978, 0.9101, 0.8652, 0.3933, 0.9101, 0.9326, 0.9551,
    0.9213, 0.9551, 0.9213, 0.8652, 0.9213, 0.8876, 0.8876, 0.8652, 0.9101, 0.8989,
    0.8315, 0.9326, 0.9663, 0.9775, 0.9213, 0.9551, 0.9438, 0.9438, 0.9551, 0.9551,
    0.9438, 0.8764, 0.9438, 0.9551, 0.9101, 0.9326, 0.9213, 0.9101, 0.9101, 0.8764,
    0.8989, 0.9438, 0.8764, 0.9213, 0.9213, 0.8989, 0.8989, 0.8876, 0.9326, 0.9663
]

# ---------------------- 第二步：绘制双轴图 ----------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制训练Loss（左轴）
color1 = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color=color1)
ax1.plot(epochs, train_loss, color=color1, marker='o', markersize=3, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# 创建右轴绘制验证IoU
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Validation IoU', color=color2)
ax2.plot(epochs, val_iou, color=color2, marker='s', markersize=3, label='Val IoU')
ax2.tick_params(axis='y', labelcolor=color2)

# 添加标题和图例
fig.suptitle('Training Loss and Validation IoU vs Epoch', fontsize=14, fontweight='bold')
fig.tight_layout()  # 避免标签重叠

# 保存图片（可直接提交）
plt.savefig('loss_iou_curve.png', dpi=300, bbox_inches='tight')
plt.show()
