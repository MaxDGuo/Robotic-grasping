import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock


class GenerativeResnet(GraspModel):
    # 
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        # TODO 1.定义第一层卷积 conv1（输入通道为 input_channels，输出通道为 channel_size，核大小为9，padding=4）
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        # TODO 2.编写下采样卷积 conv2 和 conv3（步长为2，实现特征图尺寸减半）
        self.conv2 = nn.Conv2d(channel_size, 2 * channel_size, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * channel_size)
        self.conv3 = nn.Conv2d(2 * channel_size, 4 * channel_size, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * channel_size)

        # TODO 3.构建多个残差块（建议使用5个 ResidualBlock）
        self.res1 = ResidualBlock(4 * channel_size, 4 * channel_size)
        self.res2 = ResidualBlock(4 * channel_size, 4 * channel_size)
        self.res3 = ResidualBlock(4 * channel_size, 4 * channel_size)
        self.res4 = ResidualBlock(4 * channel_size, 4 * channel_size)
        self.res5 = ResidualBlock(4 * channel_size, 4 * channel_size)

        # TODO 4: 编写上采样部分（conv4、conv5 为反卷积 ConvTranspose2d，conv6为最终输出调整）
        self.conv4 = nn.ConvTranspose2d(4 * channel_size, 2 * channel_size, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(2 * channel_size)
        self.conv5 = nn.ConvTranspose2d(2 * channel_size, channel_size, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)
        # ConvT2D版本的conv6（按图标注）
        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, padding=4, stride=1, output_padding=0)


        # TODO 5: 实现4个抓取相关输出（位置、cos、sin、宽度），均为 2x2 的卷积
        # Use 1x1 convs for final outputs so spatial size stays equal to input (e.g. 224x224)
        self.pos_output = nn.Conv2d(channel_size, output_channels, kernel_size=1, padding=0)
        self.cos_output = nn.Conv2d(channel_size, output_channels, kernel_size=1, padding=0)
        self.sin_output = nn.Conv2d(channel_size, output_channels, kernel_size=1, padding=0)
        self.width_output = nn.Conv2d(channel_size, output_channels, kernel_size=1, padding=0)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # GRCONVNET 结束
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        """
        前向传播函数：输入图像，输出四个抓取相关的图像（位置、角度cos/sin、宽度）
        """
        # TODO 7: 完成下采样部分的特征提取操作。
        # 将输入图像依次通过三层卷积、BatchNorm 和 ReLU 激活函数，逐步提取特征并缩小空间尺寸。
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # TODO 8: 利用残差结构增强深层特征表达能力。
        # 通过五个连续的 ResidualBlock 堆叠，使网络在保持特征图尺寸的同时具有更强的建模能力。
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        # TODO 9: 执行特征图的上采样操作。
        # 使用反卷积（转置卷积）逐步恢复图像空间尺寸，并通过 BatchNorm 与 ReLU 保持稳定性。
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # TODO 10: 使用一层卷积整合上采样后的特征图，为最终输出做准备。
        x = self.conv6(x)

        # TODO 11: 根据是否启用 Dropout，输出四个抓取预测图。
        # 包括抓取位置图（pos）、抓取角度的 cos/sin 表示（cos, sin）以及抓取宽度图（width）。
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

            # 前向传播 结束

        return pos_output, cos_output, sin_output, width_output
