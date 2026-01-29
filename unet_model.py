# file:net.py
import torch
from torch import nn
import torch.nn.functional as F


# ---------- 注意力模块 ---------- #
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights


# ---------- 在U-Net网络中，卷积是两次两次进行的，每次卷积都进行两次 ---------- #
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            # ------ 参数分别为：输入通道、输出通道、卷积核大小、步长、padding、padding的模式
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.1),  # 降低dropout率
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.1),  # 降低dropout率
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


# ---------- 下采样模块 ---------- #
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            # ------ 参数分别为：输入通道、输出通道、卷积核大小、步长、padding ------ #
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


# ---------- 上采样模块，这里采用最临近插值法进行上采样 ---------- #
class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel//2, 1, 1)

    def forward(self, x, feature_map):
        # ------ 使用最临近插值法进行上采样 ------ #
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


# ---------- 定义U-Net网络结构 ---------- #
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 减少通道数以降低显存占用
        self.c1 = Conv_Block(3, 32)  # 从64减少到32
        self.attn1 = AttentionBlock(32)  # 添加注意力
        self.d1 = DownSample(32)
        self.c2 = Conv_Block(32, 64)  # 从128减少到64
        self.attn2 = AttentionBlock(64)
        self.d2 = DownSample(64)
        self.c3 = Conv_Block(64, 128)  # 从256减少到128
        self.attn3 = AttentionBlock(128)
        self.d3 = DownSample(128)
        self.c4 = Conv_Block(128, 256)  # 从512减少到256
        self.attn4 = AttentionBlock(256)
        self.d4 = DownSample(256)
        self.c5 = Conv_Block(256, 512)  # 从1024减少到512
        self.attn5 = AttentionBlock(512)
        self.u1 = UpSample(512)
        self.c6 = Conv_Block(512, 256)  # 调整以匹配新的通道数
        self.u2 = UpSample(256)
        self.c7 = Conv_Block(256, 128)  # 调整以匹配新的通道数
        self.u3 = UpSample(128)
        self.c8 = Conv_Block(128, 64)   # 调整以匹配新的通道数
        self.u4 = UpSample(64)
        self.c9 = Conv_Block(64, 32)    # 调整以匹配新的通道数
        # 修改输出层，输出单通道用于二分类分割
        self.out = nn.Conv2d(32, 1, 3, 1, 1)  # 调整输入通道为32
        self.Th = nn.Sigmoid()

    def forward(self, x):
        R1 = self.c1(x)
        R1 = self.attn1(R1)  # 应用注意力
        R2 = self.c2(self.d1(R1))
        R2 = self.attn2(R2)
        R3 = self.c3(self.d2(R2))
        R3 = self.attn3(R3)
        R4 = self.c4(self.d3(R3))
        R4 = self.attn4(R4)
        R5 = self.c5(self.d4(R4))
        R5 = self.attn5(R5)
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))  # 输出单通道概率图


if __name__ == '__main__':
    x = torch.randn(2, 3, 448, 448)  # 修改为448x448
    net = UNet()
    print(net(x).shape)  # 应该输出 torch.Size([2, 1, 448, 448])
