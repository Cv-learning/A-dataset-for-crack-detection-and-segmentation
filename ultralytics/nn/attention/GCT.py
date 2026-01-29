import torch
import torch.nn as nn
import torch.nn.functional as F

class GCT(nn.Module):
    def __init__(self, channels, c=2, eps=1e-5):
        """
        GCT (Global Context Transformer) 模块。

        Args:
            channels (int): 输入的通道数。（虽然当前计算逻辑中未直接使用，但保留用于接口一致性）
            c (float): 指数函数中的缩放因子，默认为 2。
            eps (float): 用于 sqrt 中的微小值，以保证数值稳定性，默认为 1e-5。
        """
        super(GCT, self).__init__()
        # 全局平均池化层，将空间维度压缩为 1x1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.eps = eps
        self.c = c
        # 注意: 'channels' 参数保留用于接口一致性，但在核心计算逻辑中未直接使用
        # 因为统计量是根据输入张量的现有通道数计算的。

    def forward(self, x):
        # 输入 x 的形状: (N, C, H, W)
        
        # 步骤 1: 全局平均池化，获取每个通道的全局响应
        # 输出 y 的形状: (N, C, 1, 1)
        y = self.avgpool(x) 
        
        # 步骤 2: 计算每个通道的方差 (Var = E[x^2] - (E[x])^2)
        # 先计算 x^2 的全局平均池化
        y_squared = self.avgpool(x * x) # 形状: (N, C, 1, 1)
        # 计算方差
        var = y_squared - y * y # 形状: (N, C, 1, 1)

        # 步骤 3: 计算标准化项 T (类似标准差)
        # 在 sqrt 中加入 eps 以保证数值稳定性
        t = torch.sqrt(var + self.eps) # 形状: (N, C, 1, 1)

        # 步骤 4: 对池化后的特征 y 进行标准化 (类似 Z-Score)
        y_norm = y / t # 形状: (N, C, 1, 1)

        # 步骤 5: 应用指数变换，生成通道注意力权重
        # 原始代码公式: exp(-(y_norm ** 2 / 2 * self.c))
        # 这里解释为: exp(- (y_norm^2 * c) / 2)，更符合高斯函数形式
        y_transform = torch.exp(-(y_norm ** 2 * self.c) / 2.0) # 形状: (N, C, 1, 1)

        # 步骤 6: 将生成的权重应用到原始输入 x 上
        # 利用 PyTorch 的广播机制，无需显式 expand_as
        # (N, C, 1, 1) * (N, C, H, W) -> (N, C, H, W)
        return x * y_transform 

# 示例用法:
# gct_layer = GCT(channels=64, c=2, eps=1e-5)
# input_tensor = torch.randn(1, 64, 32, 32)
# output_tensor = gct_layer(input_tensor)
# print(output_tensor.shape) # 应为 torch.Size([1, 64, 32, 32])