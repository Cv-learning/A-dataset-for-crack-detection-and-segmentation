import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self,
                 in_features,  # 输入特征数量
                 out_features,  # 输出特征数量
                 kernel_size=(3, 3),  # 卷积核大小
                 stride=(1, 1),  # 步长
                 padding=(1, 1),  # 填充
                 dilation=(1, 1),  # 空洞卷积率
                 norm_type='bn',  # 归一化类型
                 activation=True,  # 是否使用激活函数
                 use_bias=True,  # 是否使用偏置
                 groups=1  # 分组卷积数量
                 ):
        super().__init__()

        # 定义卷积层
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        # 定义归一化层
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        
        # 定义激活函数
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)

        if self.act:
            x = self.relu(x)
        return x
    

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # 重新排列张量维度 (B, H, W, C)
        B, H, W, C = x.shape
        P = self.patch_size

        # 局部分支
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # 将特征分块 (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # 重新调整形状 (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # 对最后一维取均值 (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # 全连接层 (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # 层归一化 (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # 全连接层 (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # 计算注意力 (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # 应用注意力权重 (B, H/P*W/P, output_dim)

        # 计算与提示向量的余弦相似度
        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # (B, N, 1)
        mask = cos_sim.clamp(0, 1)  # 计算掩码
        local_out = local_out * mask  # 应用掩码
        local_out = local_out @ self.top_down_transform  # 变换

        # 恢复形状
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)  # 插值恢复原始尺寸
        output = self.conv(local_out)  # 卷积层

        return output


class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel,2)+b)/gamma))  # 计算卷积核大小
        kernel_size = k if k % 2 else k+1  # 保证卷积核大小为奇数
        padding = kernel_size // 2  # 填充
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)  # 自适应平均池化
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),  # 1D卷积层
            nn.Sigmoid()  # Sigmoid激活函数
        )

    def forward(self, x):
        out = self.pool(x)  # 平均池化
        out = out.view(x.size(0), 1, x.size(1))  # 调整形状
        out = self.conv(out)  # 1D卷积
        out = out.view(x.size(0), x.size(1), 1, 1)  # 调整形状
        return out * x  # 注意力加权


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)  # 2D卷积层
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        maxout, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        out = torch.cat([avgout, maxout], dim=1)  # 连接
        out = self.sigmoid(self.conv2d(out))  # 卷积和激活
        return out * x  # 注意力加权


class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()

        # 定义跳跃连接卷积块
        self.skip = conv_block(in_features=in_features,
                               out_features=filters,
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               norm_type='bn',
                               activation=False)
        
        # 定义连续卷积块
        self.c1 = conv_block(in_features=in_features,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c2 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c3 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        
        # 定义空间注意力模块
        self.sa = SpatialAttentionModule()
        # 定义ECA模块
        self.cn = ECA(filters)
        # 定义局部和全局注意力模块
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        # 定义批归一化层、dropout层和激活函数
        self.bn1 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x):
        x_skip = self.skip(x)  # 跳跃连接输出
        x_lga2 = self.lga2(x_skip)  # 局部和全局注意力输出（大小为2的patch）
        x_lga4 = self.lga4(x_skip)  # 局部和全局注意力输出（大小为4的patch）
        x1 = self.c1(x)  # 第一个卷积块输出
        x2 = self.c2(x1)  # 第二个卷积块输出
        x3 = self.c3(x2)  # 第三个卷积块输出
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4  # 合并所有输出
        x = self.cn(x)  # ECA模块
        x = self.sa(x)  # 空间注意力模块
        x = self.drop(x)  # Dropout层
        x = self.bn1(x)  # 批归一化层
        x = self.relu(x)  # 激活函数
        return x

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
     
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
 
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

 

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
 
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
 
    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k_PPA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PPA(c_, c_) for _ in range(n)))

class C3k2_PPA(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(C3k_PPA(self.c, self.c, 2, shortcut, g) if c3k else PPA(self.c, self.c) for _ in range(n))

