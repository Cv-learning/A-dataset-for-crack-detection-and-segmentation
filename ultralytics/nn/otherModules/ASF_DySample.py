import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return self.sigmoid(x_out)


class AFFModule(nn.Module):
    """自适应特征融合模块 - 用于多尺度特征融合"""
    def __init__(self, channels, reduction=16):
        super(AFFModule, self).__init__()
        self.channels = channels
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 2, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, feat1, feat2):
        """
        自适应融合两个特征图
        Args:
            feat1: 特征图1
            feat2: 特征图2（通常是上采样后的特征）
        Returns:
            融合后的特征图
        """
        # 确保两个特征图尺寸一致
        if feat1.size()[2:] != feat2.size()[2:]:
            feat2 = F.interpolate(feat2, size=feat1.size()[2:], mode='bilinear', align_corners=False)
        
        concat_features = torch.cat([feat1, feat2], dim=1)
        weights = self.weight_net(concat_features)
        w1, w2 = weights.chunk(2, dim=1)
        
        fused_features = w1 * feat1 + w2 * feat2
        return fused_features


class DySampleModule(nn.Module):
    """动态采样模块 - 用于精确的上采样"""
    def __init__(self, in_channels, scale_factor=2, groups=4):
        super(DySampleModule, self).__init__()
        self.scale_factor = scale_factor
        self.groups = groups
        self.in_channels = in_channels
        
        # 生成动态权重的卷积层
        self.offset_conv = nn.Conv2d(in_channels, groups * 2, 3, 1, 1)
        self.mask_conv = nn.Conv2d(in_channels, groups, 3, 1, 1)
        
        # 用于将动态调整应用到所有通道的卷积层
        self.adjustment_conv = nn.Conv2d(groups, in_channels, kernel_size=1, groups=min(groups, in_channels))
        
        # 位置编码
        self.position_bias = nn.Parameter(torch.zeros(scale_factor, scale_factor))
        
        # 初始化权重
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.constant_(self.mask_conv.weight, 0)
        nn.init.constant_(self.mask_conv.bias, 0)
        nn.init.constant_(self.adjustment_conv.weight, 0.1)  # 小的初始权重
        nn.init.constant_(self.adjustment_conv.bias, 0)

    def forward(self, x):
        """
        动态上采样
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            上采样后的特征图 [B, C, H*scale_factor, W*scale_factor]
        """
        B, C, H, W = x.size()
        
        # 生成动态偏移和掩码
        offset = self.offset_conv(x)  # [B, 2*groups, H, W]
        mask = self.mask_conv(x)      # [B, groups, H, W]
        
        # 标准双线性插值上采样
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        # 对offset进行上采样以匹配输出尺寸
        offset_up = F.interpolate(offset, size=(H * self.scale_factor, W * self.scale_factor), 
                                  mode='bilinear', align_corners=False)  # [B, 2*groups, H*sf, W*sf]
        
        # 对mask进行上采样并应用softmax
        mask_up = F.interpolate(mask, size=(H * self.scale_factor, W * self.scale_factor), 
                                mode='bilinear', align_corners=False)  # [B, groups, H*sf, W*sf]
        mask_up = torch.softmax(mask_up.flatten(2), dim=-1).view_as(mask_up)  # 沿空间维度softmax
        
        # 将mask通过卷积扩展到所有通道
        adjustment_map = self.adjustment_conv(mask_up)  # [B, C, H*sf, W*sf]
        
        # 最终输出：基础插值 + 动态调整
        refined_output = upsampled + adjustment_map
        
        return refined_output


class ASFModule(nn.Module):
    """注意力空间融合模块 - 用于特征增强"""
    def __init__(self, in_channels, reduction=16):
        super(ASFModule, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 通道注意力
        channel_weight = self.channel_att(x)
        attended_by_channel = x * channel_weight
        
        # 空间注意力
        spatial_weight = self.spatial_att(x)
        attended_by_spatial = x * spatial_weight
        
        # 特征融合
        concatenated = torch.cat([attended_by_channel, attended_by_spatial], dim=1)
        out = self.conv(concatenated)
        out = self.norm(out)
        out = self.act(out)
        
        return out


class ASFDySample(nn.Module):
    """
    ASF-DySample: 结合自适应特征融合与动态采样的上采样模块
    专为裂缝检测等需要高精度定位的任务设计
    """
    def __init__(self, in_channels, out_channels=None, scale_factor=2, groups=4, reduction=16):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数，默认为None，使用in_channels
            scale_factor: 上采样倍数
            groups: 动态采样分组数
            reduction: 注意力机制压缩比
        """
        super(ASFDySample, self).__init__()
        
        self.scale_factor = scale_factor
        self.out_channels = out_channels or in_channels
        
        # 1. 动态上采样模块 - 提升空间分辨率
        self.dysample = DySampleModule(in_channels, scale_factor, groups)
        
        # 2. 注意力空间融合模块 - 增强特征表示能力
        self.asf = ASFModule(in_channels, reduction)
        
        # 3. 自适应特征融合模块 - 融合多尺度信息
        self.aff = AFFModule(in_channels, reduction)
        
        # 4. 输出调整层 - 调整通道数和进一步特征提取
        if in_channels != self.out_channels:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.out_conv = nn.Identity()
        
        # 5. 残差连接（可选）
        self.use_residual = in_channels == self.out_channels
        if self.use_residual:
            self.residual_conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False)

    def forward(self, low_res_feat, high_res_feat=None):
        """
        前向传播
        Args:
            low_res_feat: 低分辨率特征（需要上采样的特征）
            high_res_feat: 高分辨率特征（可选，用于多尺度融合）
        Returns:
            处理后的特征图
        """
        # 步骤1: 动态上采样 - 提升特征分辨率
        upsampled_feat = self.dysample(low_res_feat)
        
        # 步骤2: 注意力空间融合 - 增强特征质量
        enhanced_feat = self.asf(upsampled_feat)
        
        # 步骤3: 自适应特征融合（如果有高分辨率特征输入）
        if high_res_feat is not None:
            fused_feat = self.aff(high_res_feat, enhanced_feat)
        else:
            fused_feat = enhanced_feat
        
        # 步骤4: 输出调整
        output = self.out_conv(fused_feat)
        
        # 步骤5: 残差连接（可选）
        if self.use_residual:
            residual = self.residual_conv(low_res_feat)
            # 调整残差尺寸以匹配输出
            if residual.size()[2:] != output.size()[2:]:
                residual = F.interpolate(residual, size=output.size()[2:], mode='bilinear', align_corners=False)
            output = output + residual
        
        return output


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建模块实例
    asf_dysample = ASFDySample(
        in_channels=256,
        out_channels=256,
        scale_factor=2,
        groups=4,
        reduction=16
    )
    
    # 创建测试输入
    low_res_input = torch.randn(1, 256, 20, 20)  # 低分辨率特征
    high_res_input = torch.randn(1, 256, 40, 40) # 高分辨率特征（可选）
    
    print("测试开始...")
    print(f"输入尺寸: {low_res_input.shape}")
    print(f"高分辨率特征尺寸: {high_res_input.shape}")
    
    # 前向传播
    try:
        output_with_high_res = asf_dysample(low_res_input, high_res_input)
        print(f"输出尺寸 (with high-res fusion): {output_with_high_res.shape}")
    except Exception as e:
        print(f"带高分辨率特征融合的前向传播出错: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        output_without_high_res = asf_dysample(low_res_input)  # 不使用高分辨率特征融合
        print(f"输出尺寸 (without high-res fusion): {output_without_high_res.shape}")
    except Exception as e:
        print(f"不带高分辨率特征融合的前向传播出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 验证输出尺寸是否正确
    expected_size = (1, 256, 40, 40)
    try:
        assert output_without_high_res.shape == expected_size, f"Expected {expected_size}, got {output_without_high_res.shape}"
        print("ASF-DySample模块测试通过！")
    except NameError:
        print("由于前面的错误，无法进行尺寸验证")
    except AssertionError as e:
        print(f"尺寸验证失败: {e}")