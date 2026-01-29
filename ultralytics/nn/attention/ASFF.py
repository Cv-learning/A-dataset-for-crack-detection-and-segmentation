import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Zoom_cat(nn.Module):
    """
    Zoom_cat module for adaptive scale fusion.
    Combines features from different scales by resizing them to the same dimensions
    and concatenating them together.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        # 1x1卷积用于调整通道数，使输出通道数等于输入总通道数的一部分
        self.adjust = nn.Conv2d(in_dim * 3, in_dim, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        """
        Args:
            x: List containing 3 feature tensors [large, medium, small] in decreasing order of spatial size
        Returns:
            Concatenated tensor combining features from all scales
        """
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]

        # Resize large feature map to target size using both pooling methods
        l_resized = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)

        # Resize small feature map to target size
        s_resized = F.interpolate(s, m.shape[2:], mode='nearest')

        # Concatenate all three feature maps along the channel dimension
        lms = torch.cat([l_resized, m, s_resized], dim=1)
        
        # 调整通道数以匹配输入
        lms = self.bn(self.adjust(lms))
        return lms


class ScalSeq(nn.Module):
    """
    ScalSeq module for attentional scale sequence fusion.
    Processes multi-scale features using 3D convolutions to capture scale relationships.
    """
    def __init__(self, channel):
        super(ScalSeq, self).__init__()
        self.channel = channel
        # 用于调整不同层级的通道数
        self.conv1 = nn.Conv2d(channel, channel, 1)
        self.conv2 = nn.Conv2d(channel, channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        """
        Args:
            x: List containing 3 feature tensors [p3, p4, p5] at different scales
        Returns:
            Fused tensor after processing with 3D convolutions
        """
        p3, p4, p5 = x[0], x[1], x[2]

        # Process p4 to match p3 size
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')

        # Process p5 to match p3 size
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')

        # Add dimension for 3D processing
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)

        # Concatenate along the new dimension
        combined = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)

        # Apply 3D convolution and processing
        conv_3d = self.conv3d(combined)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)  # Remove the extra dimension

        return x


class ChannelAttention(nn.Module):
    """
    Channel attention module that adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, channel, b=1, gamma=2):
        super(ChannelAttention, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class LocalAttention(nn.Module):
    """
    Local attention module that focuses on spatial relationships within feature maps.
    """
    def __init__(self, channel, reduction=16):
        super(LocalAttention, self).__init__()

        self.conv_1x1 = nn.Conv2d(
            in_channels=channel, 
            out_channels=channel//reduction, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(
            in_channels=channel//reduction, 
            out_channels=channel, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        self.F_w = nn.Conv2d(
            in_channels=channel//reduction, 
            out_channels=channel, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class AttentionModule(nn.Module):
    """
    Attention module that combines channel and local attention mechanisms.
    """
    def __init__(self, channel):
        super().__init__()
        self.channel_att = ChannelAttention(channel)
        self.local_att = LocalAttention(channel)

    def forward(self, x):
        """
        Args:
            x: List containing 2 feature tensors [input1, input2] to be fused
        Returns:
            Fused tensor after applying attention mechanisms
        """
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x


class ASFF(nn.Module):
    """
    Adaptive Scale Fusion Framework (ASFF) module that combines all ASF components.
    This is the main module that can be easily integrated into YOLOv8.
    """
    def __init__(self, level, channel=None):
        super(ASFF, self).__init__()
        self.level = level
        if channel:
            self.channel = channel
        else:
            # Default channel values for different levels
            default_channels = [64, 128, 256, 512, 1024]
            if level < len(default_channels):
                self.channel = default_channels[level]
            else:
                self.channel = default_channels[-1]  # Use the last one if level is out of bounds

        # 1x1卷积层用于确保输出通道数与输入一致
        self.reduce_conv = nn.Conv2d(self.channel * 3 if level <= 2 else self.channel * 2, 
                                    self.channel, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(self.channel)

        # Different operations based on the feature level
        if level <= 2:  # For smaller levels (higher resolution)
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.zoom_cat = Zoom_cat(self.channel)
        else:  # For larger levels (lower resolution)
            self.scal_seq = ScalSeq(self.channel)
            self.attention = AttentionModule(self.channel)

    def forward(self, x):
        """
        Args:
            x: List of feature tensors from different scales
        Returns:
            Adaptively fused feature tensor
        """
        # 过滤掉None值
        features = [xi for xi in x if xi is not None]
        
        if self.level <= 2:
            # Process using zoom_cat for higher resolution levels
            if len(features) >= 3:
                # 确保所有特征具有相同的空间尺寸
                tgt_size = features[1].shape[2:]
                resized_features = []
                for feat in features[:3]:
                    resized_feat = F.interpolate(feat, size=tgt_size, mode='nearest')
                    resized_features.append(resized_feat)
                
                # 使用zoom_cat融合
                result = self.zoom_cat(resized_features)
            elif len(features) == 2:
                # 如果只有两个特征，则直接拼接并调整通道数
                f1 = F.interpolate(features[0], size=features[1].shape[2:], mode='nearest')
                result = torch.cat([f1, features[1]], dim=1)
            elif len(features) == 1:
                # 如果只有一个特征，直接返回
                return features[0]
            else:
                raise ValueError("At least one feature tensor is required")
        else:
            # Process using scal_seq and attention for lower resolution levels
            if len(features) >= 3:
                p3, p4, p5 = features[0], features[1], features[2]
                x_input = [p3, p4, p5]
                seq_result = self.scal_seq(x_input)
                att_result = self.attention([x_input[0], seq_result])
                result = torch.cat([x_input[0], att_result], dim=1)
            elif len(features) >= 2:
                f0, f1 = features[0], features[1]
                result = torch.cat([f0, f1], dim=1)
            elif len(features) == 1:
                return features[0]
            else:
                raise ValueError("At least one feature tensor is required")

        # 使用1x1卷积调整通道数以匹配期望的输出通道数
        result = self.bn(self.reduce_conv(result))
        return result