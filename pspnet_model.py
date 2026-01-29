import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PSPNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(PSPNet, self).__init__()
        
        # 使用ResNet34作为骨干网络
        resnet = models.resnet34(weights='DEFAULT' if pretrained else None)
        
        # 提取骨干网络的组件
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        # PSP模块
        self.psp_layer = PSPModule(in_features=512, out_features=512, sizes=(1, 2, 3, 6))
        
        # 辅助分支 - 用于中间监督
        self.aux_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        # 主分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),  # 512*3 + 512 from psp = 1024
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        
        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 编码器前部
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        
        x = self.encoder1(x)
        x = self.encoder2(x)
        x_aux = x  # 辅助分支输入
        
        x = self.encoder3(x)
        x = self.encoder4(x)
        
        # PSP模块
        psp_out = self.psp_layer(x)
        
        # 将PSP输出与原始特征连接
        x = torch.cat([x, psp_out], dim=1)
        
        # 主分类头
        x = self.classifier(x)
        
        # 上采样到输入尺寸
        x = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)
        
        # 辅助分支
        aux_out = self.aux_branch(x_aux)
        aux_out = F.interpolate(aux_out, size=(448, 448), mode='bilinear', align_corners=False)
        
        # 使用sigmoid激活函数得到分割概率
        x = torch.sigmoid(x)
        aux_out = torch.sigmoid(aux_out)
        
        return x, aux_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class PSPModule(nn.Module):
    """PSP模块实现"""
    def __init__(self, in_features, out_features, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        
        # 每个金字塔级别的输出通道数
        mid_features = in_features // len(sizes)
        
        self.stages = nn.ModuleList([
            self._make_stage(in_features, mid_features, size) 
            for size in sizes
        ])
        
        # 最终投影层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_features + mid_features * len(sizes), out_features, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, in_features, out_features, size):
        """创建金字塔的某一级"""
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        relu = nn.ReLU(inplace=True)
        
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), 
                      mode='bilinear', align_corners=False) for stage in self.stages]
        
        bottle = self.bottleneck(torch.cat([feats] + priors, 1))
        
        return bottle


# 简化版PSPNet，避免预训练权重问题
class PSPNetSimple(nn.Module):
    def __init__(self, num_classes=1):
        super(PSPNetSimple, self).__init__()
        
        # 编码器部分 - 自定义卷积块
        self.enc_conv1 = self._make_encoder_block(3, 64, 2)  # 448->224
        self.enc_conv2 = self._make_encoder_block(64, 128, 2)  # 224->112
        self.enc_conv3 = self._make_encoder_block(128, 256, 2)  # 112->56
        self.enc_conv4 = self._make_encoder_block(256, 512, 2)  # 56->28
        
        # PSP模块 - 修复尺寸问题，避免出现1x1的情况
        self.psp_layer = PSPModuleSimple(in_features=512, out_features=256, sizes=(1, 2, 3, 6))
        
        # 主分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 256, 512, kernel_size=3, padding=1, bias=False),  # 512 from encoder + 256 from PSP
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.Upsample(size=(448, 448), mode='bilinear', align_corners=False)
        )
        
        # 初始化权重
        self._initialize_weights()

    def _make_encoder_block(self, in_channels, out_channels, pool_size=2):
        """创建编码器块"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool_size > 1:
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 编码器
        x1 = self.enc_conv1(x)  # 224
        x2 = self.enc_conv2(x1)  # 112
        x3 = self.enc_conv3(x2)  # 56
        x4 = self.enc_conv4(x3)  # 28
        
        # PSP模块
        psp_out = self.psp_layer(x4)
        
        # 将PSP输出与原始特征连接
        x = torch.cat([x4, psp_out], dim=1)
        
        # 主分类头
        x = self.classifier(x)
        
        # 使用sigmoid激活函数得到分割概率
        x = torch.sigmoid(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class PSPModuleSimple(nn.Module):
    """简化版PSP模块实现，修复小特征图问题"""
    def __init__(self, in_features, out_features, sizes=(1, 2, 3, 6)):
        super(PSPModuleSimple, self).__init__()
        
        # 每个金字塔级别的输出通道数
        mid_features = in_features // len(sizes)
        
        self.stages = nn.ModuleList([
            self._make_stage(in_features, mid_features, size) 
            for size in sizes
        ])
        
        # 最终投影层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_features + mid_features * len(sizes), out_features, 
                     kernel_size=1, bias=False),  # 使用1x1卷积减少参数
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, in_features, out_features, size):
        """创建金字塔的某一级"""
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        # 使用GroupNorm替代BatchNorm，避免小特征图的BN问题
        gn = nn.GroupNorm(min(32, out_features), out_features)
        relu = nn.ReLU(inplace=True)
        
        return nn.Sequential(prior, conv, gn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # 为每个stage单独处理，避免批量归一化错误
        priors = []
        for stage in self.stages:
            # 处理每个金字塔级别
            stage_feat = stage(feats)
            # 插值到原图大小
            stage_upsampled = F.interpolate(input=stage_feat, size=(h, w), 
                                          mode='bilinear', align_corners=False)
            priors.append(stage_upsampled)
        
        # 拼接原始特征和金字塔特征
        output = self.bottleneck(torch.cat([feats] + priors, 1))
        
        return output


if __name__ == "__main__":
    # 测试PSPNet模型
    model = PSPNetSimple(num_classes=1)
    x = torch.randn(1, 3, 448, 448)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")