import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision


class DeepLabV3Model(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super(DeepLabV3Model, self).__init__()

        # 根据是否使用预训练权重来创建模型
        if pretrained:
            # 使用预训练权重
            self.deeplabv3 = deeplabv3_resnet50(pretrained=True, progress=True)
            # 修改分类器以适应我们的任务
            classifier = list(self.deeplabv3.classifier.children())
            last_conv = nn.Conv2d(
                classifier[-1].in_channels, num_classes, kernel_size=1
            )
            self.deeplabv3.classifier = nn.Sequential(*classifier[:-1], last_conv)
        else:
            # 不使用预训练权重，从头开始构建
            # 创建骨干网络（ResNet50）
            backbone = torchvision.models.resnet50(weights=None)  # 不使用预训练权重
            backbone_children = list(backbone.children())

            # 分离各个组件
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4

            # ASPP模块 - 处理来自layer4的高级特征
            aspp_features = 256
            self.aspp = ASPP(2048, [12, 24, 36], aspp_features)

            # 低级特征处理 (来自maxpool的输出，有64个通道)
            self.low_level_conv = nn.Sequential(
                nn.Conv2d(64, 48, 1, bias=False),  # 改为64通道输入而不是256
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
            )

            # 最终分类器
            self.classifier = nn.Sequential(
                nn.Conv2d(aspp_features + 48, aspp_features, 3, padding=1, bias=False),
                nn.BatchNorm2d(aspp_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(aspp_features, aspp_features // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(aspp_features // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(aspp_features // 2, num_classes, 1),
            )

            # 初始化权重
            self._initialize_weights()

    def forward(self, x):
        input_size = x.shape[-2:]

        # 如果使用预训练模型
        if hasattr(self, "deeplabv3"):
            output = self.deeplabv3(x)
            if isinstance(output, dict):
                out = output["out"]
            else:
                out = output
        else:
            # 从头构建的模型
            # 提取低级特征（来自maxpool）
            x1 = self.conv1(x)
            x2 = self.bn1(x1)
            x3 = self.relu(x2)
            low_level_features = self.maxpool(x3)  # 这里是64通道，大小约为输入的1/4

            # 继续提取高级特征
            high_level_features = self.layer1(low_level_features)
            high_level_features = self.layer2(high_level_features)
            high_level_features = self.layer3(high_level_features)
            high_level_features = self.layer4(high_level_features)  # 2048通道

            # ASPP处理高级特征
            aspp_out = self.aspp(high_level_features)

            # 上采样ASPP输出以匹配低级特征尺寸
            aspp_out = F.interpolate(
                aspp_out,
                size=low_level_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            # 处理低级特征
            low_level_features = self.low_level_conv(low_level_features)

            # 合并高低级特征
            merged_features = torch.cat([aspp_out, low_level_features], dim=1)

            # 最终分类
            out = self.classifier(merged_features)

        # 上采样到原始输入尺寸
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        # 应用sigmoid以获得概率
        out = torch.sigmoid(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ASPP(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling) module
    """

    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()

        modules = []
        # 1x1 卷积
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # 不同扩张率的卷积
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 图像级特征
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 投影层
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # 降低dropout率
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        # 使用GroupNorm替代BatchNorm，这样在batch size为1时也能正常工作
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)  # 使用GroupNorm
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        pooled = self.avg_pool(x)
        x = self.conv(pooled)
        x = self.norm(x)
        x = self.relu(x)
        # 将结果插值回原始尺寸
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x