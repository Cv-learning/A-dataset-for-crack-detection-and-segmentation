import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SegNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(SegNet, self).__init__()
        
        # 使用VGG16的编码器部分作为预训练权重
        vgg16 = models.vgg16(weights='DEFAULT' if pretrained else None)
        
        # 编码器 (下采样路径)
        # Block 1 - 输入: 3 -> 输出: 64
        self.enc_block1 = nn.Sequential(*list(vgg16.features)[:4])  # 只取conv+relu+conv+relu，不包括池化
        # Block 2 - 输入: 64 -> 输出: 128
        self.enc_block2 = nn.Sequential(*list(vgg16.features)[5:9]) # 从第5层开始，跳过池化
        # Block 3
        self.enc_block3 = nn.Sequential(*list(vgg16.features)[10:16]) # 从第10层开始
        # Block 4
        self.enc_block4 = nn.Sequential(*list(vgg16.features)[17:23]) # 从第17层开始
        # Block 5
        self.enc_block5 = nn.Sequential(*list(vgg16.features)[24:29]) # 从第24层开始
        
        # 解码器 (上采样路径)
        # Block 5 (反向)
        self.dec_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Block 4 (反向)
        self.dec_block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Block 3 (反向)
        self.dec_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Block 2 (反向)
        self.dec_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Block 1 (反向)
        self.dec_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes)
        )
        
        # MaxUnpooling 层
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        # 编码器路径 (下采样)
        # Block 1
        x = self.enc_block1(x)
        size1 = x.size()
        x, idx1 = self.pool(x)
        
        # Block 2
        x = self.enc_block2(x)
        size2 = x.size()
        x, idx2 = self.pool(x)
        
        # Block 3
        x = self.enc_block3(x)
        size3 = x.size()
        x, idx3 = self.pool(x)
        
        # Block 4
        x = self.enc_block4(x)
        size4 = x.size()
        x, idx4 = self.pool(x)
        
        # Block 5
        x = self.enc_block5(x)
        size5 = x.size()
        x, idx5 = self.pool(x)
        
        # 解码器路径 (上采样)
        # Block 5 (反向)
        x = self.unpool(x, idx5, output_size=size5)
        x = self.dec_block5(x)
        
        # Block 4 (反向)
        x = self.unpool(x, idx4, output_size=size4)
        x = self.dec_block4(x)
        
        # Block 3 (反向)
        x = self.unpool(x, idx3, output_size=size3)
        x = self.dec_block3(x)
        
        # Block 2 (反向)
        x = self.unpool(x, idx2, output_size=size2)
        x = self.dec_block2(x)
        
        # Block 1 (反向)
        x = self.unpool(x, idx1, output_size=size1)
        x = self.dec_block1(x)
        
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


# 替代的SegNet实现，使用标准卷积代替VGG预训练权重
class SegNetSimple(nn.Module):
    def __init__(self, num_classes=1):
        super(SegNetSimple, self).__init__()
        
        # 编码器 (下采样路径)
        # Block 1
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Block 2
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Block 3
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Block 4
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Block 5
        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # 解码器 (上采样路径)
        # Block 5
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Block 4
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Block 3
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Block 2
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Block 1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        # 编码器路径 (下采样)
        # Block 1
        x = self.enc_conv1(x)
        size1 = x.size()
        x, idx1 = self.pool1(x)
        
        # Block 2
        x = self.enc_conv2(x)
        size2 = x.size()
        x, idx2 = self.pool2(x)
        
        # Block 3
        x = self.enc_conv3(x)
        size3 = x.size()
        x, idx3 = self.pool3(x)
        
        # Block 4
        x = self.enc_conv4(x)
        size4 = x.size()
        x, idx4 = self.pool4(x)
        
        # Block 5
        x = self.enc_conv5(x)
        size5 = x.size()
        x, idx5 = self.pool5(x)
        
        # 解码器路径 (上采样)
        # Block 5
        x = self.unpool5(x, idx5, output_size=size5)
        x = self.dec_conv5(x)
        
        # Block 4
        x = self.unpool4(x, idx4, output_size=size4)
        x = self.dec_conv4(x)
        
        # Block 3
        x = self.unpool3(x, idx3, output_size=size3)
        x = self.dec_conv3(x)
        
        # Block 2
        x = self.unpool2(x, idx2, output_size=size2)
        x = self.dec_conv2(x)
        
        # Block 1
        x = self.unpool1(x, idx1, output_size=size1)
        x = self.dec_conv1(x)
        
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


if __name__ == "__main__":
    # 测试SegNet模型
    model = SegNetSimple(num_classes=1)  # 使用简化版本以避免预训练权重问题
    x = torch.randn(1, 3, 448, 448)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")