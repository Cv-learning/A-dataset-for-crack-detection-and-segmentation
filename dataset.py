# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import logging
from config import Config

logger = logging.getLogger(__name__)


class CrackDataset(Dataset):
    """裂缝检测数据集"""

    def __init__(self, image_dir, label_dir, transform=None, augment_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.augment_transform = augment_transform

        # 获取所有图像文件
        self.images = []
        self.labels = []

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            logger.error(f"Directory not found: {image_dir} or {label_dir}")
            return

        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, img_file)  # 假设标签文件名与图像相同

            if os.path.exists(label_path):
                self.images.append(img_path)
                self.labels.append(label_path)
            else:
                # 尝试不同的扩展名匹配
                for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                    alt_label_path = os.path.join(
                        label_dir, img_file.rsplit(".", 1)[0] + ext
                    )
                    if os.path.exists(alt_label_path):
                        self.images.append(img_path)
                        self.labels.append(alt_label_path)
                        break
                else:
                    logger.warning(f"No matching label found for {img_file}")

        logger.info(f"Dataset initialized with {len(self.images)} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            # 加载图像和标签
            image = Image.open(self.images[idx]).convert("RGB")
            label = Image.open(self.labels[idx]).convert("L")  # 转为灰度图

            # 如果标签是多通道，取第一个通道
            if len(np.array(label).shape) > 2:
                label = label.convert("L")

            # 转换为二值化标签
            label_array = np.array(label)
            label_array = (label_array > 127).astype(np.float32)  # 二值化阈值
            label = Image.fromarray(label_array * 255.0)

            # 应用变换
            if self.augment_transform and np.random.rand() < Config.AUGMENT_PROBABILITY:
                # 对图像和标签应用相同的随机变换
                # 随机水平翻转
                if np.random.rand() < 0.5:
                    image = transforms.functional.hflip(image)
                    label = transforms.functional.hflip(label)

                # 随机垂直翻转
                if np.random.rand() < 0.3:
                    image = transforms.functional.vflip(image)
                    label = transforms.functional.vflip(label)

                # 随机旋转（注意：这可能会影响图像质量，谨慎使用）
                if np.random.rand() < 0.2:
                    angle = np.random.uniform(-15, 15)
                    image = transforms.functional.rotate(image, angle)
                    label = transforms.functional.rotate(label, angle)

                # 颜色抖动只应用于图像
                if np.random.rand() < 0.2:
                    color_jitter = transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    )
                    image = color_jitter(image)

                # 应用基础变换
                if self.transform:
                    image = self.transform(image)
                    label = self.transform(label)  # 对标签也应用基础变换
            else:
                # 只应用基础变换
                if self.transform:
                    image = self.transform(image)
                    label = self.transform(label)

            # 确保标签是0-1范围
            label = torch.clamp(label, 0, 1)  # 限制在0-1范围内

            return image, label
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # 返回默认张量
            return torch.zeros(3, Config.IMG_HEIGHT, Config.IMG_WIDTH), torch.zeros(
                1, Config.IMG_HEIGHT, Config.IMG_WIDTH
            )


def get_transforms():
    """获取数据预处理变换"""
    # 基础变换
    base_transform = transforms.Compose(
        [
            transforms.Resize((Config.IMG_HEIGHT, Config.IMG_WIDTH)),
            transforms.ToTensor(),
        ]
    )

    # 数据增强变换
    augment_transform = transforms.Compose(
        [
            transforms.Resize((Config.IMG_HEIGHT, Config.IMG_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]
    )

    return base_transform, augment_transform
