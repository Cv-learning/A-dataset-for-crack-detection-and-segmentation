# config.py
import os
import torch
from typing import Dict, Any


class Config:
    # 路径配置
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = "./01_dataset"
    TEST_IMAGE_PATH = os.path.join(DATA_PATH, "03_test_image")
    TEST_LABEL_PATH = os.path.join(DATA_PATH, "04_test_label")
    PARAMS_DIR = "params"
    TRAIN_IMAGES_DIR = "train_images"
    RESULT_DIR = "result"

    # 训练参数
    BATCH_SIZE = 4  # 适中的batch size
    LEARNING_RATE = 0.0001  # 降低学习率
    NUM_EPOCHS = 3  # 增加训练轮数
    VALIDATION_INTERVAL = 1
    SAVE_BEST_ONLY = True
    EARLY_STOPPING_PATIENCE = 30  # 增加耐心值

    # 模型参数
    IMG_HEIGHT = 448
    IMG_WIDTH = 448

    # 数据增强参数
    USE_AUGMENTATION = True
    AUGMENT_PROBABILITY = 0.5

    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 损失函数参数
    DICE_WEIGHT = 0.5
    BCE_WEIGHT = 0.5

    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        dirs_to_create = [cls.PARAMS_DIR, cls.TRAIN_IMAGES_DIR, cls.RESULT_DIR]
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def to_dict(cls):
        """返回可序列化的配置字典"""
        config_dict = {}
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and not callable(getattr(cls, attr_name)):
                attr_value = getattr(cls, attr_name)
                # 确保值是基本数据类型或可序列化的类型
                if isinstance(attr_value, (str, int, float, bool, type(None))):
                    config_dict[attr_name] = attr_value
                elif isinstance(attr_value, (list, tuple)) and all(
                    isinstance(item, (str, int, float, bool, type(None)))
                    for item in attr_value
                ):
                    config_dict[attr_name] = attr_value
                else:
                    # 跳过不可序列化的对象
                    continue
        return config_dict
