# utils.py
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import os

def calculate_metrics(pred, target, threshold=0.5):
    """
    计算分割指标
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # 计算TP, TN, FP, FN
    tp = (pred_flat * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    # 计算指标
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    accuracy = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1_score = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item()
    }

def save_sample_images(images, labels, predictions, save_path):
    """
    保存样本图像：原图 | 标签 | 预测结果
    """
    batch_size = min(4, images.size(0))  # 最多显示4张
    
    # 准备图像网格
    sample_images = []
    
    for i in range(batch_size):
        img = images[i]
        # 如果标签是单通道，复制到3通道以便可视化
        lbl = labels[i].repeat(3, 1, 1) if labels[i].size(0) == 1 else labels[i][:3]
        # 如果预测是单通道，复制到3通道以便可视化
        pred = predictions[i].repeat(3, 1, 1) if predictions[i].size(0) == 1 else predictions[i][:3]
        
        sample_images.extend([img, lbl, pred])
    
    # 保存图像
    grid_image = torch.stack(sample_images, dim=0)
    save_image(grid_image, save_path, nrow=3, normalize=True, padding=2)

def save_validation_results(images, labels, predictions, save_path):
    """
    保存验证结果图像：原图 | 标签 | 预测结果
    """
    batch_size = min(4, images.size(0))  # 最多显示4张
    
    # 准备图像网格
    val_images = []
    
    for i in range(batch_size):
        img = images[i]
        # 如果标签是单通道，复制到3通道以便可视化
        lbl = labels[i].repeat(3, 1, 1) if labels[i].size(0) == 1 else labels[i][:3]
        # 如果预测是单通道，复制到3通道以便可视化
        pred = predictions[i].repeat(3, 1, 1) if predictions[i].size(0) == 1 else predictions[i][:3]
        
        val_images.extend([img, lbl, pred])
    
    # 保存图像
    grid_image = torch.stack(val_images, dim=0)
    save_image(grid_image, save_path, nrow=3, normalize=True, padding=2)

def keep_image_size_open(path):
    """
    保持图像尺寸打开
    """
    return Image.open(path).convert('RGB')

def create_confusion_matrix(pred, target, threshold=0.5):
    """
    创建混淆矩阵
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()
    
    # 计算混淆矩阵元素
    tp = (pred_binary * target_binary).sum()
    tn = ((1 - pred_binary) * (1 - target_binary)).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()
    
    return tp.item(), tn.item(), fp.item(), fn.item()