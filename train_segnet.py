import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime
import json
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import Config
from dataset import CrackDataset, get_transforms
from segnet_model import SegNetSimple  # 使用简化版本避免预训练权重问题
from utils import save_sample_images, save_validation_results

# 设置日志 - 使用固定文件名
log_filename = "segnet_training.log"  # SegNet专用日志文件名
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DiceBCELoss(nn.Module):
    """结合Dice损失和BCE损失的复合损失函数"""

    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(DiceBCELoss, self).__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        # BCE部分
        bce = F.binary_cross_entropy(inputs, targets)

        # Dice部分
        smooth = 1.0
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (
            inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth
        )
        dice_loss = 1 - dice.mean()

        return self.weight_bce * bce + self.weight_dice * dice_loss


class SegNetTrainer:
    def __init__(self):
        Config.ensure_directories()

        # 初始化设备
        self.device = torch.device(
            Config.DEVICE if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # 初始化模型 - 使用SegNetSimple
        self.model = SegNetSimple(num_classes=1).to(self.device)

        # 初始化损失函数和优化器
        self.criterion = DiceBCELoss(
            weight_bce=Config.BCE_WEIGHT, weight_dice=Config.DICE_WEIGHT
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=1e-5,  # 减小权重衰减
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
        )

        # 早停机制
        self.best_loss = float("inf")
        self.patience_counter = 0

        # TensorBoard日志
        self.writer = SummaryWriter(
            f'runs/segnet_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

        # 指标记录
        self.train_metrics = {"epoch": [], "loss": [], "iou": [], "lr": [], "dice": []}
        self.val_metrics = {"epoch": [], "loss": [], "iou": [], "dice": []}

        # CSV文件路径
        self.csv_path = os.path.join(
            Config.TRAIN_IMAGES_DIR, "segnet_training_metrics.csv"
        )

    def log_gpu_memory(self):
        """记录GPU显存使用情况"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            logger.info(
                f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
            )

            # 将显存信息写入TensorBoard
            if hasattr(self, "writer"):
                self.writer.add_scalar(
                    "Memory/Allocated_GB",
                    memory_allocated,
                    getattr(self, "_current_step", 0),
                )
                self.writer.add_scalar(
                    "Memory/Reserved_GB",
                    memory_reserved,
                    getattr(self, "_current_step", 0),
                )
        else:
            logger.info("CUDA not available, skipping GPU memory logging")

    def load_data(self):
        """加载训练和验证数据"""
        train_transform, train_augment = get_transforms()
        val_transform, _ = get_transforms()

        # 训练集
        train_dataset = CrackDataset(
            image_dir=os.path.join(Config.DATA_PATH, "01_rgb_image"),
            label_dir=os.path.join(Config.DATA_PATH, "02_mask_label"),
            transform=train_transform,
            augment_transform=train_augment,
        )

        # 验证集（如果存在测试数据则用作验证）
        val_image_dir = Config.TEST_IMAGE_PATH
        val_label_dir = Config.TEST_LABEL_PATH
        if os.path.exists(val_image_dir) and os.path.exists(val_label_dir):
            val_dataset = CrackDataset(
                image_dir=val_image_dir,
                label_dir=val_label_dir,
                transform=val_transform,
            )
        else:
            logger.warning(
                "Validation/test data not found, using training data for validation"
            )
            val_dataset = train_dataset  # 临时方案

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        logger.info(
            f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
        )
        logger.info(
            f"Training batches: {len(self.train_loader)}, Validation batches: {len(self.val_loader)}"
        )

    def calculate_iou(self, pred, target, threshold=0.5):
        """计算IoU指标"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > 0.5).float()

        intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
        union = (pred_binary + target_binary - pred_binary * target_binary).sum(
            dim=(1, 2, 3)
        )

        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()

    def calculate_dice_coefficient(self, pred, target, threshold=0.5):
        """计算Dice系数"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > 0.5).float()

        intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
        denominator = pred_binary.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3))

        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return dice.mean()

    def calculate_class_balance(self, labels):
        """计算类别平衡"""
        pos_pixels = (labels > 0.5).float().sum()
        total_pixels = labels.numel()
        return pos_pixels / total_pixels

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_pos_ratio = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{Config.NUM_EPOCHS} [Train]",
            leave=False,
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):
            # 记录当前步骤用于显存监控
            self._current_step = epoch * len(self.train_loader) + batch_idx

            images, labels = images.to(self.device), labels.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 计算IoU和Dice
            iou = self.calculate_iou(outputs, labels)
            dice = self.calculate_dice_coefficient(outputs, labels)

            # 计算类别平衡
            pos_ratio = self.calculate_class_balance(labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_iou += iou.item()
            total_dice += dice.item()
            total_pos_ratio += pos_ratio.item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.6f}",
                    "IoU": f"{iou.item():.4f}",
                    "Dice": f"{dice.item():.4f}",
                    "LR": f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                }
            )

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        avg_dice = total_dice / num_batches
        avg_pos_ratio = total_pos_ratio / num_batches
        current_lr = self.optimizer.param_groups[0]["lr"]

        # 记录指标
        self.train_metrics["epoch"].append(epoch)
        self.train_metrics["loss"].append(avg_loss)
        self.train_metrics["iou"].append(avg_iou)
        self.train_metrics["dice"].append(avg_dice)
        self.train_metrics["lr"].append(current_lr)

        # 写入TensorBoard
        self.writer.add_scalar("Train/Loss", avg_loss, epoch)
        self.writer.add_scalar("Train/IoU", avg_iou, epoch)
        self.writer.add_scalar("Train/Dice", avg_dice, epoch)
        self.writer.add_scalar("Train/LR", current_lr, epoch)
        self.writer.add_scalar("Train/PosPixelRatio", avg_pos_ratio, epoch)

        return avg_loss, avg_iou, avg_dice

    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_pos_ratio = 0.0
        num_batches = 0

        # 用于保存验证结果的列表
        validation_images = []
        validation_labels = []
        validation_outputs = []

        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch}/{Config.NUM_EPOCHS} [Val]",
                leave=False,
            )

            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                iou = self.calculate_iou(outputs, labels)
                dice = self.calculate_dice_coefficient(outputs, labels)

                # 计算类别平衡
                pos_ratio = self.calculate_class_balance(labels)

                total_loss += loss.item()
                total_iou += iou.item()
                total_dice += dice.item()
                total_pos_ratio += pos_ratio.item()
                num_batches += 1

                # 保存前几个批次的图像用于可视化（最多保存1个批次）
                if len(validation_images) < Config.BATCH_SIZE:
                    validation_images.append(images.cpu())
                    validation_labels.append(labels.cpu())
                    validation_outputs.append(outputs.cpu())

                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.6f}",
                        "IoU": f"{iou.item():.4f}",
                        "Dice": f"{dice.item():.4f}",
                    }
                )

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        avg_dice = total_dice / num_batches
        avg_pos_ratio = total_pos_ratio / num_batches

        # 记录指标
        self.val_metrics["epoch"].append(epoch)
        self.val_metrics["loss"].append(avg_loss)
        self.val_metrics["iou"].append(avg_iou)
        self.val_metrics["dice"].append(avg_dice)

        # 写入TensorBoard
        self.writer.add_scalar("Validation/Loss", avg_loss, epoch)
        self.writer.add_scalar("Validation/IoU", avg_iou, epoch)
        self.writer.add_scalar("Validation/Dice", avg_dice, epoch)
        self.writer.add_scalar("Validation/PosPixelRatio", avg_pos_ratio, epoch)

        # 保存验证结果图片
        if validation_images:
            val_images = torch.cat(validation_images, dim=0)[:4]  # 最多保存4张
            val_labels = torch.cat(validation_labels, dim=0)[:4]
            val_outputs = torch.cat(validation_outputs, dim=0)[:4]

            val_result_path = os.path.join(
                Config.TRAIN_IMAGES_DIR, f"segnet_validation_epoch_{epoch}.png"
            )
            save_validation_results(
                val_images, val_labels, val_outputs, val_result_path
            )
            logger.info(f"Validation results saved to {val_result_path}")

        # 更新学习率调度器
        self.scheduler.step()

        return avg_loss, avg_iou, avg_dice

    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": Config.to_dict(),  # 使用to_dict()方法而不是直接使用__dict__
        }

        # 保存当前模型
        current_path = os.path.join(Config.PARAMS_DIR, "segnet_current.pth")
        torch.save(checkpoint, current_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(Config.PARAMS_DIR, "segnet_best.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"Best SegNet model saved with loss: {loss:.6f}")

    def early_stopping_check(self, val_loss):
        """检查是否需要早停"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return True  # 需要保存最佳模型
        else:
            self.patience_counter += 1
            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping triggered after {Config.EARLY_STOPPING_PATIENCE} epochs without improvement"
                )
                return False
            return False

    def save_metrics_to_csv(self):
        """保存指标到CSV文件"""
        # 确定最大的epoch数
        max_epochs = max(
            len(self.train_metrics["epoch"]), len(self.val_metrics["epoch"])
        )

        # 创建数据字典
        data = {
            "epoch": list(range(1, max_epochs + 1)),
            "train_loss": [np.nan] * max_epochs,
            "train_iou": [np.nan] * max_epochs,
            "train_dice": [np.nan] * max_epochs,
            "train_lr": [np.nan] * max_epochs,
            "val_loss": [np.nan] * max_epochs,
            "val_iou": [np.nan] * max_epochs,
            "val_dice": [np.nan] * max_epochs,
        }

        # 填充训练指标
        for i, epoch in enumerate(self.train_metrics["epoch"]):
            if i < len(self.train_metrics["loss"]):
                data["train_loss"][i] = self.train_metrics["loss"][i]
                data["train_iou"][i] = self.train_metrics["iou"][i]
                data["train_dice"][i] = self.train_metrics["dice"][i]
                data["train_lr"][i] = self.train_metrics["lr"][i]

        # 填充验证指标
        for i, epoch in enumerate(self.val_metrics["epoch"]):
            if i < len(self.val_metrics["loss"]):
                data["val_loss"][i] = self.val_metrics["loss"][i]
                data["val_iou"][i] = self.val_metrics["iou"][i]
                data["val_dice"][i] = self.val_metrics["dice"][i]

        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)
        logger.info(f"SegNet training metrics saved to {self.csv_path}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.train_metrics["epoch"]:
            logger.warning("No training metrics to plot")
            return

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("SegNet Training Curves", fontsize=16)

        # 损失曲线
        if len(self.train_metrics["epoch"]) > 0:
            axes[0, 0].plot(
                self.train_metrics["epoch"],
                self.train_metrics["loss"],
                label="Train Loss",
                color="blue",
                marker="o",
                markersize=3,
            )
        if len(self.val_metrics["epoch"]) > 0:
            axes[0, 0].plot(
                self.val_metrics["epoch"],
                self.val_metrics["loss"],
                label="Validation Loss",
                color="red",
                marker="s",
                markersize=3,
            )
        axes[0, 0].set_title("Loss Curve")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # IoU曲线
        if len(self.train_metrics["epoch"]) > 0:
            axes[0, 1].plot(
                self.train_metrics["epoch"],
                self.train_metrics["iou"],
                label="Train IoU",
                color="blue",
                marker="o",
                markersize=3,
            )
        if len(self.val_metrics["epoch"]) > 0:
            axes[0, 1].plot(
                self.val_metrics["epoch"],
                self.val_metrics["iou"],
                label="Validation IoU",
                color="red",
                marker="s",
                markersize=3,
            )
        axes[0, 1].set_title("IoU Curve")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("IoU")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Dice曲线
        if len(self.train_metrics["epoch"]) > 0:
            axes[1, 0].plot(
                self.train_metrics["epoch"],
                self.train_metrics["dice"],
                label="Train Dice",
                color="blue",
                marker="o",
                markersize=3,
            )
        if len(self.val_metrics["epoch"]) > 0:
            axes[1, 0].plot(
                self.val_metrics["epoch"],
                self.val_metrics["dice"],
                label="Validation Dice",
                color="red",
                marker="s",
                markersize=3,
            )
        axes[1, 0].set_title("Dice Coefficient Curve")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Dice Coefficient")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 综合指标对比
        ax4 = axes[1, 1]
        if len(self.val_metrics["epoch"]) > 0:
            ax4.plot(
                self.train_metrics["epoch"],
                self.train_metrics["iou"],
                label="Train IoU",
                color="blue",
                marker="o",
                markersize=3,
            )
            ax4.plot(
                self.val_metrics["epoch"],
                self.val_metrics["iou"],
                label="Validation IoU",
                color="red",
                marker="s",
                markersize=3,
            )
            ax4_twin = ax4.twinx()
            ax4_twin.plot(
                self.train_metrics["epoch"],
                self.train_metrics["loss"],
                label="Train Loss",
                color="blue",
                linestyle="--",
                marker="o",
                markersize=3,
            )
            ax4_twin.plot(
                self.val_metrics["epoch"],
                self.val_metrics["loss"],
                label="Validation Loss",
                color="red",
                linestyle="--",
                marker="s",
                markersize=3,
            )
            ax4.set_ylabel("IoU", color="black")
            ax4_twin.set_ylabel("Loss", color="black")
            ax4.set_xlabel("Epoch")
            ax4.legend(loc="upper left")
            ax4_twin.legend(loc="upper right")
        else:
            ax4.plot(
                self.train_metrics["epoch"],
                self.train_metrics["iou"],
                label="Train IoU",
                color="blue",
                marker="o",
                markersize=3,
            )
            ax4.plot(
                self.train_metrics["epoch"],
                self.train_metrics["loss"],
                label="Train Loss",
                color="red",
                linestyle="--",
                marker="s",
                markersize=3,
            )
            ax4.set_title("Train IoU and Loss")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Value")
            ax4.legend()
        ax4.grid(True)

        # 保存图像
        plot_path = os.path.join(
            Config.TRAIN_IMAGES_DIR, "segnet_training_curves.png"
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"SegNet training curves saved to {plot_path}")

    def train(self):
        """主训练循环"""
        logger.info("=" * 80)
        logger.info("Starting SegNet Training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch Size: {Config.BATCH_SIZE}")
        logger.info(f"Learning Rate: {Config.LEARNING_RATE}")
        logger.info(f"Number of Epochs: {Config.NUM_EPOCHS}")
        logger.info(f"Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}")
        logger.info("=" * 80)

        # 记录初始显存使用情况
        logger.info("Initial GPU Memory Status:")
        self.log_gpu_memory()

        # 加载数据
        self.load_data()

        # 记录数据加载后的显存使用情况
        logger.info("GPU Memory Status after data loading:")
        self.log_gpu_memory()

        # 训练循环
        start_epoch = 0

        # 尝试加载检查点
        checkpoint_path = os.path.join(Config.PARAMS_DIR, "segnet_current.pth")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint["epoch"]
                self.best_loss = checkpoint["loss"]
                logger.info(
                    f"Resumed training from epoch {start_epoch}, best loss: {self.best_loss:.6f}"
                )
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

        for epoch in range(start_epoch + 1, Config.NUM_EPOCHS + 1):
            # 训练
            train_loss, train_iou, train_dice = self.train_epoch(epoch)

            # 验证
            val_loss, val_iou, val_dice = self.validate(epoch)

            # 保存样本图像
            if epoch % 10 == 0:
                self.save_sample_predictions(epoch)

            # 输出训练信息
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch:3d}/{Config.NUM_EPOCHS}] "
                f"Train Loss: {train_loss:.6f} | Train IoU: {train_iou:.4f} | Train Dice: {train_dice:.4f} | "
                f"Val Loss: {val_loss:.6f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | "
                f"LR: {lr:.6f}"
            )

            # 每隔一定epoch记录显存使用情况
            if epoch % 10 == 0:
                logger.info(f"GPU Memory Status at epoch {epoch}:")
                self.log_gpu_memory()

            # 检查是否需要保存最佳模型
            should_save_best = self.early_stopping_check(val_loss)

            if should_save_best:
                self.save_checkpoint(epoch, val_loss, is_best=True)

            # 保存当前模型
            self.save_checkpoint(epoch, val_loss)

            # 检查是否早停
            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info("Training stopped due to early stopping")
                break

        # 训练完成
        logger.info("=" * 80)
        logger.info("SegNet training completed!")
        logger.info(f"Best validation loss: {self.best_loss:.6f}")
        logger.info(f"Total epochs trained: {epoch}")

        # 记录最终显存使用情况
        logger.info("Final GPU Memory Status:")
        self.log_gpu_memory()
        logger.info("=" * 80)

        # 保存指标到CSV
        self.save_metrics_to_csv()

        # 绘制训练曲线
        self.plot_training_curves()

        # 关闭TensorBoard writer
        self.writer.close()

        # 保存训练历史
        self.save_training_history()

    def save_sample_predictions(self, epoch):
        """保存样本预测结果"""
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.val_loader):
                if i >= 1:  # 只保存第一批的一个样本
                    break

                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                # 保存样本图像
                sample_path = os.path.join(
                    Config.TRAIN_IMAGES_DIR, f"segnet_sample_epoch_{epoch}.png"
                )
                save_sample_images(
                    images.cpu(), labels.cpu(), outputs.cpu(), sample_path
                )
                break

    def save_training_history(self):
        """保存训练历史"""
        history = {
            "train_loss": self.train_metrics["loss"],
            "train_iou": self.train_metrics["iou"],
            "train_dice": self.train_metrics["dice"],
            "val_loss": self.val_metrics["loss"],
            "val_iou": self.val_metrics["iou"],
            "val_dice": self.val_metrics["dice"],
            "best_val_loss": self.best_loss,
            "config": Config.to_dict(),  # 使用to_dict()方法
        }

        history_path = os.path.join(
            Config.TRAIN_IMAGES_DIR, "segnet_training_history.json"
        )
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"SegNet training history saved to {history_path}")


def main():
    trainer = SegNetTrainer()
    trainer.train()


if __name__ == "__main__":
    main()