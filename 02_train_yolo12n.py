from ultralytics import YOLO
import sys
import os
from datetime import datetime


# 自定义日志重定向类：同时输出到终端 + 保存到文件
class Logger:
    def __init__(self, log_file_path):
        # 保留原始终端输出流
        self.terminal = sys.stdout
        self.stderr = sys.stderr
        # 创建日志文件（追加模式，避免覆盖）
        self.log_file = open(log_file_path, "a", encoding="utf-8")

    def write(self, message):
        # 同时写入终端和文件
        self.terminal.write(message)
        self.log_file.write(message)
        # 强制刷新，确保实时写入（避免缓存）
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        # 关闭文件流（训练结束后调用）
        if not self.log_file.closed:
            self.log_file.close()


def train_model(log_filename=None):
    """
    训练YOLO11模型，同时保存终端输出到logs文件夹
    :param log_filename: 自定义日志文件名（如"train_20250520.txt"），None则自动生成带时间戳的文件名
    """
    # ====================== 1. 初始化日志配置 ======================
    # 创建logs目录（如果不存在）
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 自定义文件名逻辑：默认生成「yolo11_train_时间戳.txt」，支持用户自定义
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"yolo11_train_{timestamp}.txt"
    log_file_path = os.path.join(log_dir, log_filename)

    # 重定向stdout/stderr到自定义Logger（捕获所有终端输出）
    logger = Logger(log_file_path)
    sys.stdout = logger
    sys.stderr = logger

    try:
        # ====================== 2. 模型训练核心逻辑 ======================
        # 加载预训练模型（核心调整：用.pt而非.yaml）
        model = YOLO("yolo12.yaml")  # yolo11n/l/m/s/x可选，n是轻量版，速度最快
        results = model.train(
            data="01_crack.yaml",  # 数据集配置文件路径（确认路径正确）
            epochs=300,  # 训练轮次（搭配早停，无需担心过拟合）
            imgsz=640,  # 输入图片尺寸（适配4090D，无需调整）
            batch=1,  # 批量大小（4090D+320尺寸，64完全适配）
            device=0,  # GPU设备（单卡无需调整）
            name="001_yolo12_origin",  # 子目录名（logs/yolo11_train/），建议加时间戳避免覆盖
            single_cls=True,
            mosaic=0.8,
            amp=False,  # 关闭混合精度
            workers=0,  # 数据加载线程
            pretrained=False,  # 不使用预训练权重
            patience=7,  # 早停
            val=True,  # 启用验证
        )
        # 打印训练关键结果（可选）
        print("\n===== 训练完成 =====")
        print(f"模型训练完成，最高验证集mAP：{results.box.map:.4f}")
        print(f"日志文件已保存至：{log_file_path}")

    except Exception as e:
        # 捕获异常并写入日志
        print(f"\n训练过程出错：{str(e)}", file=sys.stderr)
        raise e

    finally:
        # ====================== 3. 恢复原始输出流 + 关闭文件 ======================
        # 恢复stdout/stderr（避免后续输出异常）
        sys.stdout = logger.terminal
        sys.stderr = logger.stderr
        # 关闭日志文件
        logger.close()


if __name__ == "__main__":
    # -------------------------- 终端自定义控制示例 --------------------------
    # 方式1：使用默认文件名（自动生成带时间戳，如 logs/yolo11_train_20250520_143025.txt）
    # train_model()

    # 方式2：自定义文件名（如 logs/my_train_log.txt）
    train_model(log_filename="001_yolo12_origin.txt")
