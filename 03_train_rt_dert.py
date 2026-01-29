from ultralytics import RTDETR
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    model = RTDETR(r"ultralytics/cfg/models/rt-detr/rtdetr.yaml")
    # model.load('rtdetr-l.pt') # 是否加载预训练权重
    model.train(
        data="01_crack.yaml",  # 训练参数均可以重新设置
        epochs=300,
        imgsz=640,
        workers=0,
        batch=1,
        device=0,
        optimizer="AdamW",
        amp=False,
    )
