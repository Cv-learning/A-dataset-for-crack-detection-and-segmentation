# A-dataset-for-crack-detection-and-segmentation

This code repository contains the image data and supporting code for training computer vision models, which are used in the paper *Post-Earthquake Bridge Damage Diagnosis via Multi-Domain Fusion of UAV-Based Crack Detection and Finite Element Surrogate Modeling*.

The crack detection dataset (crack dataset) includes 11,584 annotated crack images. Image files are stored in the `images` folder, and the corresponding annotation files are placed in the `labels` folder.

The crack segmentation dataset (crack segmentation dataset) contains 4,634 images for crack segmentation tasks. Original images and segmentation annotation files are stored in the `images` and `masks` folders respectively.

When training YOLO11n, YOLO12n and RT-DETR models, you can train custom models based on the ultralytics library by creating adapted yaml configuration files.

The code for dataset splitting is as follows:

```python
import os
import shutil
import random


def split_dataset(images_dir, labels_dir, output_base_dir, train_ratio=0.8, seed=42):
    # Set random seed to ensure experimental reproducibility
    random.seed(seed)

    # Get all image filenames (supports common formats like .jpg, .jpeg, .png)
    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        raise ValueError("The image folder is empty or contains no supported image formats!")

    # Randomly shuffle the list of image files
    random.shuffle(image_files)

    # Calculate the split point for training and validation sets
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]

    # Define output directories after dataset splitting
    train_image_output_dir = os.path.join(output_base_dir, "images", "train")
    val_image_output_dir = os.path.join(output_base_dir, "images", "val")
    train_label_output_dir = os.path.join(output_base_dir, "labels", "train")
    val_label_output_dir = os.path.join(output_base_dir, "labels", "val")

    # Create output directories in batches, no re-creation if they already exist
    for dir in [
        train_image_output_dir,
        val_image_output_dir,
        train_label_output_dir,
        val_label_output_dir,
    ]:
        os.makedirs(dir, exist_ok=True)

    def move_files(
        file_list, image_source_dir, label_source_dir, image_dest_dir, label_dest_dir
    ):
        for img in file_list:
            # Copy image files to the target directory
            src_img = os.path.join(image_source_dir, img)
            dst_img = os.path.join(image_dest_dir, img)
            shutil.copy(src_img, dst_img)
            # Match the corresponding label file by image filename and copy it
            label_name = os.path.splitext(img)[0] + ".txt"
            src_lbl = os.path.join(label_source_dir, label_name)
            if os.path.exists(src_lbl):  # Verify the existence of label files to avoid missing
                dst_lbl = os.path.join(label_dest_dir, label_name)
                shutil.copy(src_lbl, dst_lbl)
            else:
                print(f"⚠️ Warning: Missing corresponding label file for {img}, skipped this file.")

    # Copy image and label files for training and validation sets separately
    move_files(
        train_images,
        images_dir,
        labels_dir,
        train_image_output_dir,
        train_label_output_dir,
    )
    move_files(
        val_images, images_dir, labels_dir, val_image_output_dir, val_label_output_dir
    )

    # Print dataset splitting completion information
    print(f"✅ Dataset splitting and file migration completed:")
    print(f"   Training set images: {len(train_images)} | Storage path: {train_image_output_dir}")
    print(f"   Validation set images: {len(val_images)} | Storage path: {val_image_output_dir}")


# Example of function call
split_dataset(
    images_dir="..\\01_crack_datasets\\01_images",  # Original storage path of images before splitting
    labels_dir="..\\01_crack_datasets\\02_labels",  # Original storage path of labels before splitting
    output_base_dir="..\\01_crack_datasets",  # Root storage path of the dataset after splitting
    train_ratio=0.8,
    seed=42,
)
```

After completing the dataset splitting, you can build and train custom models with the ultralytics library using the following code, with the usage example as shown below:

```python
from ultralytics import RTDETR
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    model = RTDETR(r"ultralytics/cfg/models/rt-detr/rtdetr.yaml")
    # model.load('rtdetr-l.pt') # Load pretrained weights as needed
    model.train(
        data="01_crack.yaml",  # All training-related parameters can be reconfigured on demand
        epochs=300,
        imgsz=640,
        workers=0,
        batch=1,
        device=0,
        optimizer="AdamW",
        amp=False,
    )
```

You can also directly use the code provided in this repository for model training. In addition, you can call other codes in the repository to implement custom training for various segmentation models, such as classic models including UNet, DeepLabV3, PSPNet, etc.

The training results of some models in this repository are as follows:
