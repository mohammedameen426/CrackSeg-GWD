
# CrackSeg-GWD: Automated Lightweight Networks for Multi-Material Bridge Crack Segmentation

> A lightweight encoder–decoder model integrating Group Normalization, Weight-Standardized Convolutions, DropBlock regularization, and Symmetric Unified Focal Loss for crack segmentation across concrete, steel, and asphalt surfaces.

[![Paper](https://img.shields.io/badge/Paper-Automation%20in%20Construction-blue)](https://doi.org/10.1016/j.autcon.2026.106808)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Patent](https://img.shields.io/badge/Patent-CN%20发明专利-red)](#license)

---

## Overview

Crack segmentation on concrete, steel, and asphalt surfaces is a challenging task due to irregular crack patterns, low contrast, and noise interference, particularly in complex environments. Existing deep learning methods often struggle to balance fine-grained feature extraction with contextual understanding, and no unified model effectively detects cracks across all three materials while remaining lightweight enough for edge deployment.

**CrackSeg-GWD** (Crack Segmentation with Group Normalization, Weight-Standardized Convolutions, and DropBlock) is a lightweight encoder–decoder model designed to overcome these limitations. With only **0.414 million parameters** and **0.849 GFLOPs**, it achieves high segmentation accuracy across five benchmark datasets: SteelCrack, YCD, Crack500, DeepCrack, and Ozgenel.

On average, CrackSeg-GWD surpasses ten state-of-the-art models by **6.7% in mIoU**, **10.3% in Dice Score**, **20.1% in Recall**, and **9.9% in AUC**, while producing cleaner and more accurate crack segmentation maps. Its low computational cost makes it suitable for real-time structural health monitoring on edge devices such as drones and smartphones.

---

## Model Architecture

![CrackSeg-GWD Architecture](figures/architecture.png)

The architecture follows a U-Net-inspired encoder–decoder structure with four key innovations:

1. **Group Normalization (GN)** — Provides stable training independent of batch size, preserving fine-grained crack structures across diverse surface textures.
2. **Weight-Standardized Convolutions (WS)** — Normalizes convolutional filter weights for smoother optimization and more stable gradient flow, especially when working with high-resolution images.
3. **DropBlock Regularization** — Removes contiguous feature regions (block size 7, keep probability 0.9) to prevent overfitting and force the network to learn spatially distributed features.
4. **Symmetric Unified Focal (SUF) Loss** — Combines a region-level Focal Tversky Loss with pixel-level Mean Absolute Error to address severe class imbalance (cracks occupy < 2% of image pixels).

The encoder comprises three progressively deeper residual blocks (32 → 64 → 128 filters), followed by a three-stage decoder using `Conv2DTranspose` for learnable upsampling. Dense Conditional Random Fields (DenseCRFs) are applied as a post-processing step for sharper, more continuous crack maps.

---

## Datasets

The following publicly available datasets were used to train and evaluate CrackSeg-GWD:

**[SteelCrack](https://doi.org/10.1016/j.autcon.2024.105354)**: 4355 images (3300 train / 530 test / 525 val), resized to 512×512. Collected from the Nanjing Second Yangtze River Bridge and Humen Bridge; includes diverse steel crack types under varying environmental conditions.

**[YCD](https://doi.org/10.1111/mice.12412)**: 776 images (622 train / 154 test), 512×512. Contains road cracks and concrete wall cracks captured at varying camera distances and scales.

**[Crack500](https://doi.org/10.1109/TITS.2019.2910595)**: 3368 images (2244 train / 1124 test), 360×640. Challenging pavement crack dataset with inconsistent widths, occlusions, and visual clutter.

**[DeepCrack](https://doi.org/10.1016/j.neucom.2019.01.036)**: 530 images (300 train / 230 test), 544×384. Pixel-level annotated dataset covering diverse crack types under shadows, moisture, and variable lighting.

**[Ozgenel](https://doi.org/10.17632/jwsn7tfbrp.1)**: 458 high-resolution images (366 train / 92 test), 4032×3024. Concrete crack images collected from buildings at Middle East Technical University.

**QRCD (Field-collected)**: 128 test-only images, 3024×3024. High-resolution crack images collected from Qingyuan Road under real-world inspection conditions (not used in training).

---

## Requirements

- Python 3.10+
- TensorFlow 2.19.0 (Keras backend)
- CUDA 12.4 / NVIDIA Driver 550+

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `tensorflow>=2.19`, `numpy`, `opencv-python`, `scikit-learn`, `matplotlib`, `pydensecrf`

---

## Installation

```bash
git clone https://github.com/mohammedameen426/CrackSeg-GWD.git
cd CrackSeg-GWD
pip install -r requirements.txt
```

---

## Usage

### Dataset Preparation

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

All images should be resized to **256×256** pixels before training.

### Training

```bash
python train.py --data ./data --epochs 100 --batch-size 24 --lr 1e-4
```

Training arguments:

```
--data        Path to dataset root directory
--epochs      Number of training epochs (default: 100)
--batch-size  Batch size (default: 24)
--image-size  Input image size (default: 256)
--lr          Learning rate for Adam optimizer (default: 1e-4)
--save-dir    Directory to save model checkpoints
```

### Evaluation

```bash
python evaluate.py --weights weights/crackseg_gwd_best.h5 --data ./data/test
```

### Inference

```bash
python inference.py --weights weights/crackseg_gwd_best.h5 --input path/to/image.jpg --output results/
```

---

## Results

### Performance on Individual Datasets

| Dataset | Accuracy | Recall | Dice Score | mIoU | AUC |
|---------|----------|--------|------------|------|-----|
| SteelCrack | 0.993 | 0.801 | 0.792 | 0.825 | 0.913 |
| YCD | 0.974 | 0.790 | 0.801 | 0.820 | 0.892 |
| Crack500 | 0.963 | 0.670 | 0.680 | 0.738 | 0.836 |
| DeepCrack | 0.985 | 0.807 | 0.836 | 0.851 | 0.903 |
| Ozgenel | 0.994 | 0.821 | 0.834 | 0.854 | 0.909 |

### Computational Efficiency (256×256×1 input)

| Method | mIoU | Params (M) | GFLOPs | Size (MB) | Inference (ms) |
|--------|------|------------|--------|-----------|----------------|
| UNet | 0.715 | 31.032 | 63.552 | 118.38 | 101.99 |
| ENet | 0.763 | 0.370 | 3.830 | 1.45 | 72.55 |
| CFPNet | 0.773 | 0.550 | 1.563 | 2.39 | 68.27 |
| **CrackSeg-GWD (Ours)** | **0.825** | **0.414** | **0.849** | **1.58** | **69.68** |

### Unified Multi-Material Model (Combined Training)

| Dataset | Accuracy | Recall | Dice Score | mIoU |
|---------|----------|--------|------------|------|
| SteelCrack | 0.991 | 0.828 | 0.757 | 0.801 |
| YCD | 0.969 | 0.831 | 0.782 | 0.805 |
| Crack500 | 0.957 | 0.749 | 0.672 | 0.731 |
| DeepCrack | 0.981 | 0.777 | 0.792 | 0.817 |
| Ozgenel (unseen) | 0.991 | 0.825 | 0.794 | 0.825 |
| QRCD (unseen, field) | 0.985 | 0.462 | 0.453 | 0.639 |

---

## Pretrained Weights

Pretrained model weights are available for download:

| Model | Dataset | mIoU | Download |
|-------|---------|------|----------|
| CrackSeg-GWD | SteelCrack | 0.825 | [Link](https://github.com/mohammedameen426/CrackSeg-GWD/releases) |
| CrackSeg-GWD | DeepCrack | 0.851 | [Link](https://github.com/mohammedameen426/CrackSeg-GWD/releases) |
| CrackSeg-GWD | Combined (Unified) | 0.817 | [Link](https://github.com/mohammedameen426/CrackSeg-GWD/releases) |

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

> **Patent Notice**: The method described in this repository is protected under a Chinese Invention Patent  
> (发明专利：一种裂缝图像分割方法及系统). The code is made available for academic and non-commercial  
> research purposes. For commercial use or licensing inquiries, please contact the authors.

---

## Citation

If you use CrackSeg-GWD in your research, please cite:

```bibtex
@article{mohammed2026crackseg,
  title     = {Automated lightweight networks for multi-material bridge crack segmentation},
  author    = {Mohammed, Mohammed Ameen and Zhou, Haijun and Zhang, Jiaolei and Xu, Shikun},
  journal   = {Automation in Construction},
  volume    = {183},
  pages     = {106808},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.autcon.2026.106808},
  url       = {https://doi.org/10.1016/j.autcon.2026.106808}
}
```

---

## Related Work

If you find this work useful, you may also be interested in our earlier work on semi-supervised crack segmentation:

```bibtex
@article{mohammed2024triplet,
  title   = {Enhanced pavement crack segmentation with minimal labeled data: a triplet attention teacher-student framework},
  author  = {Mohammed, Mohammed Ameen and Han, Zhen and Li, Yan and Al-Huda, Zaid and Wang, Wei},
  journal = {International Journal of Pavement Engineering},
  volume  = {25},
  number  = {1},
  year    = {2024},
  doi     = {10.1080/10298436.2024.2400562}
}
```

---
