
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

![CrackSeg-GWD Architecture](figures/image.png)

The architecture follows a U-Net-inspired encoder–decoder structure with four key innovations:

1. **Group Normalization (GN)** — Provides stable training independent of batch size, preserving fine-grained crack structures across diverse surface textures.
2. **Weight-Standardized Convolutions (WS)** — Normalizes convolutional filter weights for smoother optimization and more stable gradient flow, especially when working with high-resolution images.
3. **DropBlock Regularization** — Removes contiguous feature regions (block size 7, keep probability 0.9) to prevent overfitting and force the network to learn spatially distributed features.
4. **Symmetric Unified Focal (SUF) Loss** — Combines a region-level Focal Tversky Loss with pixel-level Mean Absolute Error to address severe class imbalance (cracks occupy < 2% of image pixels).

The encoder comprises three progressively deeper residual blocks (32 → 64 → 128 filters), followed by a three-stage decoder using `Conv2DTranspose` for learnable upsampling. Dense Conditional Random Fields (DenseCRFs) are applied as a post-processing step for sharper, more continuous crack maps.

---

## Datasets

The following publicly available datasets were used to train and evaluate CrackSeg-GWD:
SteelCrack: 4355 images (3300 train / 530 test / 525 val), resized to 512×512. Collected from the Nanjing Second Yangtze River Bridge and Humen Bridge; includes diverse steel crack types under varying environmental conditions. https://github.com/hzlbbfrog/Civil-dataset. 
YCD: 776 images (622 train / 154 test), 512×512. Contains road cracks and concrete wall cracks captured at varying camera distances and scales. [https://drive.google.com/file/d/1imZTwMm20vKgPv9ESVxgZZQnDw01FNdB/view?usp=drive_link](https://drive.google.com/file/d/1imZTwMm20vKgPv9ESVxgZZQnDw01FNdB/view?usp=sharing)
Crack500: 3368 images (2244 train / 1124 test), 360×640. Challenging pavement crack dataset with inconsistent widths, occlusions, and visual clutter. https://github.com/fyangneil/pavement-crack-detection
DeepCrack: 530 images (300 train / 230 test), 544×384. Pixel-level annotated dataset covering diverse crack types under shadows, moisture, and variable lighting. https://github.com/yhlleo/DeepCrack
Ozgenel: 458 high-resolution images (366 train / 92 test), 4032×3024. Concrete crack images collected from buildings at Middle East Technical University. https://data.mendeley.com/datasets/jwsn7tfbrp/1
QRCD (Field-collected): 128 test-only images, 3024×3024. High-resolution crack images collected from Qingyuan Road under real-world inspection conditions (not used in training).  [https://drive.google.com/file/d/1HHUAAsJBGeT0KJ_C4NkNEAw3XMQOWd0s/view?usp=drive_link](https://drive.google.com/file/d/1qMY0zxnFS7HQYdm6etn0biimc1LT63-u/view?usp=sharing)

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

> **Patent Notice**: The method described in this repository is protected under a  
> Chinese Invention Patent (发明专利：一种裂缝图像分割方法及系统,  
> 专利号：ZL 2025 1 1400055.6).  
> The code is made available for **academic and non-commercial research purposes only**.  
> For commercial use or licensing inquiries, please contact the authors.
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
