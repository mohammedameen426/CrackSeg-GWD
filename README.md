
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

1. The SteelCrack dataset is a widely used benchmark for steel crack detection and segmentation. The dataset is divided into 3,300 training images, 525 validation images, and 530 test images, offering a well-balanced split for deep learning–based segmentation studies. "Z. He, W. Chen, J. Zhang, Y.H. Wang, Crack segmentation on steel structures using boundary guidance model, Autom Constr 162 (2024). https://doi.org/10.1016/j.autcon.2024.105354."

SteelCrack dataset :https://github.com/hzlbbfrog/Civil-dataset. 

2. The YCD dataset contains images of two main types of surface defects: road cracks and concrete wall cracks. The dataset includes 776 images, split into 622 for training and 154 for testing. All images were resized to 512 × 512, matching the preprocessing used for other datasets such as DeepCrack537. Figure 7 shows sample images that reflect the dataset’s diversity in crack types and scales. "X. Yang, H. Li, Y. Yu, X. Luo, T. Huang, X. Yang, Automatic Pixel-Level Crack Detection and Measurement Using Fully Convolutional Network, Computer-Aided Civil and Infrastructure Engineering 33 (2018). https://doi.org/10.1111/mice.12412."

YCD dataset : [https://drive.google.com/file/d/1imZTwMm20vKgPv9ESVxgZZQnDw01FNdB/view?usp=drive_link](https://drive.google.com/file/d/1imZTwMm20vKgPv9ESVxgZZQnDw01FNdB/view?usp=sharing)

3. Ozgenel Crack Segmentation Dataset: The concrete dataset comprises a total of 3,616 images, each sized 4032 by 33024 pixels. Among these, 1,744 images, along with their corresponding ground truth, were derived. This subset includes 458 high-resolution images of concrete surfaces captured at Middle East Technical University.  "Çağ lar Fırat Özgenel, Concrete Crack Images for Classification, 15 Jan 1 (2018)."
   
Ozgenel dataset:https://data.mendeley.com/datasets/jwsn7tfbrp/1

4. DeepCrack dataset comprises 537 images with a resolution of 544 x 384 pixels, accompanied by ground truth images at the pixel level. "Y. Liu, J. Yao, X. Lu, R. Xie, L. Li, DeepCrack: A deep hierarchical feature learning architecture for crack segmentation, Neurocomputing 338 (2019). https://doi.org/10.1016/j.neucom.2019.01.036."
   
DeepCrack dataset:https://github.com/yhlleo/DeepCrack

5. Crack500 dataset consists of 1896 training images and 1124 testing images, all at a resolution of 640 × 360 pixels. Notably, Crack500 offers a diverse range of crack shapes and widths, presenting significant challenges for crack segmentation tasks. "F. Yang, L. Zhang, S. Yu, D. Prokhorov, X. Mei, H. Ling, Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection, IEEE Transactions on Intelligent Transportation Systems 21 (2020). https://doi.org/10.1109/TITS.2019.2910595"

Crack500 dataset: https://github.com/fyangneil/pavement-crack-detection

6. Qingyuan_Road dataset (QRCD): We collected 128 high-resolution crack images (3024 × 3024) from sections of Qingyuan Road. The dataset includes real-world challenges such as strong shadows, uneven illumination, surface noise, and complex textures. These samples are used to test and evaluate the robustness of our model.

Qingyuan_Road dataset (QRCD): [https://drive.google.com/file/d/1HHUAAsJBGeT0KJ_C4NkNEAw3XMQOWd0s/view?usp=drive_link](https://drive.google.com/file/d/1qMY0zxnFS7HQYdm6etn0biimc1LT63-u/view?usp=sharing)
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
