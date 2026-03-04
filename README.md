
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
import { useState } from "react";

const data = [
  { method: "UNet", mIoU: 0.715, params: 31.032, gflops: 63.552, size: 118.38, inference: 101.99, isProposed: false },
  { method: "Attention UNet", mIoU: 0.688, params: 31.380, gflops: 64.266, size: 119.71, inference: 100.18, isProposed: false },
  { method: "ResBCU-Net", mIoU: 0.726, params: 9.638, gflops: 19.737, size: 36.76, inference: 107.82, isProposed: false },
  { method: "SegNet", mIoU: 0.717, params: 29.44, gflops: 62.705, size: 20.84, inference: 80.29, isProposed: false },
  { method: "FCN", mIoU: 0.679, params: 7.007, gflops: 14.349, size: 26.73, inference: 78.73, isProposed: false },
  { method: "Swin-UNet", mIoU: 0.701, params: 29.83, gflops: 65.602, size: 118.34, inference: 129.42, isProposed: false },
  { method: "UNet-small", mIoU: 0.731, params: 0.537, gflops: 4.930, size: 1.84, inference: 69.83, isProposed: false },
  { method: "ULite", mIoU: 0.728, params: 0.878, gflops: 1.725, size: 3.21, inference: 71.69, isProposed: false },
  { method: "ENet", mIoU: 0.763, params: 0.370, gflops: 3.830, size: 1.45, inference: 72.55, isProposed: false },
  { method: "CFPNet", mIoU: 0.773, params: 0.550, gflops: 1.563, size: 2.39, inference: 68.27, isProposed: false },
  { method: "LiteFusionNet", mIoU: 0.771, params: 0.493, gflops: 1.252, size: 2.32, inference: 71.69, isProposed: false },
  { method: "CrackSeg-GWD", mIoU: 0.825, params: 0.414, gflops: 0.849, size: 1.58, inference: 69.68, isProposed: true },
];

const metrics = [
  { key: "mIoU", label: "mIoU", unit: "", higherBetter: true, format: v => v.toFixed(3) },
  { key: "params", label: "Params", unit: "M", higherBetter: false, format: v => v.toFixed(3) },
  { key: "gflops", label: "GFLOPs", unit: "", higherBetter: false, format: v => v.toFixed(3) },
  { key: "size", label: "Size", unit: "MB", higherBetter: false, format: v => v.toFixed(2) },
  { key: "inference", label: "Inference", unit: "ms", higherBetter: false, format: v => v.toFixed(2) },
];

function Bar({ value, max, color, highlight }) {
  const pct = (value / max) * 100;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, width: "100%" }}>
      <div style={{ flex: 1, background: "#1a1f2e", borderRadius: 3, height: 8, overflow: "hidden" }}>
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: highlight ? "linear-gradient(90deg, #00e5ff, #00b8d9)" : color,
            borderRadius: 3,
            transition: "width 0.6s ease",
          }}
        />
      </div>
    </div>
  );
}

export default function Table11() {
  const [activeMetric, setActiveMetric] = useState("mIoU");
  const [sortBy, setSortBy] = useState("mIoU");
  const [sortAsc, setSortAsc] = useState(false);

  const metaObj = metrics.find(m => m.key === sortBy);
  const sorted = [...data].sort((a, b) => {
    const diff = a[sortBy] - b[sortBy];
    return sortAsc ? diff : -diff;
  });

  const maxVals = {};
  metrics.forEach(m => { maxVals[m.key] = Math.max(...data.map(d => d[m.key])); });

  const handleSort = (key) => {
    if (sortBy === key) setSortAsc(p => !p);
    else { setSortBy(key); setSortAsc(false); }
  };

  return (
    <div style={{
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      background: "#0d1117",
      minHeight: "100vh",
      padding: "32px 24px",
      color: "#c9d1d9",
    }}>
      {/* Header */}
      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        <div style={{ marginBottom: 6, display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 4, height: 32, background: "linear-gradient(180deg,#00e5ff,#0072ff)", borderRadius: 2 }} />
          <div>
            <div style={{ fontSize: 11, color: "#8b949e", letterSpacing: 3, textTransform: "uppercase", marginBottom: 2 }}>
              Table 11 — Automation in Construction 183 (2026) 106808
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, color: "#e6edf3", letterSpacing: -0.5 }}>
              Computational Cost Analysis
            </div>
          </div>
        </div>
        <div style={{ fontSize: 11, color: "#484f58", marginBottom: 28, paddingLeft: 14 }}>
          Input: 256 × 256 × 1 tensor · 12 segmentation models compared
        </div>

        {/* Metric filter pills */}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20 }}>
          {metrics.map(m => (
            <button
              key={m.key}
              onClick={() => setActiveMetric(m.key)}
              style={{
                padding: "5px 14px",
                borderRadius: 20,
                border: "1px solid",
                borderColor: activeMetric === m.key ? "#00e5ff" : "#30363d",
                background: activeMetric === m.key ? "rgba(0,229,255,0.08)" : "transparent",
                color: activeMetric === m.key ? "#00e5ff" : "#8b949e",
                fontSize: 11,
                cursor: "pointer",
                letterSpacing: 1,
                textTransform: "uppercase",
                fontFamily: "inherit",
                transition: "all 0.2s",
              }}
            >
              {m.label}{m.unit ? ` (${m.unit})` : ""}
            </button>
          ))}
        </div>

        {/* Bar chart for active metric */}
        <div style={{
          background: "#161b22",
          border: "1px solid #21262d",
          borderRadius: 10,
          padding: "20px 24px",
          marginBottom: 24,
        }}>
          <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 16, letterSpacing: 2, textTransform: "uppercase" }}>
            {metrics.find(m => m.key === activeMetric)?.label} comparison
            <span style={{ marginLeft: 8, color: "#484f58" }}>
              · {metrics.find(m => m.key === activeMetric)?.higherBetter ? "↑ Higher is better" : "↓ Lower is better"}
            </span>
          </div>
          {[...data]
            .sort((a, b) => {
              const m = metrics.find(x => x.key === activeMetric);
              return m.higherBetter ? b[activeMetric] - a[activeMetric] : a[activeMetric] - b[activeMetric];
            })
            .map((row, i) => {
              const m = metrics.find(x => x.key === activeMetric);
              const val = row[activeMetric];
              const best = m.higherBetter
                ? Math.max(...data.map(d => d[activeMetric]))
                : Math.max(...data.map(d => d[activeMetric]));
              return (
                <div key={row.method} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 9 }}>
                  <div style={{
                    width: 120, fontSize: 11,
                    color: row.isProposed ? "#00e5ff" : i === 0 ? "#58a6ff" : "#8b949e",
                    fontWeight: row.isProposed ? 700 : 400,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}>
                    {row.isProposed ? "★ " : ""}{row.method}
                  </div>
                  <div style={{ flex: 1 }}>
                    <Bar
                      value={val}
                      max={best}
                      color={i === 0 ? "rgba(88,166,255,0.5)" : "#21262d"}
                      highlight={row.isProposed}
                    />
                  </div>
                  <div style={{
                    width: 64, textAlign: "right", fontSize: 12,
                    color: row.isProposed ? "#00e5ff" : "#c9d1d9",
                    fontWeight: row.isProposed ? 700 : 400,
                  }}>
                    {m.format(val)}{m.unit ? " " + m.unit : ""}
                  </div>
                </div>
              );
            })}
        </div>

        {/* Full table */}
        <div style={{
          background: "#161b22",
          border: "1px solid #21262d",
          borderRadius: 10,
          overflow: "hidden",
        }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #21262d" }}>
                <th style={{ padding: "12px 16px", textAlign: "left", color: "#8b949e", fontWeight: 500, fontSize: 11, letterSpacing: 1 }}>
                  METHOD
                </th>
                {metrics.map(m => (
                  <th
                    key={m.key}
                    onClick={() => handleSort(m.key)}
                    style={{
                      padding: "12px 16px",
                      textAlign: "right",
                      color: sortBy === m.key ? "#00e5ff" : "#8b949e",
                      fontWeight: 500,
                      fontSize: 11,
                      letterSpacing: 1,
                      cursor: "pointer",
                      userSelect: "none",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {m.label.toUpperCase()}{m.unit ? ` (${m.unit})` : ""}
                    {sortBy === m.key ? (sortAsc ? " ↑" : " ↓") : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((row, i) => {
                const bg = row.isProposed
                  ? "rgba(0,229,255,0.05)"
                  : i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)";
                return (
                  <tr
                    key={row.method}
                    style={{
                      background: bg,
                      borderBottom: "1px solid #21262d",
                      borderLeft: row.isProposed ? "3px solid #00e5ff" : "3px solid transparent",
                    }}
                  >
                    <td style={{
                      padding: "10px 16px",
                      color: row.isProposed ? "#00e5ff" : "#e6edf3",
                      fontWeight: row.isProposed ? 700 : 400,
                    }}>
                      {row.isProposed ? "★ " : ""}{row.method}
                    </td>
                    {metrics.map(m => {
                      const val = row[m.key];
                      const allVals = data.map(d => d[m.key]);
                      const isBest = m.higherBetter
                        ? val === Math.max(...allVals)
                        : val === Math.min(...allVals);
                      return (
                        <td key={m.key} style={{
                          padding: "10px 16px",
                          textAlign: "right",
                          color: row.isProposed ? "#00e5ff" : isBest ? "#3fb950" : "#c9d1d9",
                          fontWeight: isBest || row.isProposed ? 700 : 400,
                        }}>
                          {m.format(val)}
                          {isBest && !row.isProposed &&
                            <span style={{ marginLeft: 4, fontSize: 9, color: "#3fb950" }}>●</span>
                          }
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Legend */}
        <div style={{ display: "flex", gap: 20, marginTop: 14, fontSize: 10, color: "#484f58" }}>
          <span><span style={{ color: "#00e5ff" }}>★ cyan</span> = Proposed model (CrackSeg-GWD)</span>
          <span><span style={{ color: "#3fb950" }}>● green</span> = Best value per metric</span>
          <span>Click column headers to sort</span>
        </div>
      </div>
    </div>
  );
}
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
