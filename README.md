# CrackSeg-GWD
CrackSeg-GWD


# Lightweight-CrackSeg-GWD
CrackSeg-GWD: Automatic lightweight networks for Multi-Material Bridge Crack Segmentation

Overview

CrackSeg-GWD is a lightweight segmentation model designed for detecting cracks on concrete, steel, and asphalt surfaces, where challenges such as irregular crack shapes, low contrast, noise, and complex backgrounds often reduce accuracy. Existing deep-learning approaches either lack strong multi-scale feature representation or are too heavy for deployment on edge devices, and no single compact model performs reliably across different material types.
CrackSeg-GWD addresses these issues using a streamlined encoder–decoder architecture that incorporates Group Normalization, Weight-Standardized Convolutions, DropBlock regularization, and a Symmetric Unified Focal Loss to improve stability, reduce overfitting, and handle class imbalance. With just 0.414M parameters and 0.849 GFLOPs, it offers high efficiency while maintaining strong performance.
Tested on five public datasets—SteelCrack, YCD, Crack500, DeepCrack, and Ozgenel—the model achieves consistent improvements over ten state-of-the-art baselines, making it well-suited for real-time structural inspection and construction automation.

# 📄 Paper 
The code and pre-trained models will be released once the corresponding paper is officially accepted.


# Datasets used on this study:
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


# References:
https://github.com/asifehmad/ResBCU-Net-for-Polyps-Segmentation.git


https://github.com/ashish-s-bisht/SwinUnetArchitecturePytorch.git

https://arxiv.org/abs/2105.05537

https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py
