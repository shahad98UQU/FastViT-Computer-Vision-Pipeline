# Hybrid Vision Transformer (FastViT) for Multi-Task Computer Vision

## Project Overview
This repository contains a comprehensive implementation and evaluation of **FastViT-S12** and **FastViT-MA36**, state-of-the-art hybrid vision transformer architectures developed by Apple. By combining the local feature extraction of Convolutional Neural Networks (CNNs) with the global contextual modeling of Vision Transformers (ViTs), FastViT achieves an optimal trade-off between structural accuracy and computational latency, making it highly efficient for real-time applications.

The project evaluates the architecture across two primary computer vision tasks: **Image Classification** and **Object Detection**, using external benchmarking datasets.

---

## Architecture Core Innovations
* **RepMixer Block:** A novel token mixer that removes residual skip connections through structural reparameterization, significantly lowering memory access costs (MACs).
* **Train-Time Overparameterization:** Introduces un-factorized linear layers during training to expand model capacity, which are compressed back during inference to eliminate computational overhead.
* **Large Kernel Convolutions:** Replaces expensive self-attention mechanisms in the early stages of the network to enhance receptive fields while keeping latency low.

---

## Implemented Tasks & Experimental Results

### 1. Image Classification (CIFAR-10 Dataset)
* **Dataset Summary:** 60,000 low-resolution (32x32) color images spanned across 10 distinct structural categories.
* **Preprocessing Pipeline:** Input dimensions resized to $224 \times 224$ pixels and converted into PyTorch Tensors before feeding into the **FastViT-S12** model.
* **Training & Loss Optimization:** Trained with a Batch Size of 64 using Cross-Entropy Loss optimization.
* **Performance Analysis:** The model convergence demonstrated solid learning stability without overfitting. Validation loss successfully dropped to ~0.5 while matching the baseline scientific paper metrics, achieving a final validation accuracy of **79%**.

### 2. Object Detection (Pascal VOC 2012 Dataset)
* **Dataset Summary:** Complex multi-object dataset encompassing diverse categories including vehicles, pedestrians, and animals with provided bounding box annotations.
* **Preprocessing Pipeline:** Input images resized to a high-resolution $448 \times 448$ spatial grid and normalized into batched PyTorch structures.
* **Training Head:** Evaluated using the faster **FastViT-MA36** variant, optimized utilizing `SmoothL1Loss` for box regression and Adam optimizer.
* **Evaluation Metric (IoU):** The intersection over Union (IoU) scores achieved on test samples settled robustly within the **0.65 to 0.78** bracket, proving high structural localization fidelity.

---

## Project Structure
```text
├── fastvit_multi_task_pipeline.ipynb   # Main Jupyter Notebook containing the full pipeline
├── images/                             # Contains model architecture diagrams and output graphs
└── README.md                           # Project documentation
