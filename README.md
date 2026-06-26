# FastViT: Hybrid Vision Transformer for Multi-Task Computer Vision

## Project Overview
This repository contains the end-to-end PyTorch implementation and evaluation of Apple's **FastViT** (`fastvit_s12` & `fastvit_ma36`) hybrid architectures. The project demonstrates the efficiency of combining CNN local feature extraction with Vision Transformer (ViT) global context for real-time deployment.

The framework is evaluated across two core computer vision tasks: **Image Classification** and **Object Detection**.

---

## Key Features & Implementations

### 1. Image Classification (CIFAR-10)
* **Model:** FastViT-S12
* **Pipeline:** Resized 60,000 images to $224 \times 224$, optimized via Cross-Entropy Loss and Adam.
* **Result:** Achieved **79% validation accuracy**, successfully validating the official baseline benchmarks.

### 2. Object Detection (Pascal VOC 2012)
* **Model:** FastViT-MA36
* **Pipeline:** Processed bounded multi-object arrays resized to $448 \times 448$, optimized via `SmoothL1Loss`.
* **Result:** Secured robust object localization with **IoU scores ranging from 0.65 to 0.78**.

---

## Project Structure
* `LAST COPY FOR AI PROJECT.ipynb`: Full PyTorch implementation, training loops, and evaluation pipelines.
* `FINAL PROJECT AI (REPORT) (1).ipynb`: Comparative analysis, architectural breakdown, and theoretical background.

## Tools & Frameworks
* **Frameworks:** PyTorch, Torchvision
* **Models:** Apple ML FastViT (`fastvit_s12`, `fastvit_ma36`)
* **Core Libraries:** NumPy, Pandas, Matplotlib, Scikit-Learn
