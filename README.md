# MEICN-GAT: A Multi-Expert Capsule Network for High-end Business Markets

This is the official source code for the paper: **"Tackling Data Imbalance in High-end Business Markets: A Multi-Expert Capsule Network Approach"**.

[![Paper-Link-Coming-Soon](https://img.shields.io/badge/Paper-Link%20Coming%20Soon-red)](#)
[![Code-License-MIT](https://img.shields.io/badge/License-MIT-blue)](#)

---

## 摘要 (Abstract)

在数字时代，高端商业市场面临着一个独特的营销挑战：如何从庞大的用户池中识别出潜在客户，而非传统地帮助用户发现商品。这种逆向推荐场景带来了两个关键的技术挑战：（1）海量用户与有限高端商品之间的极端不平衡；（2）用有限的商品集难以捕捉多样化的用户偏好。

我们提出了MEICN-GAT (Multi-Level and Expert Interest Capsule Network with Graph Attention)，一个专为商业中心化营销设计的创新推荐系统。MEICN-GAT通过三大创新解决了这些挑战：（1）一个双层GAT结构，在多个层次上捕捉复杂的用户-商品关系，有效管理用户-商品比例失衡问题；（2）一个多专家系统，从多个专业视角评估用户兴趣，缓解数据稀疏性问题；（3）一个自适应机制，根据每个商品的独有特性定制推荐，确保精准的客户定位。

以豪华酒店行业为案例研究，在真实世界数据集上的实验评估表明，MEICN-GAT在客户识别准确率、排名质量和匹配成功率方面显著优于现有方法。

## 环境配置 (Installation)

本项目在Python 3.7下开发。我们强烈建议使用`conda`来管理环境，因为部分依赖包有特定的安装通道要求。

1.  **创建并激活Conda环境:**
    ```bash
    conda create -n meicn-gat-env python=3.7
    conda activate meicn-gat-env
    ```

2.  **安装PyTorch和DGL:**
    这些包需要特定版本和通道以确保兼容性。请根据您的CUDA版本手动安装。原始实验是在CUDA 11.1下运行的。
    ```bash
    # 针对CUDA 11.1 (建议用于精确复现)
    conda install pytorch==1.9.0 torchvision==0.10.0 -c pytorch
    conda install dgl-cuda11.1==0.9.1 -c dglteam
    ```

3.  **安装其他依赖:**
    使用`pip`安装剩余的依赖包。`requirements.txt` 文件提供了核心包的版本信息。
    ```bash
    pip install numpy==1.21.6 prettytable==2.1.0 scikit-learn==1.0.2 pandas==1.3.5 matplotlib==3.5.3
    ```

## 如何运行 (Usage)

1.  **数据预处理:**
    运行预处理脚本来构建图数据。请确保您的原始数据文件已按要求放置。
    ```bash
    python data/preprocess.py
    ```
    *(注意: 您可能需要根据实际情况修改 `preprocess.py` 中的文件路径。)*

2.  **模型训练与评估:**
    运行主脚本来启动模型训练和评估。
    ```bash
    python run_meicn_gat.py --dataset [dataset_name] --gpu_id [gpu_id]
    ```
    *   `--dataset`: 指定数据集名称 (例如: `hotel`, `gift_card`).
    *   `--gpu_id`: 指定使用的GPU ID.

## 项目结构 (Project Structure)
```
MEICN-GAT_for_GitHub/
│
├── run_meicn_gat.py        # 主运行脚本
├── requirements.txt        # 依赖文件
├── README.md               # 项目说明 (本文档)
│
├── data/
│   └── preprocess.py       # 数据预处理脚本
│
├── src/
│   └── base_model.py       # MEICN-GAT模型定义
│
└── logs/
    └── ...                 # 实验日志文件
```

## 如何引用 (Citation)

我们的论文目前正在投稿中。一旦被接收，我们将在此处提供详细的引用信息。

---
<br>

# MEICN-GAT: A Multi-Expert Capsule Network for High-end Business Markets (English Version)

This is the official source code for the paper: **"Tackling Data Imbalance in High-end Business Markets: A Multi-Expert Capsule Network Approach"**.

## Abstract

In the digital era, luxury businesses face a unique marketing challenge: identifying potential customers from vast user pools rather than helping users find products. This reversed recommendation scenario presents two key technical challenges: (1) the imbalance between large user pools and limited premium items and (2) the difficulty of capturing diverse user preferences with a limited set of items.

We introduce MEICN-GAT (Multi-Level and Expert Interest Capsule Network with Graph Attention), a novel recommendation system designed for business-centric marketing. MEICN-GAT addresses these challenges with three key innovations: (1) a dual-layer structure that captures complex user-product relationships at multiple levels, effectively managing the user-item ratio imbalance; (2) a multi-expert system that evaluates user interests from diverse professional perspectives, alleviating the data sparsity problem; and (3) an adaptive mechanism that tailors recommendations based on each product's unique characteristics, ensuring precise customer targeting.

Using the luxury hotel industry as a case study, empirical evaluations on real-world datasets show that MEICN-GAT significantly outperforms existing methods in customer identification accuracy, ranking quality, and matching success rates.

## Installation

This project was developed using Python 3.7. We strongly recommend using `conda` to manage the environment, as some dependencies have specific channel requirements.

1.  **Create and activate a conda environment:**
    ```bash
    conda create -n meicn-gat-env python=3.7
    conda activate meicn-gat-env
    ```

2.  **Install PyTorch and DGL:**
    These packages require specific versions and channels to ensure compatibility. Please install them manually according to your CUDA version. The original experiment was run with CUDA 11.1.
    ```bash
    # For CUDA 11.1 (Recommended for exact reproduction)
    conda install pytorch==1.9.0 torchvision==0.10.0 -c pytorch
    conda install dgl-cuda11.1==0.9.1 -c dglteam
    ```

3.  **Install other dependencies:**
    Install the remaining packages using `pip`. The `requirements.txt` file provides the versions for key packages.
    ```bash
    pip install numpy==1.21.6 prettytable==2.1.0 scikit-learn==1.0.2 pandas==1.3.5 matplotlib==3.5.3
    ```

## Usage

1.  **Data Preprocessing:**
    Run the preprocessing script to build the graph data. Please ensure your raw data files are placed as required.
    ```bash
    python data/preprocess.py
    ```
    *(Note: You may need to modify file paths in `preprocess.py` according to your setup.)*

2.  **Model Training & Evaluation:**
    Run the main script to start model training and evaluation.
    ```bash
    python run_meicn_gat.py --dataset [dataset_name] --gpu_id [gpu_id]
    ```
    *   `--dataset`: Specify the dataset name (e.g., `hotel`, `gift_card`).
    *   `--gpu_id`: Specify the GPU ID to use.

## Project Structure
```
MEICN-GAT_for_GitHub/
│
├── run_meicn_gat.py        # Main execution script
├── requirements.txt        # Dependency file
├── README.md               # Project description (this document)
│
├── data/
│   └── preprocess.py       # Data preprocessing script
│
├── src/
│   └── base_model.py       # MEICN-GAT model definition
│
└── logs/
    └── ...                 # Experiment log files
```

## Citation

Our paper is currently under review. Upon acceptance, we will provide detailed citation information here.
