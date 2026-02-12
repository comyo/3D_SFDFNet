# 3D-SFDFNet: 3D-Structured Feature Dual-Stream Fusion Network

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/comyoy/ompsmembrane-separation-dataset)

> **Official Implementation of "Predicting Membrane Rejection of OMPs via 3D-SFDFNet"**
> 
> In this study, we propose a **3D-Structured Feature Dual-Stream Fusion Network (3D-SFDFNet)** for predicting membrane rejection of organic micropollutants (OMPs) by integrating **3D molecular features** with **structured physicochemical descriptors**.

---

## 🏗️ Core Architecture

The 3D-SFDFNet framework consists of three core modules:

### 1. OMPs 3D Molecular Representation Module
* **Voxel Mapping**: The 3D molecular structures of OMPs are mapped into regular voxel grids, systematically preserving key information such as atomic spatial distribution, chemical bond connectivity, and molecular conformations.
* **Feature Extraction**: Deep representation learning is performed on the voxelized molecules using a lightweight **VoxelCNN**.

### 2. Structured Feature Encoding Module
* **Hybrid Encoding**: To handle the diverse and heterogeneous nature of structured tabular data, this module combines **XGBoost**, **one-hot encoding**, and a **multilayer perceptron (MLP)**.
* **Uniform Mapping**: Molecular physicochemical parameters, membrane parameters, and operating parameters are uniformly encoded and mapped into low-dimensional feature vectors through the MLP.

### 3. Deep Fusion Module
* **Multi-source Integration**: The 3D molecular features and structured descriptors are concatenated and combined through **element-wise interactions** to achieve deep integration of information.
* **Prediction Output**: The fused representation is passed through a fully connected network to output the predicted membrane rejection rates of OMPs.

---

## 📊 Datasets

The dataset supporting this research is available on Kaggle:
👉 **[OMPs Membrane Separation Dataset](https://www.kaggle.com/datasets/comyoy/ompsmembrane-separation-dataset)**

**The dataset includes:**
* 3D molecular conformation data for various OMPs.
* Membrane physicochemical parameters and experimental operating conditions.

