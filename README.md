# 🚀 CIFAR-10 Image Classification: ML vs. Deep Learning Benchmarking

A comprehensive comparison project evaluating classical Machine Learning and modern Deep Learning architectures on the **CIFAR-10** dataset using **PyTorch**.

This benchmark tracks model performance, loss convergence, accuracy metrics, and computational efficiency across four distinct architectures: **Linear SVM**, **Feedforward Neural Network (FNN/MLP)**, **Custom CNN**, and **ResNet50**.

---

## 📌 Features

- **End-to-End Pipeline**: Complete data loading, validation splitting, model training, evaluation, and visualization workflows.
- **Hardware Acceleration Setup**: Built-in support for CUDA/GPU execution with automatic CPU fallback.
- **PyTorch-based Linear SVM**: Implemented using mini-batch SGD optimization with `MultiMarginLoss` for high-throughput batch parallelism.
- **Deep Learning Architectures**: Custom multi-layer CNNs and transfer learning with pre-trained `ResNet50`.
- **Performance Visualization**: Automated plotting for loss trajectories and validation accuracy comparisons across epochs.

---

## 📊 Models & Benchmark Summary

| Architecture | Model Type | Parameters | Accuracy Range | Key Characteristics |
| :--- | :--- | :---: | :---: | :--- |
| **Linear SVM** | Classical Baseline | - | ~30% | Fast tensor-based SGD, limited spatial feature extraction |
| **FNN / MLP** | Multi-Layer Perceptron | ~1.7M | ~45% - 48% | Basic fully-connected network on flattened input arrays |
| **Custom CNN** | Convolutional Network | ~300K | ~70% - 74% | Efficient feature extraction with Conv2D, BatchNorm & Dropout |
| **ResNet50** | Residual Transfer Learning | ~23.5M | **>80%** | State-of-the-art residual blocks with rapid convergence |

---

## 📁 Repository Structure

```text
├── dataset.py      # CIFAR-10 dataset loading, train/val/test split & data visualization
├── models.py       # Definitions for MLP, Custom CNN, ResNet50 & parameter counter
├── train.py        # Hardware setup, SVM pipeline, and deep learning training loops
├── evaluate.py     # Test set evaluation and comparative loss/accuracy plotting
└── main.py         # Main execution script running the full benchmarking pipeline


🛠️ Installation & Setup

1- Clone the repository:

git clone [https://github.com/mark567167/cifar10-ml-vs-dl-benchmark.git](https://github.com/mark567167/cifar10-ml-vs-dl-benchmark.git)
cd cifar10-ml-vs-dl-benchmark

2-Install dependencies:

pip install torch torchvision matplotlib numpy

3- Run the Benchmark:

python main.py


📈 Key Learnings

1-Spatial Features Matter: Traditional linear classifiers and basic FNNs struggle with raw image pixels due to losing spatial context. Convolutional layers (Conv2D) are essential for extracting meaningful visual features.

2-Transfer Learning Power: Deep residual networks (ResNet50) achieve rapid convergence and high accuracy within very few epochs thanks to pre-trained weights and skip connections.

3-Compute vs. Performance Trade-off: Higher accuracy in deep learning comes at the cost of higher parameter counts and training execution time.