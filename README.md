# Flower Species Classification with PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

Backstory
Blooms & Co is a rapidly growing floral e-commerce startup that allows customers to upload images of flowers they’d like to purchase. However, manually identifying thousands of flower images every day became overwhelming for their support team, leading to slow response times and inconsistent identifications. Blooms & Co needed an automated, reliable way to classify different flower species so they could streamline the ordering process and improve customer satisfaction.

Solution
This Flower Classifier Project demonstrates how a convolutional neural network (CNN) can solve Blooms & Co’s identification challenge. By training on a dataset of labeled flower images, the model can accurately categorize each image into specific species (e.g., daisies, roses, sunflowers). This reduces manual overhead, speeds up the workflow, and ensures a more consistent user experience. While this repository uses a sample dataset and structure, the approach can be adapted to real-world environments, helping any organization that relies on quick, accurate flower identification.


## 🌸 Project Overview
- **Dataset**: Oxford 102 Flower Dataset
- **Model**: ResNet34 with custom classifier head
- **Accuracy**: 73% on test set
- **Key Features**: CLI interface, checkpointing, top-K predictions

## 🛠️ Features
- Training pipeline with hyperparameter configuration
- Image preprocessing pipeline
- Model prediction with class probability mapping
- GPU acceleration support

## 📦 Installation
```bash
git clone https://github.com/yourusername/FlowerClassifier.git
cd FlowerClassifier
pip install -r requirements.txt 

```
## 🚀 Usage
Training:
```bash
python src/train.py data_dir --arch resnet34 --lr 0.001 --epochs 5 --gpu
```
Prediction:
```bash
python src/predict.py input_image.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json
```

## 📊 Results
Metric	        Value
Validation Acc	70%
Test Accuracy	  73%

## 📚 Resources
Dataset Source

PyTorch Documentation
