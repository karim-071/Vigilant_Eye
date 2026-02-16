# 👁️ Vigilant Eye — Deepfake Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> An AI-powered deepfake detection system capable of identifying manipulated **images and videos** using deep learning and computer vision.

---

## 🚀 Overview

**Vigilant Eye** is an end-to-end deepfake detection platform that identifies manipulated media by leveraging transfer learning, face detection, and frame-level video analysis.

The system supports both **image and video deepfake detection**, providing real-time predictions through an interactive web application.

---

## ✨ Key Features

- 🖼️ Detect deepfakes in **images**
- 🎥 Detect deepfakes in **videos**
- 👤 Automatic face detection & cropping
- ⚡ GPU-accelerated inference
- 📊 Confidence score visualization
- 🧠 Transfer learning with EfficientNet
- 🌐 Interactive web interface

---
## 🚀 Demo

👉 **Live App:** *(Add your Streamlit link here)* 

---

## 🧠 Model Architecture

### Backbone Model
- **EfficientNet-B0** (Pretrained on ImageNet)

### Training Strategy
- Transfer learning with frozen feature layers
- Fine-tuned classification head
- Binary classification using sigmoid output

---

## 📊 Performance Metrics

| Metric | Score |
|--------|------|
Accuracy | ~79% |
Precision | 0.83 |
Recall | 0.73 |
F1 Score | 0.78 |
ROC-AUC | **0.88** |

**Dataset Size:** 140,000+ images

--- 
## 🧪 How It Works

### Image Detection Pipeline

1. Upload image
2. Detect face using OpenCV
3. Crop & preprocess
4. Run model inference
5. Output prediction with confidence score

### Video Detection Pipeline

1. Extract frames at fixed intervals
2. Detect faces per frame
3. Run inference on each frame
4. Aggregate predictions using **median probability**
5. Output final classification

---

## 🛠️ Tech Stack

### 🤖 AI / Machine Learning
- PyTorch
- TorchVision
- Transfer Learning
- Computer Vision

### 🧰 Libraries
- OpenCV
- NumPy
- PIL

---

## 🔮 Future Improvements

1. Use advanced face detectors (RetinaFace / MTCNN)
2. Add Grad-CAM explainability
3. Implement temporal deepfake detection models
4. Improve dataset diversity
5. Deploy REST API backend

---
