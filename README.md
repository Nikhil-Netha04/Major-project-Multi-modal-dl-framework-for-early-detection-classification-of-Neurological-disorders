# Multimodal Deep Learning Framework for Neurological Disorder Detection

## 🧠 Project Overview
This project presents a multimodal deep learning framework for the early detection and classification of neurological disorders using MRI scans. It focuses on Alzheimer's disease and brain stroke classification using EfficientNet-based models.

## 🚀 Features
- Supports multiple MRI modalities (T1, T2, FLAIR, DWI, SWI, GRE)
- Dual-model architecture:
  - EfficientNet-B3 for Alzheimer's classification
  - EfficientNet-B3 for stroke classification
- Decision-level fusion for automatic disease identification
- Streamlit web interface for real-time predictions

## 🛠️ Technologies Used
- Python
- PyTorch
- EfficientNet (B3)
- timm library
- NumPy, OpenCV, PIL
- Streamlit
- Jupyter Notebook / Google Colab

## 📂 Project Structure
```
├── app.py
├── models/
├── notebooks/
├── data/
├── requirements.txt
└── README.md
```

## ⚙️ Installation
```bash
git clone https://github.com/Nikhil-Netha04/Major-project-Multi-modal-dl-framework-for-early-detection-classification-of-Neurological-disorders.git
cd your-repo-name
pip install -r requirements.txt
```

## ▶️ Usage
```bash
streamlit run app.py
```

Upload an MRI image to get real-time classification results.

## 📊 Model Details
- Alzheimer's Model: EfficientNet-B3
- Stroke Model: EfficientNet-B3
- Loss Function: CrossEntropyLoss
- Optimizer: Adam

## 🎯 Results
The models achieve high accuracy and robust performance across multiple MRI modalities, enabling reliable neurological disorder classification.

## 🔮 Future Work
- Extend to additional neurological disorders
- Add explainable AI (XAI) features
- Integrate with clinical systems

## 📧 Contact
Nikhil Sambarapu,Manisha seervi 
Email: nikhilsa562@gmail.com,seervimanisha2005@gmail.com
