# 🏥 AI Medical Image Analyzer

A computer vision web app that analyzes medical images and generates 
structured AI-powered reports instantly.

## 🔗 Live Demo
[👉 Click here to try it live](https://celeste-hephzibah-ai-medical-analyzer.streamlit.app)

## 📌 Overview
Built an end-to-end computer vision pipeline using MobileNetV2, a 
state-of-the-art CNN trained on 1 million+ images. Users can upload 
any medical image and get a structured analysis report with confidence 
scores — all running locally with no API needed.

## ✨ Features
- Upload chest X-rays, MRI scans, CT scans, skin lesion images
- Top 5 predictions with confidence scores
- Visual confidence bar chart
- Downloadable structured medical report
- Runs 100% locally — no internet required

## 🛠️ Tech Stack
- Python
- PyTorch
- MobileNetV2 (ImageNet pretrained)
- Streamlit
- Pillow
- Matplotlib

## 🔬 How It Works
1. User uploads a medical image (JPG/PNG)
2. Image resized and preprocessed for MobileNetV2
3. CNN extracts features and classifies image
4. Top 5 predictions returned with confidence scores
5. Structured medical report generated and displayed
6. Report available for download

## 💡 Interview Note
MobileNetV2 is pretrained on ImageNet (general objects). For a 
production medical system, I would fine-tune it on a domain-specific 
dataset like NIH ChestX-ray14 (112,000 labeled chest X-rays).

## 🚀 Run Locally
```bash
git clone https://github.com/Celeste-Hephzibah/ai-medical-analyzer
cd ai-medical-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
ai-medical-analyzer/

├── app.py              # Streamlit web app

├── requirements.txt

└── runtime.txt
## ⚠️ Disclaimer
This app is for educational purposes only and should not be used 
as a substitute for professional medical diagnosis.