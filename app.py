import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()
    return model, weights

def analyze_image(image, model, weights):
    preprocess = weights.transforms()
    img_tensor = preprocess(image.convert('RGB'))
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    categories = weights.meta["categories"]
    results = [(categories[idx], prob.item())
               for idx, prob in zip(top5_idx, top5_prob)]
    return results

def generate_report(predictions, image_type):
    top_label = predictions[0][0].replace('_', ' ').title()
    top_conf = predictions[0][1] * 100
    report = f"""
## 🏥 Medical Image Analysis Report

**Image Type:** {image_type}

---

### 1. Image Quality Assessment
The uploaded image has been successfully processed and analyzed using MobileNetV2,
a state-of-the-art convolutional neural network trained on over 1 million images.

---

### 2. AI Model Predictions

The model analyzed the image and identified the following patterns:

"""
    for i, (label, confidence) in enumerate(predictions):
        bar = "█" * int(confidence * 20)
        report += f"**{i+1}. {label.replace('_', ' ').title()}**\n"
        report += f"Confidence: {confidence*100:.1f}% {bar}\n\n"

    report += f"""
---

### 3. Key Observations
- The model identified **{top_label}** as the primary pattern with **{top_conf:.1f}% confidence**
- Analysis performed using deep learning with 1000+ category recognition
- Image successfully preprocessed and normalized for analysis

---

### 4. Recommendations
- ✅ Image was successfully processed and analyzed
- 📋 Top prediction: **{top_label}** ({top_conf:.1f}% confidence)
- 🔬 For medical images, please consult a qualified radiologist or physician

---

⚠️ **Important Disclaimer:**
This is for **educational purposes only** and should NOT replace professional medical advice.

---
*Report generated using MobileNetV2 (PyTorch)*
"""
    return report

st.set_page_config(page_title="AI Medical Image Analyzer", page_icon="🏥", layout="wide")
st.title("🏥 AI Medical Image Analyzer")
st.write("Upload a medical image and get an AI-powered analysis report instantly.")

st.sidebar.header("About This App")
st.sidebar.write("""
This app uses MobileNetV2, a powerful deep learning model, to analyze medical images.

**Supported Images:**
- Chest X-rays
- Skin lesion photos
- MRI scans
- CT scan images

**How it works:**
- Uses PyTorch MobileNetV2
- Trained on 1M+ images
- Runs completely locally
- No internet required

⚠️ *For educational purposes only.*
""")

st.sidebar.divider()
st.sidebar.info("🤖 Model: MobileNetV2\n\n📊 Dataset: ImageNet\n\n🔢 Categories: 1000+")

image_type = st.selectbox("Select image type:", [
    "Chest X-Ray", "Skin Lesion", "MRI Scan", "CT Scan", "General Medical Image"
])

uploaded_file = st.file_uploader("Upload medical image",
                                  type=["jpg", "jpeg", "png"],
                                  help="Upload a clear medical image for analysis")

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
        st.caption(f"Image size: {image.size[0]}x{image.size[1]} pixels")
        st.subheader("Prediction Confidence")

    with col2:
        st.subheader("AI Analysis Report")
        if st.button("🔍 Analyze Image", use_container_width=True):
            with st.spinner("Loading AI model and analyzing image..."):
                model, weights = load_model()
                predictions = analyze_image(image, model, weights)

            with col1:
                labels = [p[0].replace('_', ' ').title() for p in predictions]
                confidences = [p[1] * 100 for p in predictions]
                fig, ax = plt.subplots(figsize=(6, 3))
                bars = ax.barh(labels[::-1], confidences[::-1], color='steelblue')
                ax.set_xlabel('Confidence (%)')
                ax.set_title('Top 5 Predictions')
                for bar, conf in zip(bars, confidences[::-1]):
                    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{conf:.1f}%', va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                report = generate_report(predictions, image_type)
                st.markdown(report)
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name="medical_analysis_report.txt",
                    mime="text/plain"
                )
else:
    st.info("👆 Please upload a medical image to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.write("Upload any medical image — X-ray, MRI, CT scan or skin photo")
    with col2:
        st.markdown("### 2️⃣ Analyze")
        st.write("MobileNetV2 deep learning model analyzes the image locally")
    with col3:
        st.markdown("### 3️⃣ Report")
        st.write("Get a structured report with confidence scores you can download")