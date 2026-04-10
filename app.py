import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ---------------- DEVICE ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- CLASS ORDER (from training) ----------------
classes = [
    "ALZ_MildDemented",
    "ALZ_ModerateDemented",
    "ALZ_NonDemented",
    "STROKE_Haem",
    "STROKE_Ischemic",
    "STROKE_Normal"
]

BEST_MODEL_PATH = "best_model (2).pth"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

# ---------------- PREDICT FUNCTION ----------------
def predict(img):
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = np.argmax(probs)
    label = classes[pred_idx]
    confidence = probs[pred_idx]

    # Disease mapping
    if label.startswith("ALZ"):
        disease = "Alzheimer's Disease"
        subtype = label.replace("ALZ_", "")
    else:
        disease = "Stroke"
        subtype = label.replace("STROKE_", "")

    return disease, subtype, label, confidence, probs


# ---------------- PAGE LAYOUT ----------------
st.set_page_config(page_title="MRI Disease Classifier", layout="wide")

# -------- Sidebar --------
# -------- Sidebar --------
with st.sidebar:
    st.markdown(
    "<h2 style='text-align:center;'>🧠 NeuroVision — Multi-Modal MRI Classifier</h2>",
    unsafe_allow_html=True
)

    st.header("ℹ️ About System")
    st.write("""
    
This system analyzes brain MRI scans using a  
**Multi-Modal Deep Learning model (EfficientNet-B3)** to identify:

• Alzheimer's stages  
• Stroke types


    """)

    st.markdown("---")
    st.header("💡 Health Information & Guidance")
    st.caption("Educational only — not a medical diagnosis.")

    # ---------------- ALZHEIMER'S ----------------
    with st.expander("🧠 Alzheimer's Disease"):
        st.subheader("🩺 What it is")
        st.write("""
        A progressive brain disorder that affects **memory, thinking,
        decision-making and behavior**.
        """)

        st.subheader("⚠️ Symptoms")
        st.markdown("""
        • Memory loss affecting daily activities  
        • Difficulty finding words or understanding speech  
        • Forgetting familiar places / tasks  
        • Mood swings, depression, confusion  
        • Repeating questions or stories  
        • Losing track of time / getting lost
        """)

        st.subheader("💊 Medications (doctor prescribed)")
        st.markdown("""
        Doctors may prescribe medicines to **slow symptoms**, such as:

        • Cholinesterase inhibitors  
          _(Donepezil, Rivastigmine, Galantamine)_  
        • NMDA blockers  
          _(Memantine)_

        > Always take only under medical supervision.
        """)

        st.subheader("🧩 Management & Support")
        st.markdown("""
        • Cognitive exercises (puzzles, reading)  
        • Regular walking / exercise  
        • Good sleep routine  
        • Omega-3 / Mediterranean-style diet  
        • Maintain social connections  
        • Safe home environment  
        • Caregiver support
        """)

        st.subheader("🛑 When to see a doctor")
        st.markdown("""
        • Rapid memory decline  
        • Confusion with names / people  
        • Sudden behavioral changes  
        • Safety concerns at home
        """)

    # ---------------- STROKE ----------------
    with st.expander("🩺 Stroke"):
        st.subheader("🩺 What it is")
        st.write("""
        A stroke happens when blood flow to the brain is blocked
        (ischemic) or bleeding occurs in the brain (hemorrhagic).
        """)

        st.subheader("🚨 Warning Signs (FAST)")
        st.markdown("""
        **F** — Face drooping  
        **A** — Arm weakness  
        **S** — Speech difficulty  
        **T** — Time to call emergency immediately
        """)

        st.subheader("💊 Hospital Treatment (doctor guided)")
        st.markdown("""
        Depends on stroke type:

        **Ischemic Stroke (clot):**
        • Clot-busting drugs  
        • Blood thinners  
        • Sometimes surgery

        **Hemorrhagic Stroke (bleeding):**
        • Blood-pressure control  
        • Surgery if required  
        • Monitoring in ICU

        > Urgent care is critical — treatment within hours saves brain cells.
        """)

        st.subheader("🔄 Recovery & Rehabilitation")
        st.markdown("""
        • Physiotherapy  
        • Speech therapy  
        • Occupational therapy  
        • Medication management
        """)

        st.subheader("🛡 Prevention & Lifestyle")
        st.markdown("""
        • Control blood pressure & sugar  
        • Avoid smoking & alcohol  
        • Exercise regularly  
        • Healthy diet — low salt & low fat  
        • Maintain healthy weight  
        • Stress management
        """)

        st.subheader("🛑 Seek emergency care if:")
        st.markdown("""
        • Sudden weakness / numbness  
        • Severe headache  
        • Loss of consciousness  
        • Slurred speech / confusion
        """)

   
# -------- Main UI --------
st.markdown('<h1 style="text-align:center;">🧠 Multi-Modal Deep Learning Framework for Early Detection and Classification of Neurological Disorders from MRI Scans</h1>', unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload MRI Image")
    uploaded = st.file_uploader("Choose MRI image", type=['jpg', 'jpeg', 'png'])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image",width="content")

with col2:
    st.header("🔍 Prediction Results")

    if uploaded:
        with st.spinner("Analyzing..."):
            disease, subtype, label, confidence, probs = predict(image)

        st.success(f"Detected Disease Type: **{disease}**")
        st.info(f"Subtype: **{subtype}**")
        st.write(f"Model Class Label: `{label}`")
        st.write(f"Confidence: **{confidence*100:.2f}%**")

        st.markdown("### 📊 Class Probabilities")
        prob_dict = {classes[i]: probs[i] for i in range(len(classes))}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        for cname, prob in sorted_probs:
            st.write(f"**{cname}** — {prob*100:.2f}%")
            st.progress(float(prob))

        st.markdown("---")
        st.success("✅ FINAL DECISION")
        st.write(f"**Disease:** {disease}")
        st.write(f"**Subtype:** {subtype}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

    else:
        st.info("👆 Please upload an MRI image to begin.")
