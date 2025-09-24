import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")
st.title("Brain Tumor Classification")
st.markdown("---")

# --- Sidebar ---
st.sidebar.title("About")
st.sidebar.info("Upload a brain MRI image to classify using pre-trained ML models.")
st.sidebar.markdown("**MODELS USED:**\n- KNN\n- SVM\n- Random Forest\n- Voting Classifier containing the above 3 models")


# --- Load Models and Scaler ---
try:
    knn_model = joblib.load("knn_model_final.pkl")
    svm_model = joblib.load("svm_model_final.pkl")
    rf_model = joblib.load("rf_model_final.pkl")
    voting_model = joblib.load("voting_model.pkl")
    scaler = joblib.load("scaler_final.pkl")
except Exception as e:
    st.error(f"Error loading models/scaler: {e}")
    st.stop()

models = {
    "KNN": knn_model,
    "SVM": svm_model,
    "Random Forest": rf_model,
    "Voting": voting_model
}

label_map = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}

# --- Load EfficientNetB0 ---
@st.cache_resource
def load_effnet():
    return EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

feature_model = load_effnet()

# --- Feature Extraction ---
def extract_deep(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x, verbose=0)
        return feat.flatten()
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# --- Tumor Info ---
tumor_info = {
    "glioma": "A tumor originating in glial cells of the brain or spine.",
    "meningioma": "A tumor arising from the meninges, membranes surrounding the brain and spinal cord.",
    "pituitary": "A tumor in the pituitary gland, located at the base of the brain.",
    "notumor": "The scan shows no indication of a brain tumor."
}

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Extracting features and predicting..."):
        features = extract_deep(temp_path, feature_model)
        if features is not None:
            scaled_features = scaler.transform(features.reshape(1, -1))
            predictions = {}
            confidences = {}

            # Individual model predictions
            for name, model in models.items():
                pred = model.predict(scaled_features)[0]
                label = label_map[pred]
                predictions[name] = label
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(scaled_features)[0]
                    confidences[name] = np.max(proba) * 100
                else:
                    confidences[name] = 100.0

            # Final prediction from Voting model
            final_pred = predictions["Voting"]
            avg_conf = confidences.get("Voting", np.mean(list(confidences.values())))

    # --- Two Columns Layout: Image + Final Prediction ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(temp_path, caption="MRI Scan", width=350)
    with col2:
        st.subheader("PREDICTION...")
        st.markdown(f"<h2 style='color: green;'>{final_pred.upper()}</h2>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {avg_conf:.2f}%")
        st.markdown("---")
        st.subheader("Tumor Information")
        st.markdown(tumor_info[final_pred])

    # --- Individual Model Predictions in Columns ---
    st.markdown("---")
    st.subheader("Individual Model Predictions")
    
    # Compute agreement
    agreement = {}
    for label in predictions.values():
        agreement[label] = agreement.get(label, 0) + 1

    # Display each model in its own column
    model_cols = st.columns(len(models))
    for i, (name, label) in enumerate(predictions.items()):
        with model_cols[i]:
            st.metric(label=name, value=label.upper(), delta=f"{confidences[name]:.2f}%")

    # Display class agreement
    st.markdown("**Class Agreement Across Models:**")
    for cls, count in agreement.items():
        st.write(f"{cls.upper()}: {count}/{len(models)} models agreed")

    # --- Downloadable Report ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""Brain Tumor Classification Report
Date: {timestamp}
Final Predicted Tumor: {final_pred.upper()}
Confidence: {avg_conf:.2f}%

Individual Predictions:
"""
    for name, label in predictions.items():
        report += f"{name}: {label.upper()} ({confidences[name]:.2f}%)\n"

    report += f"\nTumor Information:\n{tumor_info[final_pred]}"

    st.download_button("📄 Download Prediction Report", data=report,
                       file_name="tumor_prediction.txt", mime="text/plain")
