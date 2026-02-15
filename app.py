import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image

# ===============================
# SETTINGS
# ===============================
IMG_SIZE = 32

# Load trained model
model = joblib.load("svm_model.pkl")

# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    features = hog(
        image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    return features.reshape(1, -1)

# ===============================
# UI
# ===============================
st.title("🐶🐱 Cats vs Dogs Classifier")
st.write("Upload an image and the model will predict whether it is a Cat or Dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    features = extract_features(image)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success(f"🐱 CAT")
        st.write(f"Confidence: {probabilities[0]*100:.2f}%")
    else:
        st.success(f"🐶 DOG")
        st.write(f"Confidence: {probabilities[1]*100:.2f}%")

st.markdown("---")
st.markdown("Built with using SVM + HOG + Streamlit")

st.markdown("---")
st.subheader("Model Performance")

st.write("Final Model Accuracy: ~71%")

st.image("confusion_matrix.png", caption="Confusion Matrix")

