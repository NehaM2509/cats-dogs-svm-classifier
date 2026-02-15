import cv2
import numpy as np
import joblib
from skimage.feature import hog

IMG_SIZE = 64

# Load trained model
model = joblib.load("svm_model.pkl")

def predict_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error loading image")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    if prediction[0] == 0:
        print("Prediction: CAT 🐱")
    else:
        print("Prediction: DOG 🐶")


# ===== TEST WITH YOUR OWN IMAGE =====
image_path = "test.jpg"   # Put your test image name here
predict_image(image_path)
