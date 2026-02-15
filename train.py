import os
import cv2
import numpy as np
import random
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ====================================================
# SETTINGS
# ====================================================
IMG_SIZE = 32   # 32 = faster, 64 = better accuracy (try later)
cat_path = "dataset/train/cat"
dog_path = "dataset/train/dog"

# ====================================================
# LOAD IMAGE FILE NAMES
# ====================================================
cat_images = os.listdir(cat_path)
dog_images = os.listdir(dog_path)

print("Original Cats:", len(cat_images))
print("Original Dogs:", len(dog_images))

# Balance dataset
min_count = min(len(cat_images), len(dog_images))
cat_images = random.sample(cat_images, min_count)
dog_images = random.sample(dog_images, min_count)

print("Using", min_count, "cats and dogs each")

data = []

# ====================================================
# FEATURE EXTRACTION FUNCTION
# ====================================================
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    return features

# ====================================================
# PROCESS CAT IMAGES
# ====================================================
for img_name in cat_images:
    img_path = os.path.join(cat_path, img_name)
    features = extract_features(img_path)

    if features is not None:
        data.append((features, 0))  # 0 = Cat

# ====================================================
# PROCESS DOG IMAGES
# ====================================================
for img_name in dog_images:
    img_path = os.path.join(dog_path, img_name)
    features = extract_features(img_path)

    if features is not None:
        data.append((features, 1))  # 1 = Dog

print("Total processed images:", len(data))

# ====================================================
# PREPARE DATA
# ====================================================
random.shuffle(data)

X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# ====================================================
# SPLIT DATA
# ====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ====================================================
# HYPERPARAMETER TUNING
# ====================================================
param_grid = {
    'C': [1, 10, 50],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(
    SVC(probability=True),
    param_grid,
    cv=3,
    verbose=1,
    n_jobs=-1
)

print("Running GridSearchCV...")
grid.fit(X_train, y_train)

model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# ====================================================
# EVALUATE MODEL
# ====================================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ====================================================
# CONFUSION MATRIX
# ====================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Cat", "Dog"],
            yticklabels=["Cat", "Dog"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")


# ====================================================
# SAVE MODEL
# ====================================================
joblib.dump(model, "svm_model.pkl")
print("\nModel saved as svm_model.pkl")
