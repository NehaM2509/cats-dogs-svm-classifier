# 🐶🐱 Cats vs Dogs Image Classifier

An end-to-end Machine Learning project that classifies images of Cats and Dogs using classical computer vision techniques and deploys the model using Streamlit.

## 🚀 Live Demo

👉 https://cats-dogs-svm-classifier-7qvwu5s3fyd8lxwazfna5v.streamlit.app/

## 📌 Project Overview

This project implements a complete ML pipeline:

- Data preprocessing & balancing
- HOG (Histogram of Oriented Gradients) feature extraction
- Support Vector Machine (SVM) classifier
- Hyperparameter tuning using GridSearchCV
- Confusion Matrix evaluation
- Deployment using Streamlit

## 📊 Model Performance

- Balanced dataset (889 cats & 889 dogs)
- Optimized SVM with RBF kernel
- Achieved ~71% accuracy
- Evaluated using precision, recall, and F1-score

## 🛠 Tech Stack

- Python
- OpenCV
- Scikit-learn
- Scikit-image
- Streamlit
- Matplotlib
- Seaborn
- NumPy
- Joblib

## 📂 Project Structure

cats-dogs-svm-classifier/
│
├── app.py
├── train.py
├── predict.py
├── svm_model.pkl
├── confusion_matrix.png
├── requirements.txt
└── README.md

## 💻 Run Locally

```bash
pip install -r requirements.txt
python train.py
python -m streamlit run app.py


👩‍💻 Author
Neha Manashetty
Computer Science & Design Student
Aspiring ML Engineer


