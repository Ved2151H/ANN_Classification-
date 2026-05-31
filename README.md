# 🏦 Customer Churn Prediction Web App

A clean, interactive web application that uses a Deep Learning **Artificial Neural Network (ANN)** to predict whether a bank customer is likely to leave (churn).

🔗 **Live Demo**: [https://3pkxyregnaqczcurjnzgz3.streamlit.app/](https://3pkxyregnaqczcurjnzgz3.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://3pkxyregnaqczcurjnzgz3.streamlit.app/)
[![Python 3.11+](https://img.shields.shields.shields.shields.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.21](https://img.shields.shields.shields.shields.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org)

---

## 🌟 Key Features

*   **ANN-Powered Predictions**: Uses a pre-trained Keras Deep Learning model (`model.h5`) under the hood.
*   **Production Preprocessing**: Automatically handles customer feature scaling (`StandardScaler`) and encoding (`LabelEncoder`, `OneHotEncoder`) in real time.
*   **Simple Input Fields**: Easily input details like credit score, geography, gender, age, balance, and activity status.
*   **Instant Result Panels**: Color-coded feedback showing the exact churn probability and prediction.

---

## 📁 Core Files

*   `app.py`: Streamlit frontend and prediction routing.
*   `model.h5`: Trained Keras ANN model.
*   `scaler.pkl`, `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`: Serialized data preprocessing objects.
*   `experement.ipynb`: Training notebook (exploratory analysis, scaling, network architecture, and callback configs).
*   `requirements.txt`: Project dependencies.

---

## 🚀 Running Locally

1. **Clone the project & navigate inside**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ann_classification-.git
   cd ann_classification-
   ```

2. **Create & activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\python.exe -m pip install -r requirements.txt
   # macOS/Linux:
   source venv/bin/activate && pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   # Windows:
   .\venv\Scripts\streamlit run app.py
   # macOS/Linux:
   streamlit run app.py
   ```

---

## ☁️ Deployment Note

When deploying this app to **Streamlit Community Cloud**, ensure you configure the **Advanced settings** to use **Python 3.11** or higher. This is required because modern TensorFlow versions are not supported on older Python versions (like Python 3.9).
