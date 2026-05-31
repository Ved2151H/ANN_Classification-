<<<<<<< HEAD
# 🏦 Customer Churn Prediction Web App

An interactive, premium web application built with **Streamlit** and powered by an **Artificial Neural Network (ANN)** to predict bank customer churn with real-time feedback.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.11+](https://img.shields.shields.shields.shields.shields.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.21](https://img.shields.shields.shields.shields.shields.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.shields.shields.shields.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Features

*   **Real-time Deep Learning Inference**: Uses a pre-trained Keras ANN model (`model.h5`) to output churn probabilities immediately.
*   **Intuitive UI/UX**: Feature sliders, number inputs, and selection boxes carefully designed for inputting customer profiles.
*   **Dynamic Alert Feedbacks**: Displays precise prediction probabilities with color-coded success/warning panels.
*   **Production-Ready Data Pipeline**: Bundles production encoders (`LabelEncoder`, `OneHotEncoder`) and scalers (`StandardScaler`) to process raw input data exactly like the training phase.

---

## 🛠️ Technology Stack

*   **Core Logic & Model**: Python 3.11+, TensorFlow (Keras), Scikit-Learn, Pandas, NumPy
*   **Web Framework**: Streamlit
*   **Serialization**: Pickle (`.pkl`) for preprocessing encoders/scalers and HDF5 (`.h5`) for the neural network.

---

## 📂 Repository Structure

```text
├── Churn_Modelling.csv         # Raw dataset used for training the model
├── app.py                      # Main Streamlit web application entrypoint
├── experement.ipynb            # Jupyter Notebook containing exploratory data analysis & training steps
├── prediction.ipynb            # Jupyter Notebook used to verify pre-trained models
├── requirements.txt            # Project dependencies for local and cloud environments
├── model.h5                    # Pre-trained Keras Deep Learning model
├── scaler.pkl                  # Fitted StandardScaler object
├── label_encoder_gender.pkl    # Fitted LabelEncoder for gender features
├── onehot_encoder_geo.pkl      # Fitted OneHotEncoder for geographical features
└── README.md                   # Project documentation
```

---

## 🚀 Local Installation

Get the application running locally in just a few steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ann_classification-.git
   cd ann_classification-
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Streamlit Web Server**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser to view the application.

---

## ☁️ Streamlit Community Cloud Deployment

To host this model online on Streamlit Community Cloud:

1. **Push all files to GitHub** (make sure `model.h5` and `.pkl` files are committed).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in.
3. Click **"New app"** and fill in your repository details.
4. **Crucial Configuration**: Click on **"Advanced settings..."** at the bottom of the page.
5. In the dropdown menu, select **Python 3.11** or **Python 3.12** (since modern TensorFlow/Scikit-Learn requires Python 3.11+).
6. Click **Save** and **Deploy**!

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
=======
# ANN_Classification-

https://3pkxyregnaqczcurjnzgz3.streamlit.app/
>>>>>>> c88b3b281643edf98b694e03523440898425fa75
