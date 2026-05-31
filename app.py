import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import tensorflow as tf
import os

# Get base directory of the script to resolve files robustly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Loading the model
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'model.h5'))

# Load the encoders and scaler
with open(os.path.join(BASE_DIR, 'label_encoder_gender.pkl'), 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(os.path.join(BASE_DIR, 'onehot_encoder_geo.pkl'), 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 40)
balance = st.number_input('Balance', value=60000.0)
credit_score = st.number_input('Credit Score', value=600.0)
estimated_salary = st.number_input('Estimated Salary', value=50000.0)
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1], index=1)
is_active_member = st.selectbox('Is Active Member', [0, 1], index=1)

# Build input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography' with correct feature names to avoid sklearn UserWarning
geography_df = pd.DataFrame([[geography]], columns=['Geography'])
geo_encoded = onehot_encoder_geo.transform(geography_df).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Predict churn probability
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Display results
st.write(f'### Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.warning('⚠️ The customer is likely to exit (churn).')
else:
    st.success('✅ The customer is not likely to exit (churn).')