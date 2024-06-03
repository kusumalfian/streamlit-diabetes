import pickle
import streamlit as st
import numpy as np
import os

# Print the current working directory
st.write(f"Current working directory: {os.getcwd()}")

# Check if the files exist
model_exists = os.path.isfile('diabetes_model.sav')
scaler_exists = os.path.isfile('scaler.pkl')

if not model_exists:
    st.error("diabetes_model.sav not found")
if not scaler_exists:
    st.error("scaler.pkl not found")

if model_exists and scaler_exists:
    # Load the model and scaler
    diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    # Title of the web app
    st.title('Data Mining Prediksi Diabetes')

    # Input fields
    Pregnancies = st.text_input('Input nilai Pregnancies')
    Glucose = st.text_input('Input nilai Glucose')
    BloodPressure = st.text_input('Input nilai Blood Pressure')
    SkinThickness = st.text_input('Input nilai Skin Thickness')
    Insulin = st.text_input('Input nilai Insulin')
    BMI = st.text_input('Input nilai BMI')
    DiabetesPedigreeFunction = st.text_input('Input nilai Diabetes Pedigree Function')
    Age = st.text_input('Input nilai Age')

    # Code untuk Prediksi
    diab_diagnosis = ''

    # Prediction button
    if st.button('Test Prediksi Diabetes'):
        try:
            # Convert inputs to appropriate types
            input_data = np.array([[float(Pregnancies), float(Glucose), float(BloodPressure),
                                    float(SkinThickness), float(Insulin), float(BMI),
                                    float(DiabetesPedigreeFunction), float(Age)]])

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Make prediction
            diab_prediction = diabetes_model.predict(input_data_scaled)

            if diab_prediction[0] == 1:
                diab_diagnosis = 'Pasien Terkena Penyakit Diabetes'
            else:
                diab_diagnosis = 'Pasien Tidak Terkena Penyakit Diabetes'

            st.success(diab_diagnosis)
        except ValueError:
            st.error('Masukkan nilai yang valid untuk semua input!')
else:
    st.error("Required files are not found. Please upload the model and scaler files.")
