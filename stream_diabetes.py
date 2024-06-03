import pickle
import streamlit as st

# membaca model
diabetes_model = pickle.load(open('diabates_model.sav', 'rb'))

# judul website
st.title('Data Mining Prediksi Diabetes')

Pregnancies = st.text_input ('Input nilai Pragnancies')

Glucose = st.text_input ('Input nilai Glucose')

BloodPressure = st.text_input ('Input nilai Blood Pressure')

SkinTickness = st.text_input ('Input nilai Skin Tickness')

Insulin = st.text_input ('Input nilai Insulin')

BMI = st.text_input ('Input nilai BMI')

DiabetesPedigreeFunction = st.text_input ('Input nilai Diabetes Pedigree Function')

Age = st.text_input ('Input nilai Age')

# Code untuk Prediksi

diabetes_diagnosis = ''

# Tombol prediksi
if st.button('Test Prediksi Diabetes'):
    diabetes_diagnosis = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinTickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    if(diabetes_diagnosis[0] == 0):
        diabetes_diagnosis = 'Pasien Terkena Penyakit Diabetes'
    else:
        diabetes_diagnosis = 'Pasien Tidak Terkena Penyakit Diabetes'
    
    st.success(diabetes_diagnosis)
