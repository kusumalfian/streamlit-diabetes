import pickle
import streamlit as st

# membaca model
diabetes_model = pickle.load(open('diabates_model.sav', 'rb'))

# judul website
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

# Tombol prediksi
if st.button('Test Prediksi Diabetes'):
    try:
        # Convert inputs to appropriate types
        Pregnancies = float(Pregnancies)
        Glucose = float(Glucose)
        BloodPressure = float(BloodPressure)
        SkinThickness = float(SkinThickness)
        Insulin = float(Insulin)
        BMI = float(BMI)
        DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
        Age = float(Age)

        # Make prediction
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'Pasien Terkena Penyakit Diabetes'
        else:
            diab_diagnosis = 'Pasien Tidak Terkena Penyakit Diabetes'

        st.success(diab_diagnosis)
    except ValueError:
        st.error('Masukkan nilai yang valid untuk semua input!')
