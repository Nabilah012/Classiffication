import streamlit as st
import pandas as pd
import joblib

# Load model
clf_model = joblib.load('model_jantung.pkl')
clf_cols = joblib.load('columns.pkl')

reg_model = joblib.load('model_regresi_jantung.pkl')
reg_cols = joblib.load('columns_regresi.pkl')

st.title("Prediksi Penyakit Jantung")

menu = st.selectbox("Pilih Mode", ["Klasifikasi", "Regresi"])

age = st.number_input("Usia", 20, 100)
trestbps = st.number_input("Tekanan Darah", 80, 200)
thalach = st.number_input("Detak Jantung Maks", 60, 220)
oldpeak = st.number_input("Oldpeak", 0.0, 6.0)

if menu == "Klasifikasi":
    if st.button("Prediksi Klasifikasi"):
        data = pd.DataFrame([[age, trestbps, thalach, oldpeak]],
            columns=['age','trestbps','thalach','oldpeak'])
        data = pd.get_dummies(data)
        data = data.reindex(columns=clf_cols, fill_value=0)
        hasil = clf_model.predict(data)
        st.success("Sakit" if hasil[0]==1 else "Sehat")

if menu == "Regresi":
    if st.button("Prediksi Kolesterol"):
        data = pd.DataFrame([[age, trestbps, thalach, oldpeak]],
            columns=['age','trestbps','thalach','oldpeak'])
        data = pd.get_dummies(data)
        data = data.reindex(columns=reg_cols, fill_value=0)
        hasil = reg_model.predict(data)
        st.info(f"Perkiraan Kolesterol: {hasil[0]:.2f}")
