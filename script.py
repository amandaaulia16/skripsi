import streamlit as st
import pandas as pd

# Judul aplikasi
st.title("📊 Stock Data Viewer")

# Upload file Excel
uploaded_file = st.file_uploader("📂 Upload file Excel", type=["xlsx"])

if uploaded_file is not None:
    # Baca file Excel
    df1 = pd.read_excel(uploaded_file)

    # Tampilkan data
    st.subheader("📌 Preview Data")
    st.dataframe(df1)

    # Info tambahan
    st.subheader("📊 Statistik Deskriptif")
    st.write(df1.describe())

    # Info kolom
    st.subheader("📌 Kolom yang tersedia")
    st.write(list(df1.columns))
else:
    st.info("Silakan upload file Excel untuk melihat data.")
