import streamlit as st
import pandas as pd
import numpy as np

# Sidebar menu
menu = st.sidebar.radio("Pilih Menu", [
    "Collecting Data", 
    "Preprocessing", 
    "Exploratory Data Analysis", 
    "Modeling (ARIMAX)", 
    "Evaluation & Diagnostics"
])

# ====================================
# 1. Collecting Data
# ====================================
if menu == "Collecting Data":
    st.title("📥 Collecting Data")

    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # Coba baca dengan openpyxl
            import openpyxl
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        except ImportError:
            st.error("❌ Modul 'openpyxl' belum terinstall. Tambahkan 'openpyxl' di requirements.txt")
            df = None
        except Exception as e:
            st.error(f"❌ Terjadi error saat membaca file: {e}")
            df = None

        if df is not None:
            # Simpan ke session_state agar bisa dipakai di menu lain
            st.session_state['df'] = df

            # Tampilkan preview data
            st.subheader("Preview Data")
            st.dataframe(df.head())

            st.success("✅ Data berhasil diupload!")


# ====================================
# 2. Preprocessing
# ====================================
elif menu == "Preprocessing":
    st.title("⚙️ Preprocessing Data")
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Pastikan kolom yang dibutuhkan ada
        required_cols = ['date', 'bbca', 'usd', 'sgd']
        if all(col in df.columns for col in required_cols):
            df = df[required_cols]
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            st.session_state['df'] = df

            st.subheader("Data setelah preprocessing")
            st.dataframe(df.head())

            # Split data dengan iloc agar aman
            split_point = int(len(df) * 0.9)  # 90% train
            train = df.iloc[:split_point]
            test = df.iloc[split_point:]
            st.session_state['train'] = train
            st.session_state['test'] = test
            st.write("Train shape:", train.shape, " | Test shape:", test.shape)
        else:
            st.error(f"Kolom wajib {required_cols} tidak lengkap di file Excel.")
