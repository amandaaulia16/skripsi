import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Judul aplikasi
st.title("📊 Stock Data Analysis Dashboard")

# Sidebar menu
menu = st.sidebar.radio(
    "Pilih Menu",
    ["Collecting Data", "Preprocessing Data", "Exploratory Data Analysis", "Visualisasi Data"]
)

# Session state untuk simpan data agar bisa dipakai di semua menu
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

# =========================
# 1. Collecting Data
# =========================
if menu == "Collecting Data":
    st.header("📥 Collecting Data")

    ticker = st.text_input("Masukkan kode saham (contoh: BBCA.JK)", value="BBCA.JK")
    start_date = st.date_input("Tanggal Mulai", value=pd.to_datetime("2019-01-01"))
    end_date = st.date_input("Tanggal Akhir", value=pd.to_datetime("2024-09-30"))

    if st.button("Download Data"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if data.empty:
                st.error("⚠️ Download gagal, data kosong!")
            else:
                data.reset_index(inplace=True)

                st.session_state.stock_data = data  # simpan di session state

                st.success(f"✅ Data berhasil diambil: {ticker}")
                st.dataframe(data.head())

                # Simpan ke Excel
                output = BytesIO()
                data.to_excel(output, index=False, engine="openpyxl")
                excel_data = output.getvalue()

                st.download_button(
                    label="💾 Download Excel",
                    data=excel_data,
                    file_name=f"{ticker}_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"Terjadi error: {e}")
