import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
from scipy.stats import kstest

st.set_option('deprecation.showPyplotGlobalUse', False)

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
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state['df'] = df
        st.subheader("Preview Data")
        st.dataframe(df.head())
        st.success("Data berhasil diupload!")

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

# ====================================
# 3. EDA
# ====================================
elif menu == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Plot Original
        st.subheader("Original BBCA Data")
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['bbca'])
        plt.title("BBCA Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        st.pyplot()

        # ADF Test
        adf_result = adfuller(df['bbca'])
        st.write("ADF Statistic:", adf_result[0])
        st.write("p-value:", adf_result[1])
        st.write("Critical Values:", adf_result[4])
        if adf_result[1] <= 0.05:
            st.success("Data stasioner (tolak H0)")
        else:
            st.warning("Data belum stasioner (gagal tolak H0)")

        # Differencing
        bbca_diff = np.diff(df['bbca'], n=1)
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[1:], bbca_diff)
        plt.title("Differenced BBCA Data")
        st.pyplot()

        adf_result_diff = adfuller(bbca_diff)
        st.write("ADF (Differenced):", adf_result_diff[0], "| p-value:", adf_result_diff[1])

        # ACF & PACF
        st.subheader("ACF & PACF (Differenced Data)")
        fig, ax = plt.subplots(1,2,figsize=(14,6))
        plot_acf(bbca_diff, lags=20, ax=ax[0])
        plot_pacf(bbca_diff, lags=20, ax=ax[1], method="ywm")
        st.pyplot(fig)

# ====================================
# 4. Modeling
# ====================================
elif menu == "Modeling (ARIMAX)":
    st.title("📈 ARIMAX Modeling")
    if 'train' in st.session_state:
        train = st.session_state['train']
        test = st.session_state['test']

        train_y = train['bbca']
        test_y = test['bbca']
        x_train = train[['usd','sgd']]
        x_test = test[['usd','sgd']]

        # Fit ARIMAX
        model = SARIMAX(endog=train_y, exog=x_train, order=(0,1,1))
        results = model.fit(disp=False)
        st.session_state['results'] = results

        st.subheader("Model Summary")
        st.text(str(results.summary()))

        # Forecast
        forecast = results.get_forecast(steps=len(test_y), exog=x_test)
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()

        plt.figure(figsize=(14,6))
        plt.plot(train_y.index, train_y, label="Train")
        plt.plot(test_y.index, test_y, label="Test")
        plt.plot(test_y.index, forecast_values, label="Forecast", color="green")
        plt.fill_between(test_y.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color="lightgreen", alpha=0.5)
        plt.legend()
        plt.title("Forecast vs Actual")
        st.pyplot()

        st.session_state['forecast_values'] = forecast_values
        st.session_state['test_y'] = test_y

# ====================================
# 5. Evaluation & Diagnostics
# ====================================
elif menu == "Evaluation & Diagnostics":
    st.title("📉 Evaluation & Diagnostics")
    if 'forecast_values' in st.session_state:
        forecast_values = st.session_state['forecast_values']
        test_y = st.session_state['test_y']
        results = st.session_state['results']

        # Metrics
        mae = mean_absolute_error(test_y, forecast_values)
        rmse = np.sqrt(mean_squared_error(test_y, forecast_values))
        mape = mean_absolute_percentage_error(test_y, forecast_values)

        st.write(f"**MAE**: {mae:.4f}")
        st.write(f"**RMSE**: {rmse:.4f}")
        st.write(f"**MAPE**: {mape:.4f}")

        # Residual Diagnostics
        residuals = results.resid
        st.subheader("Residual Plot")
        plt.figure(figsize=(12,6))
        plt.plot(residuals)
        plt.title("Residuals")
        st.pyplot()

        st.subheader("Ljung-Box Test")
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        st.write(lb_test)

        st.subheader("Kolmogorov-Smirnov Test")
        ks_stat, ks_pval = kstest(test_y.values, 'norm', args=(test_y.mean(), test_y.std()))
        st.write("KS Statistic:", ks_stat, "| p-value:", ks_pval)

        st.subheader("White Test (Heteroskedastisitas)")
        white_test = het_white(residuals.values, sm.add_constant(results.fittedvalues.values))
        st.write("Statistic:", white_test[0], "| p-value:", white_test[1])
