import streamlit as st
import pandas as pd
import joblib
import datetime

st.title("📦 Walmart Demand Forecast")

model = joblib.load("../models/model.pkl")

store = st.number_input("Store ID",1,45,1)

holiday = st.selectbox("Holiday Week",[0,1])

temperature = st.slider("Temperature",0.0,120.0,70.0)

fuel_price = st.slider("Fuel Price",1.0,5.0,3.0)

cpi = st.slider("CPI",100.0,250.0,200.0)

unemployment = st.slider("Unemployment",3.0,15.0,7.0)

date = st.date_input("Select Date")

if st.button("Predict Demand"):

    year = date.year
    month = date.month
    week = date.isocalendar()[1]
    dayofweek = date.weekday()

    input_data = pd.DataFrame({
        "Store":[store],
        "Holiday_Flag":[holiday],
        "Temperature":[temperature],
        "Fuel_Price":[fuel_price],
        "CPI":[cpi],
        "Unemployment":[unemployment],
        "year":[year],
        "month":[month],
        "week":[week],
        "dayofweek":[dayofweek]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Weekly Sales: ${prediction[0]:,.2f}")