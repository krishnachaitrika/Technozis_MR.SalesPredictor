
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="MR. Sales Predictor",
    layout="wide"
)

st.title("📊 MR. Sales Predictor")

# --------------------------------
# LOAD MODEL
# --------------------------------
model = joblib.load("models/model.pkl")

# --------------------------------
# LOAD DATA
# --------------------------------
df = pd.read_csv("data/Walmart.csv")

# --------------------------------
# TABS
# --------------------------------
tab1, tab2 = st.tabs(["Simulate", "Chart View"])

# =================================
# SIMULATION TAB
# =================================
with tab1:

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("Simulation Inputs")

        store = st.number_input("Store ID",1,45,1)

        holiday = st.selectbox(
            "Holiday Week",
            [0,1],
            format_func=lambda x: "Holiday Week" if x==1 else "Normal Week"
        )

        temperature = st.slider("Temperature",0.0,120.0,70.0)

        fuel_price = st.slider("Fuel Price",1.0,5.0,3.0)

        cpi = st.slider("CPI",100.0,250.0,200.0)

        date = st.date_input("Select Date")

        predict = st.button("Predict Sales")

    with col2:

        st.subheader("Prediction Dial")

        if predict:

            year = date.year
            month = date.month
            week = date.isocalendar()[1]
            dayofweek = date.weekday()

            unemployment = 7.0

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

            prediction = model.predict(input_data)[0]

            st.metric("Predicted Weekly Sales",
                      f"${prediction:,.0f}")

            # --------------------------------
            # GAUGE DIAL
            # --------------------------------
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Sales Prediction"},
                gauge={
                    'axis': {'range': [0, 2500000]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 800000], 'color': "lightgray"},
                        {'range': [800000, 1600000], 'color': "gray"},
                        {'range': [1600000, 2500000], 'color': "lightgreen"}
                    ],
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

# =================================
# CHART VIEW
# =================================
with tab2:

    st.subheader("📈 Store Sales Trend")

    store_graph = st.selectbox(
        "Select Store",
        df["Store"].unique()
    )

    store_data = df[df["Store"] == store_graph].copy()

    store_data["Date"] = pd.to_datetime(
        store_data["Date"],
        dayfirst=True
    )

    store_data = store_data.sort_values("Date")

    fig, ax = plt.subplots()

    ax.plot(
        store_data["Date"],
        store_data["Weekly_Sales"],
        label="Actual Sales",
        color="blue"
    )

    ax.set_title(f"Store {store_graph} Weekly Sales")

    ax.set_xlabel("Date")

    ax.set_ylabel("Weekly Sales")

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

