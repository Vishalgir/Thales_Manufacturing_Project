import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Thales Smart Factory Dashboard",
    layout="wide"
)

st.title("🏭 6G-Enabled Smart Factory Dashboard")
st.markdown("Manufacturing Process Health and Operational Efficiency Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv("Thales_Group_Manufacturing.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    return df

df = load_data()

st.sidebar.header("🔍 Factory Filters")

machine_options = ["All"] + sorted(df["Machine_ID"].dropna().unique().tolist())
selected_machine = st.sidebar.selectbox("Select Machine", machine_options)

mode_options = ["All"] + sorted(df["Operation_Mode"].dropna().unique().tolist())
selected_mode = st.sidebar.selectbox("Select Operation Mode", mode_options)

df_filtered = df.copy()

if selected_machine != "All":
    df_filtered = df_filtered[df_filtered["Machine_ID"] == selected_machine]

if selected_mode != "All":
    df_filtered = df_filtered[df_filtered["Operation_Mode"] == selected_mode]

tab1, tab2, tab3, tab4 = st.tabs([
    "🏭 Factory Health Overview",
    "⚙️ Machine Health Dashboard",
    "📈 Production & Quality Panel",
    "📊 Efficiency Diagnostics"
])

with tab1:
    st.subheader("Factory Health Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df_filtered))
    col2.metric("Avg Temperature", round(df_filtered["Temperature_C"].mean(), 2))
    col3.metric("Avg Production Speed", round(df_filtered["Production_Speed_units_per_hr"].mean(), 2))
    col4.metric("Avg Defect Rate", round(df_filtered["Quality_Control_Defect_Rate_%"].mean(), 2))

    efficiency_counts = df_filtered["Efficiency_Status"].value_counts().reset_index()
    efficiency_counts.columns = ["Efficiency_Status", "Count"]

    fig_eff = px.pie(
        efficiency_counts,
        names="Efficiency_Status",
        values="Count",
        title="Efficiency Status Distribution",
        hole=0.4
    )
    st.plotly_chart(fig_eff, use_container_width=True)

with tab2:
    st.subheader("Machine Health Dashboard")

    machine_health = df_filtered.groupby("Machine_ID").agg({
        "Temperature_C": "mean",
        "Vibration_Hz": "mean",
        "Power_Consumption_kW": "mean",
        "Predictive_Maintenance_Score": "mean"
    }).reset_index()

    fig_temp = px.bar(
        machine_health,
        x="Machine_ID",
        y="Temperature_C",
        title="Average Temperature by Machine"
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    fig_vib = px.bar(
        machine_health,
        x="Machine_ID",
        y="Vibration_Hz",
        title="Average Vibration by Machine"
    )
    st.plotly_chart(fig_vib, use_container_width=True)

    st.dataframe(machine_health)

with tab3:
    st.subheader("Production & Quality Panel")

    col1, col2 = st.columns(2)

    with col1:
        fig_prod = px.box(
            df_filtered,
            x="Operation_Mode",
            y="Production_Speed_units_per_hr",
            color="Operation_Mode",
            title="Production Speed by Operation Mode"
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    with col2:
        fig_defect = px.scatter(
            df_filtered,
            x="Temperature_C",
            y="Quality_Control_Defect_Rate_%",
            color="Efficiency_Status",
            title="Temperature vs Defect Rate"
        )
        st.plotly_chart(fig_defect, use_container_width=True)

    fig_error = px.scatter(
        df_filtered,
        x="Vibration_Hz",
        y="Error_Rate_%",
        color="Efficiency_Status",
        title="Vibration vs Error Rate"
    )
    st.plotly_chart(fig_error, use_container_width=True)

with tab4:
    st.subheader("Efficiency Diagnostics")

    mode_eff = df_filtered.groupby(
        ["Operation_Mode", "Efficiency_Status"]
    ).size().reset_index(name="Count")

    fig_mode_eff = px.bar(
        mode_eff,
        x="Operation_Mode",
        y="Count",
        color="Efficiency_Status",
        title="Efficiency Status by Operation Mode",
        barmode="group"
    )
    st.plotly_chart(fig_mode_eff, use_container_width=True)

    machine_eff = df_filtered.groupby(
        ["Machine_ID", "Efficiency_Status"]
    ).size().reset_index(name="Count")

    fig_machine_eff = px.bar(
        machine_eff,
        x="Machine_ID",
        y="Count",
        color="Efficiency_Status",
        title="Machine-wise Efficiency Status",
        barmode="stack"
    )
    st.plotly_chart(fig_machine_eff, use_container_width=True)

    st.markdown("### Sample Filtered Data")
    st.dataframe(df_filtered.head(50))