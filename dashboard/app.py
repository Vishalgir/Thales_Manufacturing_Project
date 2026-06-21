import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Thales_Group_Manufacturing.csv")

    df = pd.read_csv(csv_path)

    return df

@st.cache_data
def load_data():
    df = pd.read_csv("Thales_Group_Manufacturing.csv")

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        dayfirst=True,
        errors="coerce"
    )
    
    df["Machine_Health_Index"] = (
        100
        - (df["Temperature_C"] / df["Temperature_C"].max()) * 30
        - (df["Vibration_Hz"] / df["Vibration_Hz"].max()) * 30
        - (df["Power_Consumption_kW"] / df["Power_Consumption_kW"].max()) * 40
    )
    
    df["Defect_Density_Score"] = (
        df["Quality_Control_Defect_Rate_%"] /
        df["Production_Speed_units_per_hr"]
    )
    
    df["Error_Frequency_Index"] = df["Error_Rate_%"]

    return df

df = load_data()

st.sidebar.header("🔍 Factory Filters")

machine_options = ["All"] + sorted(df["Machine_ID"].dropna().unique().tolist())
selected_machine = st.sidebar.selectbox(
    "Select Machine",
    machine_options,
    key="machine_filter"
)

mode_options = ["All"] + sorted(df["Operation_Mode"].dropna().unique().tolist())
selected_mode = st.sidebar.selectbox("Select Operation Mode", mode_options)

mode_options = ["All"] + sorted(df["Operation_Mode"].dropna().unique().tolist())
selected_mode = st.sidebar.selectbox(
    "Select Operation Mode",
    mode_options,
    key="operation_mode_filter"
)

start_date = st.sidebar.date_input(
    "Start Date",
    df["Date"].min().date()
)

end_date = st.sidebar.date_input(
    "End Date",
    df["Date"].max().date()
)

df_filtered = df.copy()

df_filtered = df_filtered[
    (df_filtered["Date"].dt.date >= start_date) &
    (df_filtered["Date"].dt.date <= end_date)
]

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

    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Records", len(df_filtered))
    col2.metric("Machine Health Index", round(df_filtered["Machine_Health_Index"].mean(), 2))
    col3.metric("Avg Production Speed", round(df_filtered["Production_Speed_units_per_hr"].mean(), 2))
    col4.metric("Defect Density", round(df_filtered["Defect_Density_Score"].mean(), 4))
    col5.metric("Error Frequency", round(df_filtered["Error_Frequency_Index"].mean(), 2))

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

    sample_df = df_filtered.sample(
        min(5000, len(df_filtered)),
        random_state=42
    )

    fig_defect = px.scatter(
        sample_df,
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

    st.markdown("### Underperforming Machines")
    
    bottom_machines = (
        df_filtered.groupby("Machine_ID")["Production_Speed_units_per_hr"]
        .mean()
        .sort_values()
        .head(10)
        .reset_index()
    )
    
    fig_bottom = px.bar(
        bottom_machines,
        x="Machine_ID",
        y="Production_Speed_units_per_hr",
        title="Bottom 10 Machines by Production Speed"
    )
    
    st.plotly_chart(fig_bottom, use_container_width=True)

    st.markdown("### Quality Bottleneck Machines")
    
    defect_machines = (
        df_filtered.groupby("Machine_ID")["Quality_Control_Defect_Rate_%"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    
    fig_defect_rank = px.bar(
        defect_machines,
        x="Machine_ID",
        y="Quality_Control_Defect_Rate_%",
        title="Top 10 Machines by Defect Rate"
    )
    
    st.plotly_chart(fig_defect_rank, use_container_width=True)
    
    st.dataframe(df_filtered.head(50))
