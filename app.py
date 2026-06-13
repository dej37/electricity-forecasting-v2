import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import numpy as np
import joblib

# ============================================================
# BRAND COLORS
# ============================================================
SPE_BLUE   = "#003087"
SPE_ORANGE = "#F5A800"
SPE_LIGHT  = "#E8EEF7"
SPE_GRAY   = "#6B7280"
SPE_WHITE  = "#FFFFFF"

MODEL_COLORS = {
    "Actual":        SPE_BLUE,
    "Random Forest": SPE_ORANGE,
    "LSTM":          "#00A86B",
    "SARIMA":        "#C0392B",
    "XGBoost":       "#8E44AD",
    "Holt-Winters":  "#2980B9",
    "Prophet":       "#E67E22",
}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SPE — Electricity Demand Forecasting",
    layout="wide"
)

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown(f"""
<style>
    .stApp {{ background-color: #F4F6FA; }}

    [data-testid="stSidebar"] {{ background-color: {SPE_BLUE}; }}
    [data-testid="stSidebar"] * {{ color: {SPE_WHITE} !important; }}
    [data-testid="stSidebar"] .stRadio label {{ font-size: 15px; }}

    h1 {{
        color: {SPE_BLUE} !important;
        border-bottom: 3px solid {SPE_ORANGE};
        padding-bottom: 8px;
    }}
    h2, h3 {{ color: {SPE_BLUE} !important; }}

    [data-testid="stMetric"] {{
        background-color: {SPE_WHITE};
        border: 1px solid #D1D9E6;
        border-left: 4px solid {SPE_ORANGE};
        border-radius: 8px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricLabel"] {{ color: {SPE_GRAY} !important; font-size: 13px !important; }}
    [data-testid="stMetricValue"] {{ color: {SPE_BLUE} !important; font-weight: 700 !important; }}

    .stButton > button {{
        background-color: {SPE_BLUE}; color: {SPE_WHITE};
        border: none; border-radius: 6px;
        padding: 8px 20px; font-weight: 600;
    }}
    .stButton > button:hover {{ background-color: {SPE_ORANGE}; color: {SPE_WHITE}; }}

    .stDownloadButton > button {{
        background-color: {SPE_ORANGE}; color: {SPE_WHITE};
        border: none; border-radius: 6px; font-weight: 600;
    }}
    .stDownloadButton > button:hover {{ background-color: {SPE_BLUE}; color: {SPE_WHITE}; }}

    .stTabs [data-baseweb="tab-list"] {{ background-color: {SPE_LIGHT}; border-radius: 8px; }}
    .stTabs [data-baseweb="tab"] {{ color: {SPE_BLUE}; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{
        background-color: {SPE_BLUE} !important;
        color: {SPE_WHITE} !important;
        border-radius: 6px;
    }}

    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stDateInput"] label,
    [data-testid="stSlider"] label {{
        color: {SPE_BLUE} !important; font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PLOTLY THEME HELPER
# ============================================================
def spe_layout(fig, title=None):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(color=SPE_BLUE, size=18, family="Arial Black"), x=0.5, xanchor="center"),
        font=dict(color=SPE_BLUE, family="Arial", size=14),
        paper_bgcolor=SPE_WHITE,
        plot_bgcolor="#FAFBFD",
        xaxis=dict(
            gridcolor="#E8EEF7", linecolor="#D1D9E6",
            tickfont=dict(color=SPE_BLUE, size=13, family="Arial"),
            title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"),
            tickangle=0,
            showgrid=True,
            zeroline=False,
            automargin=True
        ),
        yaxis=dict(
            gridcolor="#E8EEF7", linecolor="#D1D9E6",
            tickfont=dict(color=SPE_BLUE, size=13, family="Arial"),
            title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"),
            showgrid=True,
            zeroline=False,
            automargin=True
        ),
        legend=dict(
            bgcolor=SPE_WHITE,
            bordercolor="#D1D9E6",
            borderwidth=1,
            font=dict(color=SPE_BLUE, size=13, family="Arial")
        ),
        margin=dict(t=70, b=100, l=90, r=50),
        height=550
    )
    return fig


# ============================================================
# BASE PATH
# ============================================================
BASE = "."

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(rf"{BASE}/data/feature_engineered_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df

df = load_data()
target_col = "load"

# ============================================================
# MODEL FILES
# ============================================================
model_files = {
    "Holt-Winters":  rf"{BASE}/notebooks/hw_predictions.csv",
    "Prophet":       rf"{BASE}/notebooks/prophet_walkforward_predictions.csv",
    "Random Forest": rf"{BASE}/notebooks/rf_walkforward_predictions.csv",
    "XGBoost":       rf"{BASE}/notebooks/xgb_predictions.csv",
    "LSTM":          rf"{BASE}/notebooks/lstm_predictions.csv",
    "SARIMA":        rf"{BASE}/notebooks/sarima_walkforward_predictions.csv",
}

monthly_files = {
    "Holt-Winters":  rf"{BASE}/notebooks/hw_monthly_metrics.csv",
    "Prophet":       rf"{BASE}/notebooks/prophet_monthly_metrics.csv",
    "Random Forest": rf"{BASE}/notebooks/rf_monthly_metrics.csv",
    "XGBoost":       rf"{BASE}/notebooks/xgb_monthly_metrics.csv",
    "LSTM":          rf"{BASE}/notebooks/lstm_monthly_metrics.csv",
    "SARIMA":        rf"{BASE}/notebooks/sarima_monthly_metrics.csv",
}

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image(rf"{BASE}/asset/logo.jpg", width=140)
    st.markdown(f"""
    <div style='text-align:center; color:{SPE_ORANGE}; font-weight:700;
                font-size:13px; margin-bottom:20px; margin-top:-8px;'>
        Electricity Demand Forecasting
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "🗂  Data Management",
            "📊  Data Visualization",
            "🤖  Model Results",
            "🔮  Future Forecasting"
        ]
    )

page = page.split("  ")[-1]

# ============================================================
# PAGE 1 — HOME
# ============================================================
if page == "Home":
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image(rf"{BASE}/asset/logo.jpg", width=110)
    with col_title:
        st.title("Electricity Demand Forecasting Dashboard")
        st.markdown(
            f"<span style='color:{SPE_GRAY}; font-size:15px;'>"
            "Sonelgaz — Production de l'Électricité (S-PE) &nbsp;|&nbsp; "
            "Réseau Interconnecté National (RIN)</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown(f"""
    <div style='background:{SPE_WHITE}; border-radius:10px; padding:20px 24px;
                border-left:5px solid {SPE_ORANGE}; margin-bottom:20px;'>
        <p style='color:{SPE_GRAY}; margin:0; line-height:1.8;'>
        This dashboard supports the analysis and short-term forecasting of hourly electricity
        demand on Algeria's national interconnected grid. It covers
        <b>January 2023 – December 2025</b> (~26,300 hourly observations) and compares
        six forecasting models evaluated via walk-forward validation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Start date",         str(df.index.min().date()))
    c2.metric("End date",           str(df.index.max().date()))
    c3.metric("Total observations", f"{len(df):,}")
    c4.metric("Best model (MAPE)",  "Random Forest")

    st.markdown("---")
    st.subheader("What's inside")

    col1, col2, col3, col4 = st.columns(4)
    for col, icon, title, desc in [
        (col1, "🗂", "Data Management",    "Filter, inspect, and download the dataset"),
        (col2, "📊", "Data Visualization", "Explore demand patterns and seasonality"),
        (col3, "🤖", "Model Results",      "Compare model metrics and forecast quality"),
        (col4, "🔮", "Future Forecasting", "Generate forecasts with RF and LSTM"),
    ]:
        col.markdown(f"""
        <div style='background:{SPE_WHITE}; border-radius:10px; padding:16px;
                    border-top:4px solid {SPE_BLUE}; text-align:center;'>
            <div style='font-size:28px;'>{icon}</div>
            <div style='color:{SPE_BLUE}; font-weight:700; margin:6px 0 4px;'>{title}</div>
            <div style='color:{SPE_GRAY}; font-size:13px;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 2 — DATA MANAGEMENT
# ============================================================
elif page == "Data Management":
    st.title("Data Management")

    st.subheader("Filter dataset by period")
    start_date = st.date_input("Start date", df.index.min().date())
    end_date   = st.date_input("End date",   df.index.max().date())

    filtered_df = df.loc[str(start_date):str(end_date)]
    st.write(f"Filtered observations: **{len(filtered_df):,}**")
    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("Missing values")
    missing_df = filtered_df.isna().sum().reset_index()
    missing_df.columns = ["Column", "Missing values"]
    st.dataframe(missing_df, use_container_width=True)

    st.download_button(
        label="⬇ Download filtered data",
        data=filtered_df.to_csv().encode("utf-8"),
        file_name="filtered_demand_data.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 3 — DATA VISUALIZATION
# ============================================================
elif page == "Data Visualization":
    st.title("Data Visualization")

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    st.subheader("Select visualization period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", df.index.min().date(), key="viz_start_date")
    with col2:
        end_date = st.date_input("End date", df.index.max().date(), key="viz_end_date")

    filtered_df = df.loc[str(start_date):str(end_date)].copy()

    if filtered_df.empty:
        st.warning("No data available for the selected period.")
        st.stop()

    # Hourly time series
    st.subheader("Hourly electricity demand")
    fig = go.Figure()
    fig.add_scatter(x=filtered_df.index, y=filtered_df[target_col],
                    mode="lines", line=dict(color=SPE_BLUE, width=1), name="Demand")
    spe_layout(fig, "Hourly Electricity Demand")
    fig.update_xaxes(title_text="Date", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)

    fig.update_yaxes(title_text="Demand (MW)", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)
    st.plotly_chart(fig, use_container_width=True)

    # Aggregated demand
    st.subheader("Aggregated demand")
    agg_choice = st.selectbox("Aggregation level", ["Hourly","Daily","Weekly","Monthly"], key="agg_choice")
    freq_map = {"Hourly":"h","Daily":"D","Weekly":"W","Monthly":"ME"}
    agg_series = filtered_df[target_col].resample(freq_map[agg_choice]).mean()
    fig_agg = go.Figure()
    fig_agg.add_scatter(x=agg_series.index, y=agg_series.values,
                        mode="lines", line=dict(color=SPE_ORANGE, width=2), name=agg_choice)
    spe_layout(fig_agg, f"{agg_choice} Average Electricity Demand")
    fig_agg.update_xaxes(title_text="Date", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)

    fig_agg.update_yaxes(title_text="Demand (MW)", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)
    st.plotly_chart(fig_agg, use_container_width=True)

    # Box by day of week
    st.subheader("Demand distribution by day of week")
    temp = filtered_df.copy()
    temp["day_name"] = temp.index.day_name()
    fig_box = px.box(temp, x="day_name", y=target_col,
                     category_orders={"day_name": day_order},
                     color_discrete_sequence=[SPE_BLUE])
    spe_layout(fig_box, "Electricity Demand Distribution by Day of Week")
    fig_box.update_xaxes(title_text="Day of Week", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)

    fig_box.update_yaxes(title_text="Demand (MW)", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)
    st.plotly_chart(fig_box, use_container_width=True)

    # Hourly profile
    st.subheader("Average demand by hour of day")
    hourly_profile = filtered_df.groupby(filtered_df.index.hour)[target_col].mean().reset_index()
    hourly_profile.columns = ["Hour", "Average Demand"]
    hourly_profile["Hour"] = hourly_profile["Hour"].astype(str) + ":00"
    fig_hour = go.Figure()
    fig_hour.add_scatter(x=hourly_profile["Hour"], y=hourly_profile["Average Demand"],
                         mode="lines+markers",
                         line=dict(color=SPE_BLUE, width=2),
                         marker=dict(color=SPE_ORANGE, size=7))
    spe_layout(fig_hour, "Average Electricity Demand by Hour of Day")
    fig_hour.update_xaxes(title_text="Hour of Day", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)

    fig_hour.update_yaxes(title_text="Demand (MW)", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)
    st.plotly_chart(fig_hour, use_container_width=True)

    # Monthly profile
    st.subheader("Average demand by month")
    monthly_profile = filtered_df.groupby(filtered_df.index.month)[target_col].mean().reset_index()
    monthly_profile.columns = ["Month", "Average Demand"]
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    monthly_profile["Month_Name"] = monthly_profile["Month"].map(month_names)
    fig_month = go.Figure()
    fig_month.add_bar(x=monthly_profile["Month_Name"], y=monthly_profile["Average Demand"],
                      marker_color=SPE_BLUE)
    spe_layout(fig_month, "Average Electricity Demand by Month")
    fig_month.update_xaxes(title_text="Month", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)

    fig_month.update_yaxes(title_text="Demand (MW)", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)
    st.plotly_chart(fig_month, use_container_width=True)

    # STL decomposition
    st.subheader("STL decomposition")
    st.markdown("<p style='color:#333333; font-size:14px; margin-bottom:10px;'>Decomposes hourly demand into trend, seasonal, and residual components (period=24h).</p>", unsafe_allow_html=True)
    stl_series = filtered_df[target_col].dropna().asfreq("h").interpolate()
    if len(stl_series) > 48:
        stl = STL(stl_series, period=24, robust=True)
        result = stl.fit()
        stl_df = pd.DataFrame({
            "Observed": result.observed, "Trend": result.trend,
            "Seasonal": result.seasonal, "Residual": result.resid
        })
        stl_colors = [SPE_BLUE, SPE_ORANGE, "#00A86B", SPE_GRAY]
        fig_stl = go.Figure()
        for col, color in zip(stl_df.columns, stl_colors):
            fig_stl.add_scatter(x=stl_df.index, y=stl_df[col],
                                mode="lines", name=col, line=dict(color=color, width=1))
        spe_layout(fig_stl, "STL Decomposition of Electricity Demand")
        fig_stl.update_layout(
            xaxis_title=dict(text="Date", font=dict(color=SPE_BLUE, size=15, family="Arial Black")),
            yaxis_title=dict(text="MW", font=dict(color=SPE_BLUE, size=15, family="Arial Black"))
        )
        st.plotly_chart(fig_stl, use_container_width=True)
    else:
        st.warning("Selected period is too short for STL decomposition.")

    # ACF / PACF
    st.subheader("ACF and PACF analysis")
    st.markdown("<p style='color:#333333; font-size:14px; margin-bottom:10px;'>Computed after seasonal differencing at lag 24, consistent with SARIMA identification.</p>", unsafe_allow_html=True)
    acf_series = filtered_df[target_col].dropna().asfreq("h").interpolate().diff(24).dropna()
    lags = st.slider("Number of lags", 10, 168, 72, key="acf_lags")
    if len(acf_series) > lags + 10:
        fig_acf, ax = plt.subplots(figsize=(12, 3))
        ax.set_facecolor("#FAFBFD")
        fig_acf.patch.set_facecolor(SPE_WHITE)
        plot_acf(acf_series, lags=lags, ax=ax, color=SPE_BLUE)
        ax.set_title("ACF after seasonal differencing at lag 24", color=SPE_BLUE, fontsize=14, fontweight='bold')
        ax.set_xlabel("Lag", color=SPE_BLUE, fontsize=12, fontweight='bold')
        ax.set_ylabel("Autocorrelation", color=SPE_BLUE, fontsize=12, fontweight='bold')
        ax.tick_params(colors=SPE_BLUE, labelsize=11)
        st.pyplot(fig_acf)
        plt.close()

        fig_pacf, ax = plt.subplots(figsize=(12, 3))
        ax.set_facecolor("#FAFBFD")
        fig_pacf.patch.set_facecolor(SPE_WHITE)
        plot_pacf(acf_series, lags=lags, ax=ax, method="ywm", color=SPE_BLUE)
        ax.set_title("PACF after seasonal differencing at lag 24", color=SPE_BLUE, fontsize=14, fontweight='bold')
        ax.set_xlabel("Lag", color=SPE_BLUE, fontsize=12, fontweight='bold')
        ax.set_ylabel("Partial Autocorrelation", color=SPE_BLUE, fontsize=12, fontweight='bold')
        ax.tick_params(colors=SPE_BLUE, labelsize=11)
        st.pyplot(fig_pacf)
        plt.close()
    else:
        st.warning("Selected period is too short for ACF/PACF analysis.")

    # Heatmap
    st.subheader("Demand heatmap by hour and day of week")
    heatmap_df = filtered_df.copy()
    heatmap_df["hour"]     = heatmap_df.index.hour
    heatmap_df["day_name"] = heatmap_df.index.day_name()
    heatmap_data = heatmap_df.groupby(["day_name","hour"])[target_col].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index="day_name", columns="hour", values=target_col).reindex(day_order)
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg Demand (MW)"),
        color_continuous_scale=[[0, SPE_LIGHT],[0.5, SPE_BLUE],[1, SPE_ORANGE]],
        aspect="auto"
    )
    spe_layout(fig_heatmap, "Average Electricity Demand Heatmap")
    fig_heatmap.update_layout(
        xaxis=dict(title=dict(text="Hour of Day", font=dict(color=SPE_BLUE, size=15, family="Arial Black")), automargin=True),
        yaxis=dict(title=dict(text="Day of Week", font=dict(color=SPE_BLUE, size=15, family="Arial Black")), automargin=True),
        coloraxis_colorbar=dict(title=dict(text="Avg Demand (MW)", font=dict(color="#000000", size=14, family="Arial Black")))
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Correlation heatmap
    st.subheader("Feature correlation heatmap")
    corr_cols = [
        target_col, "hour","day_of_week","month","year",
        "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
        "lag_1","lag_24","lag_168","rolling_mean_24","rolling_std_24",
        "rolling_mean_168","rolling_std_168","Post_Break"
    ]
    corr_cols = [c for c in corr_cols if c in filtered_df.columns]
    if len(corr_cols) >= 2:
        corr_matrix = filtered_df[corr_cols].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale=[[0, SPE_ORANGE],[0.5, SPE_WHITE],[1, SPE_BLUE]]
        )
        spe_layout(fig_corr, "Correlation Heatmap of Demand and Engineered Features")
        fig_corr.update_xaxes(
            title_text="Features",
            title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"),
            tickangle=45,
            tickfont=dict(size=9, color=SPE_BLUE),
            automargin=True
        )
        fig_corr.update_yaxes(
            title_text="Features",
            title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"),
            tickfont=dict(size=9, color=SPE_BLUE),
            automargin=True
        )
        fig_corr.update_layout(
            coloraxis_colorbar=dict(title=dict(text="Correlation", font=dict(color="#000000", size=14, family="Arial Black")))
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Not enough numeric feature columns available.")


# ============================================================
# PAGE 4 — MODEL RESULTS
# ============================================================
elif page == "Model Results":
    st.title("Model Results Comparison")
    st.write("Forecasting performance over the July–December 2025 test period.")

    # --------------------------------------------------------
    # Overall metrics table
    # --------------------------------------------------------
    results = []
    for model_name, file_path in model_files.items():
        pred_df   = pd.read_csv(file_path, index_col=0, parse_dates=True)
        actual    = pred_df["actual"]
        predicted = pred_df["predicted"]
        mae  = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        results.append({"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape})

    results_df = pd.DataFrame(results).sort_values("MAPE (%)")

    st.subheader("Overall forecasting metrics")

    def highlight_best(s):
        is_best = s == s.min()
        return [
            f"background-color: {SPE_ORANGE}; color: white; font-weight:700" if v else ""
            for v in is_best
        ]

    st.dataframe(
        results_df.style
            .apply(highlight_best, subset=["MAE","RMSE","MAPE (%)"])
            .format({"MAE":"{:.2f}","RMSE":"{:.2f}","MAPE (%)":"{:.2f}"}),
        use_container_width=True
    )

    # --------------------------------------------------------
    # Bar chart
    # --------------------------------------------------------
    st.subheader("Overall metric comparison")
    metric_choice = st.selectbox("Select metric", ["MAE","RMSE","MAPE (%)"], key="overall_metric_choice")

    fig_metric = go.Figure()
    for _, row in results_df.iterrows():
        fig_metric.add_bar(
            x=[row["Model"]], y=[row[metric_choice]],
            name=row["Model"],
            marker_color=MODEL_COLORS.get(row["Model"], SPE_GRAY),
            text=[f"{row[metric_choice]:.2f}"],
            textposition="outside"
        )
    spe_layout(fig_metric, f"Overall {metric_choice} by Model")
    fig_metric.update_xaxes(title_text="Model", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)

    fig_metric.update_yaxes(title_text=metric_choice, title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)

    fig_metric.update_layout(showlegend=False)
    st.plotly_chart(fig_metric, use_container_width=True)

    # --------------------------------------------------------
    # Monthly metrics
    # --------------------------------------------------------
    st.subheader("Monthly metrics comparison")
    monthly_results = []
    for model_name, file_path in monthly_files.items():
        monthly_df = pd.read_csv(file_path)
        monthly_df.columns = [c.strip() for c in monthly_df.columns]
        monthly_df["Model"] = model_name
        monthly_results.append(monthly_df)

    monthly_all_df = pd.concat(monthly_results, ignore_index=True)
    monthly_metric_choice = st.selectbox("Select monthly metric", ["MAE","RMSE","MAPE"], key="monthly_metric_choice")
    monthly_pivot = monthly_all_df.pivot(index="origin_month", columns="Model", values=monthly_metric_choice)

    st.dataframe(monthly_pivot.style.format("{:.2f}"), use_container_width=True)

    fig_monthly = go.Figure()
    for model in monthly_pivot.columns:
        fig_monthly.add_scatter(
            x=monthly_pivot.index, y=monthly_pivot[model],
            mode="lines+markers", name=model,
            line=dict(color=MODEL_COLORS.get(model, SPE_GRAY), width=2),
            marker=dict(size=7)
        )
    spe_layout(fig_monthly, f"Monthly {monthly_metric_choice} Comparison by Model")
    fig_monthly.update_xaxes(title_text="Origin Month", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)

    fig_monthly.update_yaxes(title_text=monthly_metric_choice, title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)
    st.plotly_chart(fig_monthly, use_container_width=True)

    # --------------------------------------------------------
    # Actual vs Predicted
    # --------------------------------------------------------
    st.subheader("Actual vs Predicted")
    avp_model = st.selectbox("Select model", list(model_files.keys()), key="avp_model")
    avp_df = pd.read_csv(model_files[avp_model], index_col=0, parse_dates=True)
    avp_df.index.name = "timestamp"

    col1, col2 = st.columns(2)
    with col1:
        avp_start = st.date_input("Start date", avp_df.index.min().date(), key="avp_start")
    with col2:
        avp_end = st.date_input("End date", avp_df.index.max().date(), key="avp_end")

    avp_filtered = avp_df.loc[str(avp_start):str(avp_end)]

    if avp_filtered.empty:
        st.warning("No data for selected period.")
    else:
        fig_avp = go.Figure()
        fig_avp.add_scatter(x=avp_filtered.index, y=avp_filtered["actual"],
                            mode="lines", name="Actual",
                            line=dict(color=SPE_BLUE, width=1.5))
        fig_avp.add_scatter(x=avp_filtered.index, y=avp_filtered["predicted"],
                            mode="lines", name="Predicted",
                            line=dict(color=SPE_ORANGE, width=1.5, dash="dot"))
        spe_layout(fig_avp, f"Actual vs Predicted — {avp_model}")
        fig_avp.update_layout(
            xaxis_title=dict(text="Date", font=dict(color=SPE_BLUE, size=15, family="Arial Black")),
            yaxis_title=dict(text="Demand (MW)", font=dict(color=SPE_BLUE, size=15, family="Arial Black"))
        )
        st.plotly_chart(fig_avp, use_container_width=True)

        mae_f  = np.mean(np.abs(avp_filtered["actual"] - avp_filtered["predicted"]))
        rmse_f = np.sqrt(np.mean((avp_filtered["actual"] - avp_filtered["predicted"]) ** 2))
        mape_f = np.mean(np.abs((avp_filtered["actual"] - avp_filtered["predicted"]) / avp_filtered["actual"])) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",  f"{mae_f:.2f}")
        c2.metric("RMSE", f"{rmse_f:.2f}")
        c3.metric("MAPE", f"{mape_f:.2f}%")

    # --------------------------------------------------------
    # Diebold-Mariano
    # --------------------------------------------------------
    st.subheader("Diebold-Mariano Test Results (RF as reference)")

    dm_mse = pd.DataFrame([
        {"Model B": "SARIMA",  "Statistic": -4.2500,  "p-value": 0.0000},
        {"Model B": "XGBoost", "Statistic": -2.4058,  "p-value": 0.0161},
        {"Model B": "LSTM",    "Statistic":  0.4724,  "p-value": 0.6367},
        {"Model B": "HW",      "Statistic": -11.3904, "p-value": 0.0000},
        {"Model B": "Prophet", "Statistic": -26.0091, "p-value": 0.0000},
    ])

    dm_mae = pd.DataFrame([
        {"Model B": "SARIMA",  "Statistic": -1.6137,  "p-value": 0.1066},
        {"Model B": "XGBoost", "Statistic": -6.2164,  "p-value": 0.0000},
        {"Model B": "LSTM",    "Statistic": -1.4587,  "p-value": 0.1447},
        {"Model B": "HW",      "Statistic": -13.9394, "p-value": 0.0000},
        {"Model B": "Prophet", "Statistic": -34.3990, "p-value": 0.0000},
    ])

    def style_dm(df):
        df = df.copy()
        df["Significant (α=0.05)"] = df["p-value"].apply(
            lambda p: "✅ Yes" if p < 0.05 else "❌ No"
        )
        return df

    def dm_styler(df):
        def row_style(row):
            if row["Significant (α=0.05)"] == "✅ Yes":
                color = f"background-color: {SPE_ORANGE}; color: white"
            else:
                color = f"background-color: {SPE_LIGHT}; color: #000000"
            return [color] * len(row)
        return df.style.apply(row_style, axis=1).format(
            {"Statistic": "{:.4f}", "p-value": "{:.4f}"}
        )

    tab_mse, tab_mae = st.tabs(["MSE-based", "MAE-based"])
    with tab_mse:
        st.dataframe(dm_styler(style_dm(dm_mse)), use_container_width=True)
        st.caption("Negative statistic → RF significantly more accurate than the compared model.")
    with tab_mae:
        st.dataframe(dm_styler(style_dm(dm_mae)), use_container_width=True)
        st.caption("Negative statistic → RF significantly more accurate than the compared model.")

    # --------------------------------------------------------
    # Residuals
    # --------------------------------------------------------
    st.subheader("Residuals Analysis")
    res_model = st.selectbox("Select model", list(model_files.keys()), key="res_model")
    res_df = pd.read_csv(model_files[res_model], index_col=0, parse_dates=True)
    res_df["residual"] = res_df["actual"] - res_df["predicted"]

    fig_res = go.Figure()
    fig_res.add_scatter(x=res_df.index, y=res_df["residual"],
                        mode="lines", line=dict(color=SPE_BLUE, width=1), name="Residual")
    fig_res.add_hline(y=0, line_dash="dash", line_color=SPE_ORANGE)
    spe_layout(fig_res, f"Residuals over Time — {res_model}")
    fig_res.update_xaxes(title_text="Date", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)

    fig_res.update_yaxes(title_text="Residuals (MW)", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)
    st.plotly_chart(fig_res, use_container_width=True)

    fig_hist = go.Figure()
    fig_hist.add_histogram(x=res_df["residual"], nbinsx=60,
                           marker_color=SPE_BLUE, opacity=0.8)
    fig_hist.add_vline(x=0, line_dash="dash", line_color=SPE_ORANGE)
    spe_layout(fig_hist, f"Residuals Distribution — {res_model}")
    fig_hist.update_xaxes(title_text="Residuals (MW)", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)

    fig_hist.update_yaxes(title_text="Count", title_font=dict(color=SPE_BLUE, size=15, family="Arial Black"), tickfont=dict(color=SPE_BLUE, size=13, family="Arial"), automargin=True)
    st.plotly_chart(fig_hist, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean residual",     f"{res_df['residual'].mean():.2f}")
    c2.metric("Std residual",      f"{res_df['residual'].std():.2f}")
    c3.metric("Max overestimate",  f"{res_df['residual'].min():.2f}")
    c4.metric("Max underestimate", f"{res_df['residual'].max():.2f}")


# ============================================================
# PAGE 5 — FUTURE FORECASTING
# ============================================================
elif page == "Future Forecasting":
    import torch
    import torch.nn as nn

    st.title("Future Forecasting")
    st.write(
        "Generate future electricity demand forecasts using the two best models "
        "trained on the full dataset (January 2023 – December 2025)."
    )

    class LSTMForecaster(nn.Module):
        def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, output_size=744):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    @st.cache_resource
    def load_models():
        rf_model    = joblib.load(rf"{BASE}/notebooks/models/rf_final.pkl")
        rf_features = joblib.load(rf"{BASE}/notebooks/models/rf_feature_cols.pkl")

        checkpoint  = torch.load(rf"{BASE}/notebooks/models/lstm_final.pth", map_location="cpu")
        lstm_model  = LSTMForecaster(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            output_size=checkpoint["horizon"]
        )
        lstm_model.load_state_dict(checkpoint["model_state_dict"])
        lstm_model.eval()
        scaler_x      = joblib.load(rf"{BASE}/notebooks/models/lstm_scaler_x.pkl")
        scaler_y      = joblib.load(rf"{BASE}/notebooks/models/lstm_scaler_y.pkl")
        lstm_features = checkpoint["feature_cols"]

        return rf_model, rf_features, lstm_model, scaler_x, scaler_y, lstm_features

    rf_model, rf_features, lstm_model, scaler_x, scaler_y, lstm_features = load_models()

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        selected_models = st.multiselect(
            "Select models",
            ["Random Forest", "LSTM"],
            default=["Random Forest"]
        )
    with col2:
        horizon_label = st.selectbox(
            "Forecast horizon",
            ["1 hour", "24 hours (1 day)", "168 hours (1 week)", "744 hours (1 month)"]
        )

    horizon_map = {
        "1 hour": 1,
        "24 hours (1 day)": 24,
        "168 hours (1 week)": 168,
        "744 hours (1 month)": 744
    }
    H = horizon_map[horizon_label]

    if not selected_models:
        st.warning("Please select at least one model.")
        st.stop()

    def make_future_timestamps(H):
        last_ts = df.index[-1]
        return pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=H, freq="h")

    def build_rf_features(future_index):
        ft = pd.DataFrame(index=future_index)
        ft["hour"]        = ft.index.hour
        ft["day_of_week"] = ft.index.dayofweek
        ft["month"]       = ft.index.month
        ft["year"]        = ft.index.year
        ft["hour_sin"]    = np.sin(2 * np.pi * ft["hour"] / 24)
        ft["hour_cos"]    = np.cos(2 * np.pi * ft["hour"] / 24)
        ft["dow_sin"]     = np.sin(2 * np.pi * ft["day_of_week"] / 7)
        ft["dow_cos"]     = np.cos(2 * np.pi * ft["day_of_week"] / 7)
        ft["month_sin"]   = np.sin(2 * np.pi * ft["month"] / 12)
        ft["month_cos"]   = np.cos(2 * np.pi * ft["month"] / 12)
        return ft[rf_features]

    def build_lstm_sequence():
        SEQ_LEN = 744
        tail = df.dropna(subset=lstm_features).tail(SEQ_LEN)
        if len(tail) < SEQ_LEN:
            st.error(f"Not enough data for LSTM sequence. Need {SEQ_LEN}, got {len(tail)}.")
            st.stop()
        X = scaler_x.transform(tail[lstm_features].values)
        return torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    if st.button("Generate Forecast"):
        future_index = make_future_timestamps(H)
        forecasts = {}

        with st.spinner("Generating forecasts..."):
            if "Random Forest" in selected_models:
                X_future = build_rf_features(future_index)
                forecasts["Random Forest"] = rf_model.predict(X_future)

            if "LSTM" in selected_models:
                X_tensor = build_lstm_sequence()
                with torch.no_grad():
                    y_scaled = lstm_model(X_tensor).squeeze().numpy()
                y_pred = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
                forecasts["LSTM"] = y_pred[:H]

        forecast_df = pd.DataFrame(forecasts, index=future_index)

        context_hours = min(168, len(df))
        actuals_tail  = df[target_col].tail(context_hours)

        fig_fc = go.Figure()
        fig_fc.add_scatter(
            x=actuals_tail.index, y=actuals_tail.values,
            mode="lines", name="Actual ",
            line=dict(color=SPE_BLUE, width=2)
        )
        for model_name, preds in forecasts.items():
            fig_fc.add_scatter(
                x=future_index, y=preds,
                mode="lines", name=model_name,
                line=dict(color=MODEL_COLORS.get(model_name, SPE_GRAY), width=2, dash="dot")
            )
        fig_fc.add_scatter(
            x=[actuals_tail.index[-1], actuals_tail.index[-1]],
            y=[min(actuals_tail.min(), forecast_df.min().min()),
               max(actuals_tail.max(), forecast_df.max().max())],
            mode="lines", name="Forecast start",
            line=dict(color=SPE_GRAY, dash="dash", width=1)
        )
        spe_layout(fig_fc, f"Electricity Demand Forecast — {horizon_label}")
        fig_fc.update_layout(
            xaxis_title=dict(text="Date", font=dict(color=SPE_BLUE, size=15, family="Arial Black")),
            yaxis_title=dict(text="Demand (MW)", font=dict(color=SPE_BLUE, size=15, family="Arial Black"))
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.download_button(
            label="⬇ Download forecast CSV",
            data=forecast_df.to_csv().encode("utf-8"),
            file_name=f"forecast_{horizon_label.replace(' ','_')}.csv",
            mime="text/csv"
        )
