import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import forecasting_engine as fe

st.set_page_config(page_title="MIT SC1x Forecasting Suite", layout="wide")

st.title("📦 MIT SC1x Supply Chain Forecasting Suite")
st.markdown("Automatización integral de modelos cuantitativos, métricas de error y transiciones puntuales de examen.")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Calculadora Rápida (Exámenes)",
    "📈 Series Históricas Completas", 
    "📦 Demanda Intermitente (Croston)", 
    "🚀 Nuevos Productos (Bass)"
])

# -------------------------------------------------------------
# TAB 1: CALCULADORA RÁPIDA DE TRANSICIONES (TIPO PROBLEMA CERVEZA / BAGELS)[cite: 2, 4]
# -------------------------------------------------------------
with tab1:
    st.subheader("Paso Individual de Transición y Pronóstico a Tau Períodos")
    st.info("Diseñado para resolver preguntas que dan un estado previo consolidado (at-1, bt-1, Ft-P) y solicitan el pronóstico a futuro sin armar la tabla histórica previa.")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        a_prev = st.number_input("Nivel Previo (a_prev)", value=361.00)
        b_prev = st.number_input("Tendencia Previa (b_prev)", value=0.80)
        actual_x = st.number_input("Demanda Observada del Período (x_t)", value=520.00)
    with col_b:
        alpha = st.number_input("Alpha (α)", value=0.15, min_value=0.0, max_value=1.0)
        beta = st.number_input("Beta (β)", value=0.06, min_value=0.0, max_value=1.0)
        gamma = st.number_input("Gamma (γ)", value=0.05, min_value=0.0, max_value=1.0)
    with col_c:
        f_prev = st.number_input("Factor Estacional Previo Relevante (F_t-P)", value=1.38)
        tau = st.number_input("Horizonte de Pronóstico (Tau / períodos hacia adelante)", value=4, min_value=1)
        target_f = st.number_input("Factor Estacional Objetivo para el Pronóstico", value=1.38)

    if st.button("Calcular Actualización y Pronóstico", type="primary"):
        # Actualización según MIT SC1x[cite: 2, 4]
        a_t = alpha * (actual_x / f_prev) + (1.0 - alpha) * (a_prev + b_prev)
        b_t = beta * (a_t - a_prev) + (1.0 - beta) * b_prev
        f_updated = gamma * (actual_x / a_t) + (1.0 - gamma) * f_prev
        
        # Si el pronóstico apunta a la misma estación, toma el factor actualizado
        eff_factor = f_updated if target_f == f_prev else target_f
        forecast_tau = (a_t + tau * b_t) * eff_factor
        
        res1, res2, res3, res4 = st.columns(4)
        res1.metric("Nuevo Nivel (a_t)", f"{a_t:.4f}")
        res2.metric("Nueva Tendencia (b_t)", f"{b_t:.4f}")
        res3.metric("Factor Actualizado (F_t)", f"{f_updated:.4f}")
        res4.metric(f"Pronóstico x^(t, t+{tau})", f"{forecast_tau:.2f}")

# -------------------------------------------------------------
# TAB 2: SERIES DE TIEMPO COMPLETAS
# -------------------------------------------------------------
with tab2:
    st.subheader("Evaluación de Series Temporales y Métricas de Error")
    
    side_col, main_col = st.columns([1, 3])
    with side_col:
        model_choice = st.selectbox("Seleccione el Modelo", [
            "Naïve", "Cumulative", "Moving Average", "Simple Exponential Smoothing", 
            "Holt Linear Trend", "Holt-Winters Multiplicative"
        ])
        
        data_input_mode = st.radio("Ingreso de Datos", ["Texto Manual", "Subir CSV/Excel"])
        if data_input_mode == "Texto Manual":
            raw_text = st.text_area("Demandas separadas por coma", "28, 27, 30, 34, 32, 33, 32, 36, 33, 36")
            demand_list = [float(x.strip()) for x in raw_text.split(",") if x.strip()]
        else:
            uploaded_file = st.file_uploader("Cargar Archivo", type=["csv", "xlsx"])
            if uploaded_file:
                df_up = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                col_sel = st.selectbox("Columna de Demanda", df_up.columns)
                demand_list = df_up[col_sel].dropna().tolist()
            else:
                demand_list = [28, 27, 30, 34, 32, 33, 32, 36, 33, 36]

        st.markdown("---")
        # Parámetros condicionales
        if model_choice == "Moving Average":
            n_win = st.number_input("Ventana (N)", value=3, min_value=1)
        elif model_choice == "Simple Exponential Smoothing":
            s_alpha = st.number_input("Alpha", value=0.2, min_value=0.0, max_value=1.0)
            s_a0 = st.number_input("Nivel Inicial (a0)", value=float(demand_list[0]))
        elif model_choice == "Holt Linear Trend":
            h_alpha = st.number_input("Alpha", value=0.25, min_value=0.0, max_value=1.0)
            h_beta = st.number_input("Beta", value=0.10, min_value=0.0, max_value=1.0)
            h_a0 = st.number_input("a0", value=28.0)
            h_b0 = st.number_input("b0", value=1.35)
            h_phi = st.number_input("Damping (phi)", value=1.0, min_value=0.0, max_value=1.0)
        elif model_choice == "Holt-Winters Multiplicative":
            hw_alpha = st.number_input("Alpha", value=0.10, min_value=0.0, max_value=1.0)
            hw_beta = st.number_input("Beta", value=0.05, min_value=0.0, max_value=1.0)
            hw_gamma = st.number_input("Gamma", value=0.05, min_value=0.0, max_value=1.0)
            hw_p = st.number_input("Períodos por Ciclo (P)", value=4, min_value=2)
            hw_a0 = st.number_input("a0", value=100.0)
            hw_b0 = st.number_input("b0", value=1.0)
            factors_str = st.text_input("Factores Iniciales (separados por coma)", "1.38, 0.18, 0.94, 1.50")
            hw_factors = [float(x.strip()) for x in factors_str.split(",")]

    with main_col:
        # Ejecutar modelo
        if model_choice == "Naïve":
            res_df = fe.naive_forecast(demand_list)
        elif model_choice == "Cumulative":
            res_df = fe.cumulative_forecast(demand_list)
        elif model_choice == "Moving Average":
            res_df = fe.moving_average(demand_list, int(n_win))
        elif model_choice == "Simple Exponential Smoothing":
            res_df = fe.simple_exponential_smoothing(demand_list, s_alpha, s_a0)
        elif model_choice == "Holt Linear Trend":
            res_df = fe.holt_linear(demand_list, h_alpha, h_beta, h_a0, h_b0, h_phi)
        elif model_choice == "Holt-Winters Multiplicative":
            res_df = fe.holt_winters(demand_list, hw_alpha, hw_beta, hw_gamma, int(hw_p), hw_a0, hw_b0, hw_factors)

        # Cálculo de errores
        metrics, smoothed_mse = fe.compute_metrics(res_df["Actual"], res_df["Forecast"])
        if not smoothed_mse.empty:
            res_df["Smoothed_MSE"] = smoothed_mse

        # Mostrar KPIs
        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Bias (ME)", f"{metrics.get('Mean Error (Bias)', 0):.2f}")
        kpi_cols[1].metric("MAD / MAE", f"{metrics.get('MAD (MAE)', 0):.2f}")
        kpi_cols[2].metric("MSE", f"{metrics.get('MSE', 0):.2f}")
        kpi_cols[3].metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
        kpi_cols[4].metric("MAPE", f"{metrics.get('MAPE (%)', 0):.2f}%")

        # Gráfico Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=res_df["Actual"], mode='lines+markers', name='Demanda Real'))
        fig.add_trace(go.Scatter(y=res_df["Forecast"], mode='lines+markers', name='Pronóstico'))
        fig.update_layout(title="Demanda Real vs Pronóstico", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(res_df.style.format("{:.2f}", na_rep="-"), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: DEMANDA INTERMITENTE (CROSTON)[cite: 3]
# -------------------------------------------------------------
with tab3:
    st.subheader("Método de Croston para Repuestos y Demanda Esporádica")
    c_c1, c_c2 = st.columns([1, 3])
    with c_c1:
        croston_raw = st.text_area("Serie con Ceros (Demanda)", "0, 0, 10, 0, 0, 0, 15, 0, 20, 0")
        croston_dem = [float(x.strip()) for x in croston_raw.split(",") if x.strip()]
        c_alpha = st.number_input("Alpha (Magnitud z)", value=0.1, min_value=0.0, max_value=1.0)
        c_beta = st.number_input("Beta (Frecuencia n)", value=0.1, min_value=0.0, max_value=1.0)
        c_z0 = st.number_input("z0 Inicial", value=10.0)
        c_n0 = st.number_input("n0 Inicial", value=2.0)
        c_first_n = st.number_input("n primer período", value=1)
        
    with c_c2:
        c_res = fe.croston_method(croston_dem, c_alpha, c_beta, c_z0, c_n0, int(c_first_n))
        last_fcst = c_res["Forecast"].iloc[-1]
        st.metric("Tasa de Pronóstico Actual (Demanda/Período)", f"{last_fcst:.4f}")
        st.dataframe(c_res.style.format("{:.4f}"), use_container_width=True)

# -------------------------------------------------------------
# TAB 4: DIFUSIÓN DE BASS (NUEVOS PRODUCTOS)[cite: 3]
# -------------------------------------------------------------
with tab4:
    st.subheader("Modelo de Difusión de Bass")
    b_c1, b_c2 = st.columns([1, 3])
    with b_c1:
        p_val = st.number_input("Coeficiente de Innovación (p)", value=0.03, format="%.4f")
        q_val = st.number_input("Coeficiente de Imitación (q)", value=0.38, format="%.4f")
        m_val = st.number_input("Mercado Potencial (M)", value=100000.0)
        horiz = st.slider("Horizonte de Períodos", min_value=5, max_value=30, value=12)
        
    with b_c2:
        bass_df = fe.bass_diffusion(p_val, q_val, m_val, horiz)
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(x=bass_df["Period"], y=bass_df["Adopters Sales S(t)"], name="Ventas por Período S(t)"))
        fig_b.add_trace(go.Scatter(x=bass_df["Period"], y=bass_df["Cumulative Adopters N(t)"], name="Adoptantes Acumulados N(t)", yaxis="y2"))
        fig_b.update_layout(
            title="Curva de Difusión de Bass",
            yaxis=dict(title="Ventas Periódicas"),
            yaxis2=dict(title="Adoptantes Acumulados", overlaying="y", side="right"),
            template="plotly_white"
        )
        st.plotly_chart(fig_b, use_container_width=True)
        st.dataframe(bass_df.style.format({"Adopters Sales S(t)": "{:.1f}", "Cumulative Adopters N(t)": "{:.1f}"}), use_container_width=True)