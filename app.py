import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import forecasting_engine as fe

st.set_page_config(page_title="MIT SC1x Forecasting Suite", layout="wide")

st.title("📦 MIT SC1x Supply Chain Forecasting Suite")

tab1, tab2, tab3 = st.tabs([
    "⚡ Calculadora Rápida (Exámenes)",
    "📊 Comparador Multi-Modelo",
    "📦 Demanda Intermitente (Croston)"
])

# -------------------------------------------------------------
# TAB 1: CALCULADORA RÁPIDA DE EXÁMENES (PROYECCIÓN & TRANSICIÓN)
# -------------------------------------------------------------
with tab1:
    st.subheader("Calculadora Rápida para Preguntas Puntuales de Examen")
    
    calc_mode = st.radio(
        "Seleccione el tipo de problema:",
        ["Proyección Directa (Ya conozco a_t y b_t)", "Actualización con Nueva Demanda (Conozco x_t, α, β, γ)"],
        horizontal=True
    )
    
    if calc_mode == "Proyección Directa (Ya conozco a_t y b_t)":
        st.caption("Aplica la fórmula directa: x^(t, t+tau) = (a_t + tau * b_t) * F_target")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            a_curr = st.number_input("Nivel Actual (a_t)", value=1053.00)
        with c2:
            b_curr = st.number_input("Tendencia Actual (b_t)", value=41.20)
        with c3:
            tau_dir = st.number_input("Horizonte Tau (períodos hacia adelante)", value=1, min_value=1)
        with c4:
            f_target = st.number_input("Factor Estacional del período objetivo", value=1.20)
            
        if st.button("Calcular Pronóstico Directo", type="primary"):
            direct_fcst = (a_curr + tau_dir * b_curr) * f_target
            deseasonalized = a_curr + tau_dir * b_curr
            
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("Demanda Desestacionalizada Base", f"{deseasonalized:.2f}")
            res_c2.metric(f"Pronóstico Final x^(t, t+{tau_dir})", f"{direct_fcst:.2f}")

    else:
        st.caption("Actualiza nivel, tendencia y factor tras observar una venta real en t.")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            a_prev = st.number_input("Nivel previo (a_t-1)", value=361.00)
            b_prev = st.number_input("Tendencia previa (b_t-1)", value=0.80)
            actual_x = st.number_input("Demanda observada (x_t)", value=520.00)
        with col_b:
            alpha = st.number_input("Alpha (α)", value=0.15, min_value=0.0, max_value=1.0)
            beta = st.number_input("Beta (β)", value=0.06, min_value=0.0, max_value=1.0)
            gamma = st.number_input("Gamma (γ)", value=0.05, min_value=0.0, max_value=1.0)
        with col_c:
            f_prev = st.number_input("Factor Estacional previo (F_t-P)", value=1.38)
            tau = st.number_input("Horizonte tau (períodos hacia adelante)", value=4, min_value=1)
            target_f = st.number_input("Factor objetivo", value=1.38)

        if st.button("Calcular Transición y Pronóstico", type="primary"):
            a_t = alpha * (actual_x / f_prev) + (1.0 - alpha) * (a_prev + b_prev)
            b_t = beta * (a_t - a_prev) + (1.0 - beta) * b_prev
            f_updated = gamma * (actual_x / a_t) + (1.0 - gamma) * f_prev
            eff_factor = f_updated if target_f == f_prev else target_f
            forecast_tau = (a_t + tau * b_t) * eff_factor
            
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Nuevo Nivel (a_t)", f"{a_t:.4f}")
            r2.metric("Nueva Tendencia (b_t)", f"{b_t:.4f}")
            r3.metric("Factor Actualizado (F_t)", f"{f_updated:.4f}")
            r4.metric(f"Pronóstico x^(t, t+{tau})", f"{forecast_tau:.2f}")

# -------------------------------------------------------------
# TAB 2: COMPARADOR MULTI-MODELO EN UN SOLO GRÁFICO
# -------------------------------------------------------------
with tab2:
    st.subheader("Comparación de Múltiples Modelos")
    
    col_input, col_graph = st.columns([1, 3])
    
    with col_input:
        raw_data = st.text_area(
            "Pega la Demanda Histórica (separada por comas):",
            "28, 27, 30, 34, 32, 33, 32, 36, 33, 36"
        )
        demand = [float(x.strip()) for x in raw_data.split(",") if x.strip()]
        
        selected_models = st.multiselect(
            "Elige los modelos que quieres ver juntos:",
            ["Naïve", "Cumulative", "Moving Average", "Simple Exp Smoothing", "Holt Linear Trend"],
            default=["Naïve", "Moving Average", "Simple Exp Smoothing"]
        )
        
        st.markdown("---")
        st.caption("Ajustes de Parámetros:")
        ma_window = st.number_input("Ventana MA (N)", value=3, min_value=1)
        ses_alpha = st.slider("Alpha para SES", 0.01, 0.99, 0.20, 0.05)
        h_alpha = st.slider("Alpha para Holt", 0.01, 0.99, 0.25, 0.05)
        h_beta = st.slider("Beta para Holt", 0.01, 0.99, 0.10, 0.05)
        
    with col_graph:
        if not selected_models:
            st.warning("Selecciona al menos un modelo para graficar.")
        else:
            # 1. Gráfico interactivo con la Demanda Real de fondo
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=demand, 
                mode='lines+markers', 
                name='Demanda Real',
                line=dict(color='black', width=3),
                marker=dict(size=6)
            ))
            
            error_table = []
            
            # 2. Agregar cada modelo seleccionado al mismo gráfico
            for model_name in selected_models:
                if model_name == "Naïve":
                    df_res = fe.naive_forecast(demand)
                elif model_name == "Cumulative":
                    df_res = fe.cumulative_forecast(demand)
                elif model_name == "Moving Average":
                    df_res = fe.moving_average(demand, int(ma_window))
                elif model_name == "Simple Exp Smoothing":
                    df_res = fe.simple_exponential_smoothing(demand, ses_alpha, demand[0])
                elif model_name == "Holt Linear Trend":
                    df_res = fe.holt_linear(demand, h_alpha, h_beta, demand[0], 0.0)
                    
                # Añadir la línea del pronóstico
                fig.add_trace(go.Scatter(
                    y=df_res["Forecast"],
                    mode='lines+markers',
                    name=model_name,
                    line=dict(dash='dash')
                ))
                
                # Calcular métricas de error
                metrics, _ = fe.compute_metrics(df_res["Actual"], df_res["Forecast"])
                error_table.append({
                    "Modelo": model_name,
                    "MAD (Error Absoluto)": metrics.get("MAD (MAE)", np.nan),
                    "MSE (Error Cuadrático)": metrics.get("MSE", np.nan),
                    "MAPE (%)": metrics.get("MAPE (%)", np.nan),
                    "Bias (Sesgo)": metrics.get("Mean Error (Bias)", np.nan)
                })
                
            fig.update_layout(
                title="Comparación Visual de Pronósticos",
                xaxis_title="Período (t)",
                yaxis_title="Unidades",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. Mostrar la tabla comparativa de errores
            st.markdown("#### Desempeño y Precisión de cada Modelo:")
            df_errors = pd.DataFrame(error_table).sort_values(by="MAD (Error Absoluto)")
            st.dataframe(
                df_errors.style.format({
                    "MAD (Error Absoluto)": "{:.2f}",
                    "MSE (Error Cuadrático)": "{:.2f}",
                    "MAPE (%)": "{:.2f}%",
                    "Bias (Sesgo)": "{:.2f}"
                }).highlight_min(subset=["MAD (Error Absoluto)", "MSE (Error Cuadrático)"], color="#d4edda"),
                use_container_width=True
            )

# -------------------------------------------------------------
# TAB 3: CROSTON (DEMANDA INTERMITENTE)
# -------------------------------------------------------------
with tab3:
    st.subheader("Método de Croston para Demanda Esporádica")
    c_in, c_out = st.columns([1, 3])
    with c_in:
        c_raw = st.text_area("Serie con Ceros:", "0, 0, 10, 0, 0, 0, 15, 0, 20, 0")
        c_dem = [float(x.strip()) for x in c_raw.split(",") if x.strip()]
        c_a = st.number_input("Alpha (magnitud z)", value=0.1)
        c_b = st.number_input("Beta (frecuencia n)", value=0.1)
        c_z0 = st.number_input("z0 inicial", value=10.0)
        c_n0 = st.number_input("n0 inicial", value=2.0)
    with c_out:
        df_croston = fe.croston_method(c_dem, c_a, c_b, c_z0, c_n0)
        st.metric("Tasa de Pronóstico Final", f"{df_croston['Forecast'].iloc[-1]:.4f}")
        st.dataframe(df_croston.style.format("{:.4f}"), use_container_width=True)
