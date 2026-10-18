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
