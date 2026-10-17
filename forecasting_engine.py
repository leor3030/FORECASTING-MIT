import numpy as np
import pandas as pd

# ==========================================
# 1. MÉTRICAS DE ERROR ESTÁNDAR MIT SC1x
# ==========================================
def compute_metrics(actual, forecast, omega=0.05, mse_init=None):
    """
    Calcula Bias (ME), MAD, MSE, RMSE, MAPE y Smoothed MSE (con parámetro omega)[cite: 3, 7, 8].
    """
    df = pd.DataFrame({"Actual": actual, "Forecast": forecast}).dropna()
    if df.empty:
        return {}, pd.Series(dtype=float)
    
    e = df["Actual"] - df["Forecast"]
    abs_e = e.abs()
    sq_e = e ** 2
    pct_e = abs_e / df["Actual"].replace(0, np.nan)
    
    # Smoothed MSE (MIT SC1x Lesson: Level & Trend)[cite: 8]
    smoothed_mse = []
    current_smse = mse_init if mse_init is not None else sq_e.iloc[0]
    for val in sq_e:
        current_smse = omega * val + (1.0 - omega) * current_smse
        smoothed_mse.append(current_smse)
        
    metrics = {
        "Mean Error (Bias)": float(e.mean()),
        "MAD (MAE)": float(abs_e.mean()),
        "MSE": float(sq_e.mean()),
        "RMSE": float(np.sqrt(sq_e.mean())),
        "MAPE (%)": float(pct_e.mean() * 100.0),
        "Last Smoothed MSE": float(smoothed_mse[-1]) if smoothed_mse else 0.0
    }
    return metrics, pd.Series(smoothed_mse, index=df.index)

# ==========================================
# 2. MODELOS BÁSICOS DE SERIES DE TIEMPO
# ==========================================
def cumulative_forecast(demand):
    """Pronóstico acumulativo: promedio de toda la historia disponible[cite: 3, 5]."""
    n = len(demand)
    forecast = [np.nan] * n
    for t in range(1, n):
        forecast[t] = np.mean(demand[:t])
    return pd.DataFrame({"Actual": demand, "Forecast": forecast})

def naive_forecast(demand):
    """Pronóstico ingenuo: el pronóstico de t es la observación en t-1[cite: 3, 5]."""
    n = len(demand)
    forecast = [np.nan] * n
    for t in range(1, n):
        forecast[t] = demand[t - 1]
    return pd.DataFrame({"Actual": demand, "Forecast": forecast})

def moving_average(demand, window):
    """Promedio móvil simple de orden N[cite: 3, 5]."""
    n = len(demand)
    forecast = [np.nan] * n
    for t in range(window, n):
        forecast[t] = np.mean(demand[t - window:t])
    return pd.DataFrame({"Actual": demand, "Forecast": forecast})

# ==========================================
# 3. SUAVIZAMIENTO EXPONENCIAL
# ==========================================
def simple_exponential_smoothing(demand, alpha, a0):
    """SES con constante alpha y nivel inicial a0[cite: 3]."""
    n = len(demand)
    level = [np.nan] * n
    forecast = [np.nan] * n
    
    current_a = a0
    for t in range(n):
        forecast[t] = current_a
        current_a = alpha * demand[t] + (1.0 - alpha) * current_a
        level[t] = current_a
    return pd.DataFrame({"Actual": demand, "Level": level, "Forecast": forecast})

def holt_linear(demand, alpha, beta, a0, b0, phi=1.0):
    """Modelo de Holt con opción de factor de amortiguamiento (Damped Trend) phi[cite: 3, 8]."""
    n = len(demand)
    level = [np.nan] * n
    trend = [np.nan] * n
    forecast = [np.nan] * n
    
    at_prev, bt_prev = a0, b0
    for t in range(n):
        forecast[t] = at_prev + phi * bt_prev
        at = alpha * demand[t] + (1.0 - alpha) * (at_prev + phi * bt_prev)
        bt = beta * (at - at_prev) + (1.0 - beta) * (phi * bt_prev)
        level[t], trend[t] = at, bt
        at_prev, bt_prev = at, bt
    return pd.DataFrame({"Actual": demand, "Level": level, "Trend": trend, "Forecast": forecast})

def holt_winters(demand, alpha, beta, gamma, season_len, a0, b0, initial_factors, normalize=True):
    """
    Holt-Winters multiplicativo completo con normalización de factores (suma a P).
    """
    n = len(demand)
    P = season_len
    factors = list(initial_factors)
    level = [np.nan] * n
    trend = [np.nan] * n
    forecast = [np.nan] * n
    applied_factor = [np.nan] * n
    
    at_prev, bt_prev = a0, b0
    for t in range(n):
        f_idx = t % P
        f_prev = factors[f_idx]
        applied_factor[t] = f_prev
        
        # Pronóstico
        forecast[t] = (at_prev + bt_prev) * f_prev
        
        # Actualizaciones de nivel, tendencia y factor estacional
        at = alpha * (demand[t] / f_prev) + (1.0 - alpha) * (at_prev + bt_prev)
        bt = beta * (at - at_prev) + (1.0 - beta) * bt_prev
        new_f = gamma * (demand[t] / at) + (1.0 - gamma) * f_prev
        factors[f_idx] = new_f
        
        # Normalización estacional del MIT[cite: 4]
        if normalize:
            f_sum = sum(factors[:P])
            factors = [f * (P / f_sum) for f in factors]
            
        level[t], trend[t] = at, bt
        at_prev, bt_prev = at, bt
        
    return pd.DataFrame({
        "Actual": demand, 
        "Level": level, 
        "Trend": trend, 
        "Seasonal_Factor": applied_factor, 
        "Forecast": forecast
    })

# ==========================================
# 4. CASOS ESPECIALES: CROSTON & BASS
# ==========================================
def croston_method(demand, alpha, beta, z0, n0, first_n=1):
    """
    Método de Croston para demanda intermitente[cite: 3].
    Separa el tamaño de la orden (z^) del intervalo de tiempo entre órdenes (n^)[cite: 3].
    """
    n_obs = len(demand)
    n_counter = [0] * n_obs
    z_hat = [0.0] * n_obs
    n_hat = [0.0] * n_obs
    forecast = [0.0] * n_obs
    
    z_prev, n_prev = float(z0), float(n0)
    for t in range(n_obs):
        if t == 0:
            c_n = first_n
        else:
            c_n = 1 if demand[t - 1] > 0 else n_counter[t - 1] + 1
        n_counter[t] = c_n
        
        if demand[t] > 0:
            z_curr = alpha * demand[t] + (1.0 - alpha) * z_prev
            n_curr = beta * c_n + (1.0 - beta) * n_prev
        else:
            z_curr, n_curr = z_prev, n_prev
            
        z_hat[t], n_hat[t] = z_curr, n_curr
        forecast[t] = z_curr / n_curr
        z_prev, n_prev = z_curr, n_curr
        
    return pd.DataFrame({
        "Actual": demand,
        "n_counter": n_counter,
        "z_hat (Size)": z_hat,
        "n_hat (Interval)": n_hat,
        "Forecast": forecast
    })

def bass_diffusion(p, q, M, periods=10):
    """
    Modelo de difusión de Bass para nuevos productos[cite: 3].
    S(t) = M * [ (p + q)^2 / p ] * e^(-(p+q)t) / [ 1 + (q/p) * e^(-(p+q)t) ]^2
    """
    t_vals = np.arange(1, periods + 1)
    sales = []
    cum_sales = []
    
    N_t = 0.0
    for t in t_vals:
        s_t = p * M + (q - p) * N_t - (q / M) * (N_t ** 2)
        sales.append(max(0.0, s_t))
        N_t += s_t
        cum_sales.append(N_t)
        
    return pd.DataFrame({
        "Period": t_vals,
        "Adopters Sales S(t)": sales,
        "Cumulative Adopters N(t)": cum_sales
    })