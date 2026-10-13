import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

# Configurar path para importar módulos locales
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.models.arps import ArpsModeler
from src.models.economics import EconomicEvaluator

# Configuración de la página
st.set_page_config(page_title="Vaca Muerta DCA & Economics", layout="wide")

@st.cache_data
def load_data():
    data_path = root_dir / 'data' / 'processed' / 'vaca_muerta_prod.parquet'
    return pd.read_parquet(data_path)

# --- HEADER ---
st.title("🛢️ Motor de Decline Curve Analysis (DCA) - Vaca Muerta")
st.markdown("""
Aplicación analítica para la estimación de reservas (EUR) y evaluación económica (NPV) 
en pozos no convencionales utilizando modelos físicos (Modified Arps, Stretched Exponential) 
y limpieza de datos.
""")

try:
    df = load_data()
except FileNotFoundError:
    st.error("No se encontró el archivo de datos procesados. Por favor, corre el pipeline de datos primero.")
    st.stop()

# --- SIDEBAR (Controles) ---
st.sidebar.header("Filtros del Pozo")

# 1. Seleccionar Empresa
empresas = sorted(df['empresa'].dropna().unique())
empresa_seleccionada = st.sidebar.selectbox("Operadora", empresas, index=empresas.index('YPF S.A.') if 'YPF S.A.' in empresas else 0)

# 2. Seleccionar Pozo
df_empresa = df[df['empresa'] == empresa_seleccionada]
pozos = sorted(df_empresa['sigla'].unique())
pozo_seleccionado = st.sidebar.selectbox("Pozo (Sigla)", pozos)

st.sidebar.markdown("---")
st.sidebar.header("Parámetros del Modelo")
modelo_seleccionado = st.sidebar.selectbox("Modelo Físico", ['modified', 'hyperbolic', 'stretched'])

st.sidebar.markdown("---")
st.sidebar.header("Escenario Económico")
precio_barril = st.sidebar.number_input("Precio Crudo (USD/bbl)", value=70.0, step=1.0)
opex = st.sidebar.number_input("Lifting Cost (USD/bbl)", value=6.0, step=0.5)
capex_m_usd = st.sidebar.number_input("CAPEX D&C (Millones USD)", value=12.0, step=0.5)

# --- PROCESAMIENTO ---
# Filtrar datos del pozo
df_pozo = df_empresa[df_empresa['sigla'] == pozo_seleccionado].sort_values('mes_relativo')
t_raw = df_pozo['mes_relativo'].values
q_raw = df_pozo['prod_pet'].values

# Limpieza básica (aislar desde el pico)
if len(q_raw) > 0 and max(q_raw) > 0:
    idx_pico = np.argmax(q_raw)
    t_post = t_raw[idx_pico:]
    q_post = q_raw[idx_pico:]
    mask = q_post > 0
    t_norm = t_post[mask] - t_post[mask][0]
    q_norm = q_post[mask]
    
    # Ajuste del Modelo Físico
    model = ArpsModeler(model_type=modelo_seleccionado)
    try:
        model.fit(t_norm, q_norm)
        
        # Proyección a 10 años (120 meses)
        t_pred = np.arange(0, 120)
        q_pred = model.predict(t_pred)
        
        # Cálculos de Reservas (EUR) a 20 años (240 meses)
        eur_m3 = model.calculate_eur(240)
        eur_bbl = eur_m3 * 6.2898
        
        # Evaluación Económica
        econ = EconomicEvaluator(oil_price_usd_bbl=precio_barril, opex_usd_bbl=opex)
        npv = econ.calculate_npv(model.predict(np.arange(0, 240)))
        capex_usd = capex_m_usd * 1_000_000
        payout_meses = econ.calculate_payout_time(model.predict(np.arange(0, 240)), capex_usd)
        
        # --- DASHBOARD UI ---
        # 1. KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("EUR (Total Reservas)", f"{eur_bbl/1000:.1f}k bbl", help="Estimated Ultimate Recovery a 20 años")
        col2.metric("NPV @ 10%", f"${npv/1000000:.1f} MM", help="Valor Presente Neto descontado al 10% anual")
        
        payout_text = f"{payout_meses} meses" if payout_meses != np.inf else "Nunca"
        col3.metric("Tiempo de Payout", payout_text, help="Meses hasta recuperar el CAPEX inicial")
        
        # 2. Gráfico Interactivo (Plotly)
        st.subheader("Pronóstico de Producción vs Datos Históricos")
        fig = go.Figure()
        
        # Datos reales
        fig.add_trace(go.Scatter(x=t_norm, y=q_norm, mode='markers', name='Producción Real', 
                                 marker=dict(color='black', size=8, opacity=0.6)))
        
        # Curva de ajuste
        b_val = model.params.get('b', 0)
        n_val = model.params.get('n', 0)
        label_ajuste = f'Ajuste (b={b_val:.2f})' if modelo_seleccionado != 'stretched' else f'Ajuste (n={n_val:.2f})'
        
        fig.add_trace(go.Scatter(x=t_pred, y=q_pred, mode='lines', name=label_ajuste,
                                 line=dict(color='red', width=3)))
        
        fig.update_layout(
            xaxis_title="Meses de Producción",
            yaxis_title="Caudal de Petróleo (m³/mes)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Parámetros Matemáticos
        with st.expander("Ver Parámetros Matemáticos del Modelo"):
            st.json(model.params)
            
    except Exception as e:
        st.error(f"El modelo no pudo converger para este pozo. Excepción: {e}")
else:
    st.warning("Este pozo no tiene producción registrada o datos suficientes.")
