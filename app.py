import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# ============================================================
# CONFIGURACION
# ============================================================
st.set_page_config(page_title="Digit Classifier IA", page_icon="0", layout="centered")

# ============================================================
# MODELOS
# ============================================================
MODEL_DIR = Path("models")
@st.cache_resource
def cargar_modelos():
    return joblib.load(MODEL_DIR / "pca_mnist.pkl"), \
           joblib.load(MODEL_DIR / "kmeans_mnist.pkl"), \
           joblib.load(MODEL_DIR / "svm_mnist.pkl")

# Descomenta la siguiente línea cuando los archivos existan
# pca, kmeans, svm_models = cargar_modelos()

# ============================================================
# ESTILOS GLOBALES
# ============================================================
st.markdown("""
<style>
/* Fondo Global */
[data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0d0d0d !important;
}

/* Tipografía */
* { font-family: 'IBM Plex Mono', monospace !important; color: #f5f0e8; }
h1 { font-family: 'Cormorant Garamond', serif !important; }

/* Corrección de Iconos del Canvas */
div[data-testid="st_canvas"] button {
    background-color: #222 !important;
    border: 1px solid #e8c547 !important;
}
div[data-testid="st_canvas"] button svg {
    fill: #e8c547 !important;
}

/* Estética de Componentes */
.app-title { text-align: center; font-size: 3.2rem; color: #f5f0e8; }
.app-title span { color: #e8c547; font-style: italic; }
.section-tag { 
    color: #e8c547; border: 1px solid #e8c547; 
    padding: 2px 10px; display: inline-block; font-size: 0.6rem;
}
.gold-rule { border: 0; height: 1px; background: #e8c547; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INTERFAZ
# ============================================================
st.markdown("<h1 class='app-title'>Digit <span>Classifier</span></h1>", unsafe_allow_html=True)
st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)

st.markdown("<div class='section-tag'>ENTRADA</div>", unsafe_allow_html=True)

modo = st.radio("Modo", ["Ejemplo aleatorio", "Dibujar digito"], horizontal=True, label_visibility="collapsed")

if modo == "Dibujar digito" and CANVAS_AVAILABLE:
    col1, col2 = st.columns(2)
    with col1:
        canvas_result = st_canvas(
            fill_color="#000000", stroke_width=15, stroke_color="#FFFFFF",
            background_color="#000000", height=280, width=280,
            drawing_mode="freedraw", key="canvas_dibujo"
        )
    # Procesamiento... (tu lógica original aquí)
else:
    st.write("Selecciona una opción para comenzar.")
