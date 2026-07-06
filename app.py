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
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="MNIST Classifier IA",
    page_icon="🔢",
    layout="centered"
)

# ============================================================
# ESTILOS (DISEÑO CLARO)
# ============================================================
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #f8f9fa; }
    
    /* Títulos y texto */
    h1, h2, h3 { color: #2d3436 !important; }
    
    /* Tarjeta de resultado */
    .result-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin: 1.5rem 0;
    }
    
    /* Botón */
    div.stButton > button {
        background-color: #1976d2 !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODELOS
# ============================================================
MODEL_DIR = Path("models")

@st.cache_resource
def cargar_modelos():
    pca = joblib.load(MODEL_DIR / "pca_mnist.pkl")
    kmeans = joblib.load(MODEL_DIR / "kmeans_mnist.pkl")
    svm_models = joblib.load(SVM_PATH)
    return pca, kmeans, svm_models

# Nota: Asegúrate de tener los archivos .pkl en la carpeta 'models'
# pca, kmeans, svm_models = cargar_modelos()

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 style='text-align:center;'>🔢 Clasificador MNIST</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6c757d;'>PCA + KMeans + SVM | Angeles Euceda | 20221930061</p>", unsafe_allow_html=True)
st.divider()

# ============================================================
# INPUT
# ============================================================
st.subheader("✏️ Ingresa un dígito")
modo = st.radio("Modo de entrada", ["🎲 Ejemplo automático", "✏️ Dibujar"], horizontal=True)

sample = None

if modo == "🎲 Ejemplo automático":
    np.random.seed(None)
    sample = np.random.randint(0, 256, size=784).astype(float)
    img_example = sample.reshape(28, 28)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(img_example, caption="Ejemplo generado", width=150, clamp=True)
    st.info(f"🎲 Datos aleatorios generados.")

else:
    if CANVAS_AVAILABLE:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dibuja aquí:**")
            canvas_result = st_canvas(
                fill_color="black", stroke_width=15, stroke_color="white",
                background_color="black", height=280, width=280,
                drawing_mode="freedraw", key="canvas_dibujo"
            )
        with col2:
            st.markdown("**Procesado:**")
            if canvas_result.image_data is not None:
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                img_28 = img.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
                sample = np.array(img_28).flatten().astype(float)
                st.image(img_28, width=150, clamp=True)
                st.caption(f"Píxeles activos: {(sample > 30).sum()}/784")
            else:
                sample = np.zeros(784)
    else:
        st.error("⚠️ `streamlit-drawable-canvas` no está instalado.")

# ============================================================
# PREDICCIÓN
# ============================================================
if sample is not None:
    if st.button("🔍 Predecir dígito", type="primary", use_container_width=True):
        # Descomentar estas líneas cuando los modelos estén cargados:
        # sample_pca = pca.transform((sample / 255.0).reshape(1, -1))
        # pred = svm_models["rbf"].predict(sample_pca)[0]
        # cluster = kmeans.predict(sample_pca)[0]
        
        # Valores de prueba (simulados)
        pred, cluster = 5, 2
        
        st.divider()
        st.html(f"""
        <div class="result-card">
            <p style="color: #6c757d; margin: 0;">🎯 Dígito Predicho</p>
            <p style="font-size: 5rem; font-weight: 800; color: #1976d2; margin: 0;">{pred}</p>
            <hr style="margin: 1rem 0;">
            <p style="color: #495057;">Cluster KMeans: <b>{cluster}</b></p>
        </div>
        """)

        with st.expander("🔍 Ver detalles técnicos"):
            st.write(f"**Media de píxeles:** {sample.mean():.1f}")
            st.write(f"**Predicción:** dígito {pred}, cluster {cluster}")

st.divider()
st.caption("Desarrollado con ❤️ | 2026")
