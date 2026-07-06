import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image

# Intento de importar el canvas
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="MNIST Classifier IA",
    page_icon="🔢",
    layout="centered"
)

# ============================================================
# ESTILOS "SOFT DARK" (Legibilidad total)
# ============================================================
st.markdown("""
<style>
    /* Fondo Gris Oscuro Suave */
    .stApp { background-color: #1e1e2e; }
    
    /* Textos claros */
    h1, h2, h3, p, label, .stMarkdown { color: #e4e4e7 !important; }
    
    /* Tarjeta de resultado con contraste */
    .result-card {
        background: #2d2d44;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #444466;
        margin: 1.5rem 0;
    }
    
    /* Ajuste de Canvas */
    canvas { border: 2px solid #60a5fa !important; border-radius: 8px; }
    
    /* Botón principal */
    div.stButton > button {
        background-color: #60a5fa !important;
        color: #1e1e2e !important;
        font-weight: 700 !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODELOS
# ============================================================
MODEL_DIR = Path("models")
@st.cache_resource
def cargar_modelos():
    # Asegúrate de que estos archivos estén en la carpeta 'models'
    pca = joblib.load(MODEL_DIR / "pca_mnist.pkl")
    kmeans = joblib.load(MODEL_DIR / "kmeans_mnist.pkl")
    svm_models = joblib.load(MODEL_DIR / "svm_mnist.pkl")
    return pca, kmeans, svm_models

# Descomenta la siguiente línea cuando los modelos estén listos en la carpeta 'models'
# pca, kmeans, svm_models = cargar_modelos()

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 style='text-align:center;'>🔢 Clasificador MNIST</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#a1a1aa;'>PCA + KMeans + SVM | Angeles Euceda | 20221930061</p>", unsafe_allow_html=True)
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
    st.info("🎲 Datos aleatorios generados.")

else:
    if CANVAS_AVAILABLE:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dibuja aquí:**")
            canvas_result = st_canvas(
                fill_color="white", stroke_width=15, stroke_color="white",
                background_color="black", height=280, width=280,
                drawing_mode="freedraw", key="canvas_dibujo"
            )
        with col2:
            st.markdown("**Procesado:**")
            if canvas_result.image_data is not None:
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                img_gray = img.convert('L')
                img_28 = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
                sample = np.array(img_28).flatten().astype(float)
                st.image(img_28, width=150, clamp=True)
                st.caption(f"Píxeles activos: {(sample > 30).sum()}/784")
            else:
                st.warning("Dibuja algo 👈")
                sample = np.zeros(784)
    else:
        st.error("⚠️ `streamlit-drawable-canvas` no está instalado.")
        sample = np.zeros(784)

# ============================================================
# PREDICCIÓN
# ============================================================
if sample is not None:
    if st.button("🔍 Predecir dígito", type="primary", use_container_width=True):
        # NORMALIZACIÓN Y PREDICCIÓN
        # sample_norm = sample / 255.0
        # sample_pca = pca.transform(sample_norm.reshape(1, -1))
        # pred = svm_models["rbf"].predict(sample_pca)[0]
        # cluster = kmeans.predict(sample_pca)[0]
        
        # Valores de prueba mientras los modelos no estén cargados
        pred, cluster = 5, 2
        
        st.divider()
        st.html(f"""
        <div class="result-card">
            <p style="color: #a1a1aa; margin: 0;">🎯 Dígito Predicho</p>
            <p style="font-size: 5rem; font-weight: 800; color: #60a5fa; margin: 0; line-height: 1;">{pred}</p>
            <hr style="border-top: 1px solid #444466; margin: 1rem 0;">
            <p style="color: #e4e4e7;">Cluster KMeans: <b style="color: #fbbf24;">{cluster}</b></p>
        </div>
        """)

        with st.expander("🔍 Ver detalles técnicos"):
            st.write(f"**Media de píxeles:** {sample.mean():.1f}")
            st.write(f"**Predicción:** dígito {pred}, cluster {cluster}")

st.divider()
st.caption("Desarrollado con ❤️ | MNIST Classifier IA | 2026")
