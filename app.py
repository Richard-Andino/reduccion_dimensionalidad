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
st.set_page_config(
    page_title="Digit Classifier IA",
    page_icon="0",
    layout="centered"
)

# ============================================================
# MODELOS
# ============================================================
MODEL_DIR = Path("models")

PCA_PATH    = MODEL_DIR / "pca_mnist.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_mnist.pkl"
SVM_PATH    = MODEL_DIR / "svm_mnist.pkl"

@st.cache_resource
def cargar_modelos():
    pca        = joblib.load(PCA_PATH)
    kmeans     = joblib.load(KMEANS_PATH)
    svm_models = joblib.load(SVM_PATH)
    return pca, kmeans, svm_models

pca, kmeans, svm_models = cargar_modelos()

# ============================================================
# ESTILOS GLOBALES
# ============================================================
st.markdown("""
<style>
/* 1. Reset Global y Estilos de Fuente */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d0d0d !important;
}
* { font-family: 'IBM Plex Mono', monospace !important; }

/* 2. Corrección de Iconos Canvas (¡Importante!) */
/* Forzamos el color de los botones de la barra de herramientas del canvas */
div[data-testid="st_canvas"] button {
    color: #f5f0e8 !important; /* Color del icono */
    background: rgba(232, 197, 71, 0.1) !important;
    border: 1px solid rgba(232, 197, 71, 0.2) !important;
    border-radius: 4px !important;
}
/* Asegurar visibilidad de iconos SVG dentro de botones */
div[data-testid="st_canvas"] button svg {
    fill: #f5f0e8 !important;
}
/* Cambia el fondo del contenedor principal y el cuerpo de la página */
[data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #fffff !important; /* Gris casi negro */
}
/* 3. Ajustes de Cabecera */
.app-eyebrow {
    text-align: center; color: #e8c547; font-size: 0.6rem;
    letter-spacing: 0.3em; text-transform: uppercase; margin-bottom: 0.5rem;
}
.app-title {
    text-align: center; font-family: 'Cormorant Garamond', serif !important;
    font-size: 3.2rem; font-weight: 700; color: #f5f0e8;
}
.app-title span { color: #e8c547; font-style: italic; }

/* 4. Estilos de Controles */
.section-tag {
    display: inline-block; background: rgba(232,197,71,0.1);
    color: #e8c547; border: 1px solid rgba(232,197,71,0.3);
    font-size: 0.5rem; letter-spacing: 0.2em;
    padding: 0.2rem 0.6rem; margin: 1rem 0;
}

/* 5. Radio Buttons */
div[role="radiogroup"] label {
    color: #f5f0e8 !important;
}

/* 6. Botones Principales */
.stButton > button {
    background: transparent !important;
    color: #e8c547 !important;
    border: 1px solid #e8c547 !important;
    border-radius: 0 !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: #e8c547 !important;
    color: #0d0d0d !important;
}

/* 7. Texto y Métricas */
[data-testid="stCaptionContainer"] {
    color: rgba(245,240,232,0.5) !important;
}

/* 8. Separador */
.gold-rule {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(232,197,71,0.3), transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class='app-eyebrow'>Inteligencia Artificial &nbsp;&middot;&nbsp; MNIST &nbsp;&middot;&nbsp; 2026</div>
<h1 class='app-title'>Digit <span>Classifier</span></h1>
<p class='app-byline'>Richard Andino &nbsp;&middot;&nbsp; 20231900184</p>
<hr class='gold-rule'>
""", unsafe_allow_html=True)

# ============================================================
# INPUT
# ============================================================
st.markdown("<div class='section-tag'>entrada</div>", unsafe_allow_html=True)

modo = st.radio(
    "Modo",
    ["Ejemplo aleatorio", "Dibujar digito"],
    horizontal=True,
    key="modo",
    label_visibility="collapsed"
)

sample = None

if modo == "Ejemplo aleatorio":
    np.random.seed(None)
    sample = np.random.randint(0, 256, size=784).astype(float)

    img_example = sample.reshape(28, 28)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(img_example, caption="Muestra generada", width=150, clamp=True)

    st.info(f"Muestra aleatoria  --  rango [{sample.min():.0f}, {sample.max():.0f}]")

else:
    if CANVAS_AVAILABLE:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(
                "<p style='font-family:IBM Plex Mono,monospace; font-size:0.70rem;"
                " color:rgba(245,240,232,0.38); letter-spacing:0.12em;"
                " text-transform:uppercase; margin-bottom:0.5rem;'>Dibuja aqui</p>",
                unsafe_allow_html=True
            )
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 1)",
                stroke_width=15,
                stroke_color="#FFFFFF",
                background_color="#000000",
                background_image=None,
                update_streamlit=True,
                height=280,
                width=280,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="canvas_dibujo",
            )

        with col2:
            st.markdown(
                "<p style='font-family:IBM Plex Mono,monospace; font-size:0.70rem;"
                " color:rgba(245,240,232,0.38); letter-spacing:0.12em;"
                " text-transform:uppercase; margin-bottom:0.5rem;'>Procesado 28x28</p>",
                unsafe_allow_html=True
            )
            if canvas_result.image_data is not None:
                img      = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                img_gray = img.convert('L')
                img_28   = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(img_28)
                sample   = img_array.flatten().astype(float)

                st.image(img_28, caption="28x28", width=150, clamp=True)

                pixeles_activos = (sample > 30).sum()
                st.caption(f"Pixeles activos: {pixeles_activos}/784  |  Media: {sample.mean():.1f}")
            else:
                st.warning("Dibuja un digito en el panel izquierdo")
                sample = np.zeros(784)
    else:
        st.error("`streamlit-drawable-canvas` no instalado.")
        st.code("pip install streamlit-drawable-canvas", language="bash")
        sample = np.zeros(784)

# ============================================================
# PREDICCION
# ============================================================
if sample is not None:
    st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        predecir = st.button("Clasificar digito", type="primary", use_container_width=True)

    if predecir:
        sample_norm     = sample / 255.0
        sample_reshaped = sample_norm.reshape(1, -1)
        sample_pca      = pca.transform(sample_reshaped)

        model   = svm_models["rbf"]
        pred    = model.predict(sample_pca)[0]
        cluster = kmeans.predict(sample_pca)[0]

        # ============================================================
        # RESULTADO
        # ============================================================
        st.markdown("<div class='section-tag'>resultado</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.html(f"""
            <div style="
                background: #111111;
                border: 1px solid rgba(232,197,71,0.22);
                padding: 2.5rem 2rem 2rem;
                text-align: center;
                position: relative;
                margin: 0.5rem 0 1.5rem;">

                <div style="
                    position: absolute; top: 0; left: 0; right: 0;
                    height: 2px;
                    background: linear-gradient(90deg, transparent, #e8c547 40%, transparent);">
                </div>

                <p style="
                    font-family: 'IBM Plex Mono', monospace;
                    font-size: 0.55rem;
                    letter-spacing: 0.26em;
                    text-transform: uppercase;
                    color: rgba(232,197,71,0.5);
                    margin: 0 0 0.5rem;">Digito reconocido</p>

                <p style="
                    font-family: 'Cormorant Garamond', serif;
                    font-size: 7.5rem;
                    font-weight: 700;
                    color: #f5f0e8;
                    line-height: 1;
                    margin: 0;
                    letter-spacing: -0.05em;">{pred}</p>

                <p style="
                    font-family: 'IBM Plex Mono', monospace;
                    font-size: 0.56rem;
                    color: rgba(245,240,232,0.22);
                    margin: 0.5rem 0 1.4rem;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;">SVM &middot; Kernel RBF</p>

                <div style="
                    height: 1px;
                    background: rgba(232,197,71,0.1);
                    margin: 0 1rem 1.4rem;">
                </div>

                <p style="
                    font-family: 'IBM Plex Mono', monospace;
                    font-size: 0.56rem;
                    color: rgba(245,240,232,0.35);
                    margin: 0 0 0.3rem;
                    letter-spacing: 0.14em;
                    text-transform: uppercase;">Cluster KMeans</p>

                <p style="
                    font-family: 'Cormorant Garamond', serif;
                    font-size: 2.8rem;
                    font-weight: 600;
                    color: #e8c547;
                    margin: 0;
                    line-height: 1;">{cluster}</p>
            </div>
            """)

            with st.expander("Detalles tecnicos"):
                st.write(f"**Input:** min={sample.min():.1f}, max={sample.max():.1f}, media={sample.mean():.1f}")
                st.write(f"**Normalizado:** min={sample_norm.min():.4f}, max={sample_norm.max():.4f}")
                st.write(f"**PCA:** PC1={sample_pca[0,0]:.4f}, PC2={sample_pca[0,1]:.4f}")
                st.write(f"**Prediccion:** digito={pred}, cluster={cluster}")

st.markdown("<hr class='gold-rule'>", unsafe_allow_html=True)
st.caption("PCA  \u00b7  KMeans  \u00b7  SVM  \u00b7  MNIST Dataset  \u00b7  2026")
