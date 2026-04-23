import hashlib
import html
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import zipfile
import os
from huggingface_hub import hf_hub_download
from absa_inference import (
    ModelBundle,
    hash_texts,
    infer_texts,
    load_model_bundle,
)


APP_TITLE = "ABSA con evidencias para reseñas"

INFERENCE_MAX_LEN = 128
INFERENCE_START_THRESH = 0.35
INFERENCE_END_THRESH = 0.35
INFERENCE_ASPECT_CONF = 0.5
INFERENCE_CATEGORY_CONF = 0.5
INFERENCE_BATCH_SIZE = 8
COL_REVIEW_ID = "reseña_id"
COL_REVIEW_PREVIEW = "reseña"
COL_EVIDENCE_COUNT = "evidencias"
COL_HAS_EVIDENCE = "tiene_evidencia"
COL_SPAN_TEXT = "evidencia"
COL_ASPECT = "aspecto"
COL_CATEGORY = "categoría"
COL_SENTIMENT = "sentimiento"
EVIDENCE_COLUMNS = [
    COL_REVIEW_ID,
    COL_REVIEW_PREVIEW,
    COL_SPAN_TEXT,
    COL_ASPECT,
    COL_CATEGORY,
    COL_SENTIMENT,
]
DETAIL_COLUMNS = [COL_SPAN_TEXT, COL_ASPECT, COL_CATEGORY, COL_SENTIMENT]
REVIEW_COLUMNS = [COL_REVIEW_ID, COL_REVIEW_PREVIEW, COL_EVIDENCE_COUNT, COL_HAS_EVIDENCE]

PALETTE = {
    "burnt_red": "#6d120b",
    "wood_red": "#b02a29",
    "macadamia": "#fbf0d8",
    "royal_blue": "#113047",
    "blue_gray": "#739ab9",
    "white": "#ffffff",
    "black": "#111111",
}

SENTIMENT_COLORS = {
    "POS": "#d4edda",
    "NEU": "#e2e3e5",
    "NEG": "#f8d7da",
}

@st.cache_resource
def load_model():
    model_path = get_model_path()
    bundle = load_model_bundle(model_path)
    return bundle, model_path

def get_model_path():
    zip_path = hf_hub_download(
        repo_id="MateoV12/ABSA_evidence_model",
        filename="evidence_absa_model.pt.zip"
    )

    extract_dir = os.path.dirname(zip_path)

    # Buscar si ya existe el .pt
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".pt"):
                return os.path.join(root, file)

    # Si no existe, extraer
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Buscar de nuevo
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".pt"):
                return os.path.join(root, file)

    raise FileNotFoundError("No se encontró el archivo .pt dentro del zip")

def normalize_model_path(model_path: str) -> str:
    return str(Path(model_path).expanduser().resolve(strict=False))





def build_inference_signature(
    texts: List[str],
    model_path: str,
    max_len: int,
    start_thresh: float,
    end_thresh: float,
    aspect_conf: float,
    cat_conf: float,
    batch_size: int,
) -> str:
    payload = {
        "texts_hash": hash_texts(texts),
        "model_path": normalize_model_path(model_path),
        "max_len": max_len,
        "start_thresh": round(start_thresh, 4),
        "end_thresh": round(end_thresh, 4),
        "aspect_conf": round(aspect_conf, 4),
        "cat_conf": round(cat_conf, 4),
        "batch_size": batch_size,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def preview_text(text: str, limit: int = 90) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def build_highlight_html(text: str, evidences: List[Dict]) -> str:
    spans = []
    seen = set()
    for ev in evidences:
        span = ev.get("char_span")
        if not span or span in seen:
            continue
        seen.add(span)
        spans.append(
            {
                "char_span": span,
                "sentiment": ev.get("sentiment", "NEU"),
            }
        )
    spans.sort(key=lambda x: x["char_span"][0])

    parts = []
    last_idx = 0
    for item in spans:
        s, e = item["char_span"]
        if s < last_idx:
            continue
        parts.append(html.escape(text[last_idx:s]))
        frag = html.escape(text[s:e])
        color = SENTIMENT_COLORS.get(item["sentiment"], PALETTE["macadamia"])
        parts.append(
            f"<span style='background-color:{color}; padding:0 2px; border-radius:3px;'>{frag}</span>"
        )
        last_idx = e
    parts.append(html.escape(text[last_idx:]))
    return "".join(parts)


def build_evidence_table(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(results):
        review_text = item.get("text", "")
        for ev in item.get("evidences", []):
            rows.append(
                {
                    COL_REVIEW_ID: idx,
                    COL_REVIEW_PREVIEW: preview_text(review_text),
                    COL_SPAN_TEXT: ev.get("span_text", ""),
                    COL_ASPECT: ev.get("aspect", ""),
                    COL_CATEGORY: ev.get("category", ""),
                    COL_SENTIMENT: ev.get("sentiment", ""),
                }
            )
    return pd.DataFrame(rows, columns=EVIDENCE_COLUMNS)


def build_review_table(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(results):
        evidence_count = len(item.get("evidences", []))
        rows.append(
            {
                COL_REVIEW_ID: idx,
                COL_REVIEW_PREVIEW: preview_text(item.get("text", "")),
                COL_EVIDENCE_COUNT: evidence_count,
                COL_HAS_EVIDENCE: evidence_count > 0,
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def render_distribution_chart(data: pd.DataFrame, title: str, color: str):
    st.caption(title)
    st.vega_lite_chart(
        data,
        {
            "height": 280,
            "mark": {
                "type": "bar",
                "cornerRadiusTopLeft": 8,
                "cornerRadiusTopRight": 8,
            },
            "encoding": {
                "x": {
                    "field": "etiqueta",
                    "type": "nominal",
                    "sort": "-y",
                    "axis": {
                        "title": None,
                        "labelAngle": -30,
                        "labelLimit": 170,
                    },
                },
                "y": {
                    "field": "conteo",
                    "type": "quantitative",
                    "axis": {"title": "Frecuencia"},
                },
                "color": {"value": color},
                "tooltip": [
                    {"field": "etiqueta", "type": "nominal", "title": title},
                    {"field": "conteo", "type": "quantitative", "title": "Frecuencia"},
                ],
            },
            "config": {
                "background": None,
                "view": {"stroke": None},
                "axis": {
                    "gridColor": "rgba(17, 48, 71, 0.12)",
                    "domainColor": "rgba(17, 48, 71, 0.24)",
                    "tickColor": "rgba(17, 48, 71, 0.24)",
                    "labelColor": PALETTE["royal_blue"],
                    "titleColor": PALETTE["royal_blue"],
                },
            },
        },
        use_container_width=True,
    )





def render_page_chrome():
    st.markdown(
        f"""
        <style>
        /* ─── Variables ─────────────────────────────────────────────── */
        :root {{
            --burnt-red:  {PALETTE["burnt_red"]};
            --wood-red:   {PALETTE["wood_red"]};
            --macadamia:  {PALETTE["macadamia"]};
            --royal-blue: {PALETTE["royal_blue"]};
            --blue-gray:  {PALETTE["blue_gray"]};
            --white:      {PALETTE["white"]};
            --black:      {PALETTE["black"]};
        }}

        /* ─── App base ───────────────────────────────────────────────── */
        .stApp {{
            background: var(--white);
            color: var(--royal-blue);
        }}

        .block-container {{
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }}

        /* ─── Tipografía global ──────────────────────────────────────── */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--royal-blue);
            letter-spacing: -0.02em;
        }}

        p, span, label,
        .stMarkdown, .stCaption, .stText {{
            color: var(--royal-blue);
        }}

        /* ─── Hero card ──────────────────────────────────────────────── */
        .hero-card {{
            padding: 1.85rem 1.95rem;
            border-radius: 28px;
            background: linear-gradient(
                135deg,
                var(--royal-blue) 0%,
                #214b67 46%,
                var(--wood-red) 100%
            );
            color: var(--white);
            margin-bottom: 1.35rem;
            box-shadow: 0 18px 42px rgba(17, 48, 71, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }}

        .hero-card * {{
            color: inherit !important;
        }}
        
        .hero-card h1 {{
            color: var(--white) !important;
            margin: 0 0 0.55rem 0;
            font-size: 2.2rem;
            line-height: 1.06;
        }}

        .hero-card p {{
            margin: 0;
            max-width: 800px;
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.96) !important;
        }}

        /* ─── Métricas ───────────────────────────────────────────────── */
        [data-testid="stMetric"] {{
            background: linear-gradient(
                180deg,
                rgba(17, 48, 71, 0.98) 0%,
                rgba(33, 75, 103, 0.96) 100%
            ) !important;
            border-radius: 20px !important;
            border: 1px solid rgba(115, 154, 185, 0.28) !important;
            padding: 1rem 1.1rem !important;
            box-shadow: 0 14px 30px rgba(17, 48, 71, 0.12) !important;
        }}
        [data-testid="stMetric"] *,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] *,
        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] * {{
            color: #ffffff !important;
        }}

        /* ─── Botones ────────────────────────────────────────────────── */
        .stButton > button,
        .stDownloadButton > button {{
            background: linear-gradient(
                135deg,
                var(--burnt-red) 0%,
                var(--wood-red) 100%
            ) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 0.65rem 1.2rem !important;
            font-weight: 600 !important;
            box-shadow: 0 12px 22px rgba(109, 18, 11, 0.18) !important;
            transition: box-shadow 0.2s ease !important;
        }}
        .stButton > button *,
        .stDownloadButton > button * {{
            color: #ffffff !important;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stButton > button:focus,
        .stDownloadButton > button:focus {{
            color: #ffffff !important;
            box-shadow: 0 14px 24px rgba(176, 42, 41, 0.28) !important;
            outline: none !important;
        }}

        /* ─── File uploader ──────────────────────────────────────────── */
        [data-testid="stFileUploader"] {{
            background: rgba(251, 240, 216, 0.94) !important;
            border: 2px dashed rgba(17, 48, 71, 0.36) !important;
            border-radius: 20px !important;
            padding: 0.5rem !important;
            box-shadow: 0 12px 24px rgba(17, 48, 71, 0.06) !important;
        }}
        [data-testid="stFileUploader"] * {{
            color: var(--royal-blue) !important;
        }}

        /* ─── Labels de inputs ───────────────────────────────────────── */
        .stCheckbox label,
        .stMultiSelect label,
        .stSelectbox label,
        .stTextArea label,
        .stFileUploader label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p {{
            color: var(--royal-blue) !important;
            font-weight: 600 !important;
        }}

        /* ─── Inputs de texto y textarea ─────────────────────────────── */
        textarea,
        input[type="text"],
        input[type="search"] {{
            color: var(--royal-blue) !important;
        }}

        /* ─── Selectbox (contenedor) ─────────────────────────────────── */
        [data-testid="stSelectbox"] > div > div,
        [data-baseweb="select"] > div:first-child {{
            background: var(--macadamia) !important;
            border: 1px solid rgba(17, 48, 71, 0.18) !important;
            border-radius: 18px !important;
            box-shadow: 0 10px 22px rgba(17, 48, 71, 0.05) !important;
        }}

        /* ─── Multiselect contenedor ─────────────────────────────────── */
        [data-testid="stMultiSelect"] > div > div,
        [data-baseweb="select"] > div {{
            background: var(--macadamia) !important;
            border: 1px solid rgba(17, 48, 71, 0.18) !important;
            border-radius: 18px !important;
            box-shadow: 0 10px 22px rgba(17, 48, 71, 0.05) !important;
        }}

        /* Texto escrito dentro del select/multiselect */
        [data-baseweb="select"] input,
        [data-baseweb="select"] [data-baseweb="input"] {{
            color: var(--royal-blue) !important;
            background: transparent !important;
        }}

        /* Placeholder y valor mostrado */
        [data-baseweb="select"] [class*="placeholder"],
        [data-baseweb="select"] [class*="singleValue"] {{
            color: var(--royal-blue) !important;
        }}

        /* Iconos de chevron / clear dentro del select */
        [data-baseweb="select"] svg {{
            color: var(--royal-blue) !important;
            fill: var(--royal-blue) !important;
        }}

        /* ─── Chips / Tags del multiselect ───────────────────────────── */
        /* Selector por atributo de rol para máxima compatibilidad */
        [data-baseweb="tag"],
        [role="option"][aria-selected="true"],
        span[data-baseweb="tag"] {{
            background-color: var(--royal-blue) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 999px !important;
        }}

        /* Todo el contenido dentro del chip: texto e icono × */
        [data-baseweb="tag"] *,
        [data-baseweb="tag"] span,
        [data-baseweb="tag"] svg,
        [data-baseweb="tag"] [role="presentation"] {{
            color: #ffffff !important;
            fill: #ffffff !important;
        }}

        /* Overrides adicionales por si Streamlit inyecta estilos inline */
        [data-baseweb="tag"] > span {{
            color: #ffffff !important;
        }}

        /* ─── Dropdown del select (lista de opciones) ────────────────── */
        [data-baseweb="popover"],
        [data-baseweb="menu"] {{
            background: var(--macadamia) !important;
            border: 1px solid rgba(17, 48, 71, 0.18) !important;
            border-radius: 14px !important;
        }}
        [data-baseweb="menu"] li,
        [data-baseweb="menu"] [role="option"] {{
            color: var(--royal-blue) !important;
        }}
        [data-baseweb="menu"] [role="option"]:hover,
        [data-baseweb="menu"] [aria-selected="true"] {{
            background: rgba(17, 48, 71, 0.08) !important;
        }}

        /* ─── Dataframe ──────────────────────────────────────────────── */
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] > div {{
            border: 1px solid rgba(115, 154, 185, 0.30) !important;
            border-radius: 18px !important;
            overflow: hidden !important;
            box-shadow: 0 10px 24px rgba(17, 48, 71, 0.05) !important;
        }}

        /* ─── Notificaciones / alertas ───────────────────────────────── */
        [data-baseweb="notification"],
        [data-testid="stAlert"] {{
            background: rgba(251, 240, 216, 0.96) !important;
            border: 1px solid rgba(17, 48, 71, 0.18) !important;
            border-radius: 14px !important;
            color: var(--royal-blue) !important;
        }}
        [data-baseweb="notification"] *,
        [data-testid="stAlert"] * {{
            color: var(--royal-blue) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero-card">
            <h1>ABSA con evidencias para reseñas</h1>
            <p>
                Explora los resultados del análisis en una interfaz limpia, con métricas globales,
                distribuciones por etiqueta y detalle resaltado de las evidencias encontradas en cada reseña.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    with st.spinner("Cargando modelo..."):
        try:
            bundle, model_path = load_model()
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")
            return

    render_page_chrome()

    st.subheader("Entrada de datos")
    st.caption("Carga un CSV con columna `text` o pega una reseña por línea.")
    uploaded = st.file_uploader("Sube un archivo CSV", type=["csv"])
    text_area = st.text_area("O pega reseñas")

    texts: List[str] = []
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("El CSV debe contener una columna llamada `text`.")
            return
        texts = df["text"].astype(str).tolist()
    elif text_area.strip():
        texts = [line.strip() for line in text_area.splitlines() if line.strip()]

    if not texts:
        st.info("Carga un CSV o pega reseñas para continuar.")
        return

    st.write(f"Total de reseñas cargadas: {len(texts)}")

    inference_signature = build_inference_signature(
        texts,
        model_path=model_path,
        max_len=INFERENCE_MAX_LEN,
        start_thresh=INFERENCE_START_THRESH,
        end_thresh=INFERENCE_END_THRESH,
        aspect_conf=INFERENCE_ASPECT_CONF,
        cat_conf=INFERENCE_CATEGORY_CONF,
        batch_size=INFERENCE_BATCH_SIZE,
    )

    results = None
    if st.session_state.get("results_signature") == inference_signature:
        results = st.session_state.get("results_data")

    if st.button("Analizar reseñas"):
        

        with st.spinner("Procesando reseñas..."):
            results = infer_texts(
                texts,
                bundle,
                batch_size=INFERENCE_BATCH_SIZE,
                max_len=INFERENCE_MAX_LEN,
                start_thresh=INFERENCE_START_THRESH,
                end_thresh=INFERENCE_END_THRESH,
                aspect_conf_thresh=INFERENCE_ASPECT_CONF,
                cat_conf_thresh=INFERENCE_CATEGORY_CONF,
            )

        st.session_state["results_signature"] = inference_signature
        st.session_state["results_data"] = results

    if results is None:
        st.warning("Haz clic en `Analizar reseñas` para visualizar los resultados.")
        return

    review_df = build_review_table(results)
    evidence_df = build_evidence_table(results)

    total_evidences = len(evidence_df)
    avg_evidences = total_evidences / len(results) if results else 0.0
    reviews_with_evidence = int(review_df[COL_HAS_EVIDENCE].sum()) if not review_df.empty else 0
    coverage = (reviews_with_evidence / len(results)) if results else 0.0

    st.subheader("Resumen global")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reseñas", f"{len(results)}")
    col2.metric("Con evidencias", f"{reviews_with_evidence}")
    col3.metric("Evidencias", f"{total_evidences}")
    col4.metric("Cobertura", f"{coverage:.1%}")

    st.caption(f"Promedio de evidencias por reseña: {avg_evidences:.2f}")

    if not evidence_df.empty:
        c1, c2, c3 = st.columns(3)
        sent_counts = (
            evidence_df[COL_SENTIMENT]
            .value_counts()
            .rename_axis("etiqueta")
            .reset_index(name="conteo")
        )
        asp_counts = (
            evidence_df[COL_ASPECT]
            .value_counts()
            .rename_axis("etiqueta")
            .reset_index(name="conteo")
        )
        cat_counts = (
            evidence_df[COL_CATEGORY]
            .value_counts()
            .rename_axis("etiqueta")
            .reset_index(name="conteo")
        )
        with c1:
            render_distribution_chart(sent_counts, "Sentimiento", PALETTE["wood_red"])
        with c2:
            render_distribution_chart(asp_counts, "Aspecto", PALETTE["blue_gray"])
        with c3:
            render_distribution_chart(cat_counts, "Categoría", PALETTE["burnt_red"])
    else:
        st.info("No se encontraron evidencias en las reseñas cargadas.")

    st.subheader("Resumen por reseña")
    st.dataframe(review_df, width="stretch")

    st.subheader("Tabla de evidencias")
    if evidence_df.empty:
        st.write("No hay evidencias para mostrar.")
    else:
        sentiments = sorted(evidence_df[COL_SENTIMENT].unique().tolist())
        aspects = sorted(evidence_df[COL_ASPECT].unique().tolist())
        categories = sorted(evidence_df[COL_CATEGORY].unique().tolist())

        f_sent = st.multiselect("Filtrar por sentimiento", sentiments, default=sentiments)
        f_asp = st.multiselect("Filtrar por aspecto", aspects, default=aspects)
        f_cat = st.multiselect("Filtrar por categoría", categories, default=categories)

        filtered = evidence_df[
            evidence_df[COL_SENTIMENT].isin(f_sent)
            & evidence_df[COL_ASPECT].isin(f_asp)
            & evidence_df[COL_CATEGORY].isin(f_cat)
        ]
        st.dataframe(filtered, width="stretch")
        st.download_button(
            "Descargar evidencias filtradas en CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="absa_evidencias_filtradas.csv",
            mime="text/csv",
        )

    st.subheader("Detalle por reseña")
    only_with_evidence = st.checkbox("Mostrar solo reseñas con evidencias", value=False)
    selectable_reviews = review_df[review_df[COL_HAS_EVIDENCE]] if only_with_evidence else review_df

    if selectable_reviews.empty:
        st.info("No hay reseñas que cumplan el filtro actual.")
    else:
        options = [
            f"{int(row[COL_REVIEW_ID])}: {row[COL_REVIEW_PREVIEW]}"
            for _, row in selectable_reviews.iterrows()
        ]
        selected_option = st.selectbox("Selecciona una reseña", options)
        selected_idx = int(selected_option.split(":", 1)[0])
        selected_item = results[selected_idx]

        st.caption("Leyenda: POS (verde), NEU (gris), NEG (rojo)")
        highlighted_text = build_highlight_html(selected_item["text"], selected_item.get("evidences", []))
        st.markdown(f"<div style='line-height:1.8'>{highlighted_text}</div>", unsafe_allow_html=True)

        detail_rows = []
        for ev in selected_item.get("evidences", []):
            detail_rows.append(
                {
                    COL_SPAN_TEXT: ev.get("span_text", ""),
                    COL_ASPECT: ev.get("aspect", ""),
                    COL_CATEGORY: ev.get("category", ""),
                    COL_SENTIMENT: ev.get("sentiment", ""),
                }
            )
        st.dataframe(pd.DataFrame(detail_rows, columns=DETAIL_COLUMNS), width="stretch")

    st.download_button(
        "Descargar resultados completos en JSON",
        data=json.dumps(results, ensure_ascii=False, indent=2),
        file_name="absa_resultados.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()