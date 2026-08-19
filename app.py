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
from ui_styles import (
    CHART_CATEGORICAL,
    PALETTE,
    SENTIMENT_BG,
    SENTIMENT_COLORS,
    STYLES_HTML,
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
COL_REVIEW_TEXT = "reseña_completa"
COL_EVIDENCE_COUNT = "evidencias"
COL_HAS_EVIDENCE = "tiene_evidencia"
COL_SPAN_TEXT = "evidencia"
COL_CHAR_START = "inicio_span"
COL_CHAR_END = "fin_span"
COL_ASPECT = "aspecto"
COL_CATEGORY = "categoría"
COL_SENTIMENT = "sentimiento"
EVIDENCE_COLUMNS = [
    COL_REVIEW_ID,
    COL_REVIEW_TEXT,
    COL_REVIEW_PREVIEW,
    COL_SPAN_TEXT,
    COL_CHAR_START,
    COL_CHAR_END,
    COL_ASPECT,
    COL_CATEGORY,
    COL_SENTIMENT,
]
REVIEW_COLUMNS = [COL_REVIEW_ID, COL_REVIEW_PREVIEW, COL_EVIDENCE_COUNT, COL_HAS_EVIDENCE]

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
                "aspect": ev.get("aspect", ""),
                "category": ev.get("category", ""),
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
        bg = SENTIMENT_BG.get(item["sentiment"], PALETTE["navy_soft"])
        accent = SENTIMENT_COLORS.get(item["sentiment"], PALETTE["royal_blue"])
        title = f"{item['aspect']} · {item['category']} · {item['sentiment']}"
        parts.append(
            f"<span class='ev-highlight' title='{title}' "
            f"style='background-color:{bg}; border-color:{accent}; color:{accent};'>{frag}</span>"
        )
        last_idx = e
    parts.append(html.escape(text[last_idx:]))
    return "".join(parts)


def render_sentiment_legend() -> str:
    chips = []
    for sent in ["POS", "NEU", "NEG"]:
        bg = SENTIMENT_BG.get(sent, PALETTE["navy_soft"])
        accent = SENTIMENT_COLORS.get(sent, PALETTE["royal_blue"])
        chips.append(
            f"<span class='legend-chip'><span class='legend-dot' "
            f"style='background-color:{accent};'></span>{sent}</span>"
        )
    return (
        "<div style='margin-bottom:0.6rem;'>"
        + "".join(chips)
        + "</div>"
    )


def build_sentiment_badge(sent: str) -> str:
    bg = SENTIMENT_BG.get(sent, PALETTE["navy_soft"])
    accent = SENTIMENT_COLORS.get(sent, PALETTE["royal_blue"])
    return f"<span class='sent-badge' style='background-color:{bg}; color:{accent};'>{sent}</span>"


def render_status_note(message: str) -> None:
    icon = (
        "<svg width='18' height='18' viewBox='0 0 24 24' fill='none' "
        "stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
        "<circle cx='12' cy='12' r='5'/>"
        "<path d='M12 16v-4M12 8h.01'/>"
        "</svg>"
    )
    st.markdown(
        f"<div class='status-note'>{icon}<span>{html.escape(message)}</span></div>",
        unsafe_allow_html=True,
    )


def render_evidence_cards(evidences: List[Dict]) -> None:
    if not evidences:
        st.info("Esta reseña no tiene evidencias detectadas.")
        return
    cards = []
    for ev in evidences:
        badge = build_sentiment_badge(ev.get("sentiment", "NEU"))
        span = html.escape(ev.get("span_text", ""))
        aspect = html.escape(ev.get("aspect", ""))
        category = html.escape(ev.get("category", ""))
        cards.append(
            "<div class='evidence-card'>"
            f"{badge}"
            "<div>"
            f"<div class='ev-text'>{span}</div>"
            f"<div class='ev-meta'><b>Aspecto:</b> {aspect} &nbsp;·&nbsp; <b>Categoría:</b> {category}</div>"
            "</div>"
            "</div>"
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_metric_ring(value: float, label: str, sublabel: str = "", color: str = None, large: bool = False) -> str:
    pct = max(0.0, min(100.0, value * 100.0))
    color = color or PALETTE["royal_blue"]
    size_cls = "ring-lg" if large else "ring-sm"
    sub = f"<div class='ring-sublabel'>{sublabel}</div>" if sublabel else ""
    return (
        "<div class='ring-wrap'>"
        f"<div class='ring {size_cls}' style='--pct:{pct:.1f}; --ring-color:{color};'>"
        f"<div class='ring-inner'><div class='ring-value'>{pct:.1f}%</div></div>"
        "</div>"
        f"<div class='ring-label'>{label}</div>{sub}"
        "</div>"
    )


def render_f1_table(headers: List[str], rows: List[List[str]], num_cols: List[int]) -> str:
    thead = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cls = " num" if i in num_cols else ""
            cells.append(f"<td class='{cls.strip()}'>{cell}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='f1-table'><thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def highlight_span_in_text(text: str, span_text: str, sentiment: str) -> str:
    if not span_text:
        return html.escape(text)
    idx = text.lower().find(span_text.lower())
    if idx < 0:
        return html.escape(text)
    accent = SENTIMENT_COLORS.get(sentiment, PALETTE["royal_blue"])
    bg = SENTIMENT_BG.get(sentiment, PALETTE["navy_soft"])
    frag = html.escape(text[idx : idx + len(span_text)])
    hl = (
        f"<span class='ev-highlight' style='background-color:{bg}; "
        f"border-color:{accent}; color:{accent};'>{frag}</span>"
    )
    return html.escape(text[:idx]) + hl + html.escape(text[idx + len(span_text) :])


def highlight_span_offsets(text: str, s: int, e: int, sentiment: str) -> str:
    if not text or s is None or e is None or s < 0 or e <= s or s >= len(text):
        return html.escape(text)
    e = min(e, len(text))
    accent = SENTIMENT_COLORS.get(sentiment, PALETTE["royal_blue"])
    bg = SENTIMENT_BG.get(sentiment, PALETTE["navy_soft"])
    frag = html.escape(text[s:e])
    hl = (
        f"<span class='ev-highlight' style='background-color:{bg}; "
        f"border-color:{accent}; color:{accent};'>{frag}</span>"
    )
    return html.escape(text[:s]) + hl + html.escape(text[e:])


def render_evidence_table_html(df: pd.DataFrame) -> str:
    thead = "".join(f"<th>{h}</th>" for h in ["Reseña", "Aspecto", "Categoría", "Sentimiento"])
    body = []
    for _, row in df.iterrows():
        review_text = str(row[COL_REVIEW_TEXT])
        s = int(row[COL_CHAR_START]) if pd.notna(row[COL_CHAR_START]) else -1
        e = int(row[COL_CHAR_END]) if pd.notna(row[COL_CHAR_END]) else -1
        if s >= 0 and e > s:
            review_html = highlight_span_offsets(review_text, s, e, row[COL_SENTIMENT])
        else:
            review_html = highlight_span_in_text(
                str(row[COL_REVIEW_PREVIEW]),
                str(row[COL_SPAN_TEXT]),
                row[COL_SENTIMENT],
            )
        badge = build_sentiment_badge(row[COL_SENTIMENT])
        body.append(
            "<tr>"
            f"<td>{review_html}</td>"
            f"<td>{html.escape(str(row[COL_ASPECT]))}</td>"
            f"<td>{html.escape(str(row[COL_CATEGORY]))}</td>"
            f"<td>{badge}</td>"
            "</tr>"
        )
    return (
        f"<table class='ev-table'><thead><tr>{thead}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def build_evidence_table(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(results):
        review_text = item.get("text", "")
        for ev in item.get("evidences", []):
            char_span = ev.get("char_span")
            start = int(char_span[0]) if char_span and len(char_span) >= 2 else -1
            end = int(char_span[1]) if char_span and len(char_span) >= 2 else -1
            rows.append(
                {
                    COL_REVIEW_ID: idx,
                    COL_REVIEW_TEXT: review_text,
                    COL_REVIEW_PREVIEW: preview_text(review_text),
                    COL_SPAN_TEXT: ev.get("span_text", ""),
                    COL_CHAR_START: start,
                    COL_CHAR_END: end,
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


def render_distribution_chart(
    data: pd.DataFrame,
    title: str,
    color: str = None,
    color_map: Dict[str, str] = None,
    color_field: str = "etiqueta",
    value_field: str = "conteo",
    value_title: str = "Frecuencia",
    height: int = 280,
    horizontal: bool = False,
    value_format: str = ".0f",
    show_values: bool = True,
    legend: bool = False,
):
    import altair as alt

    chart_data = data.copy()
    if value_field not in chart_data.columns:
        chart_data[value_field] = 0.0
    if color_field not in chart_data.columns:
        chart_data[color_field] = ""

    y_sort = alt.SortField(value_field, order="descending")
    x_sort = alt.SortField(value_field, order="descending")

    if horizontal:
        base = alt.Chart(chart_data).mark_bar(
            cornerRadiusTopRight=6,
            cornerRadiusBottomRight=6,
        ).encode(
            x=alt.X(value_field, title=value_title, axis=alt.Axis(format=value_format, grid=False)),
            y=alt.Y("etiqueta", title=None, sort=x_sort, axis=alt.Axis(labelLimit=220, grid=False)),
            tooltip=[
                alt.Tooltip("etiqueta", title=title),
                alt.Tooltip(value_field, title=value_title, format=value_format),
            ],
        )
    else:
        base = alt.Chart(chart_data).mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
        ).encode(
            x=alt.X("etiqueta", title=None, sort=y_sort, axis=alt.Axis(labelAngle=-30, labelLimit=190, grid=False)),
            y=alt.Y(value_field, title=value_title, axis=alt.Axis(format=value_format, grid=False)),
            tooltip=[
                alt.Tooltip("etiqueta", title=title),
                alt.Tooltip(value_field, title=value_title, format=value_format),
            ],
        )

    if color_map:
        legend_spec = None
        if legend:
            legend_spec = alt.Legend(
                title=None,
                orient="bottom",
                labelFont="Inter, sans-serif",
                labelColor=PALETTE["royal_blue"],
                labelFontSize=12,
                symbolSize=90,
                symbolType="square",
            )
        base = base.encode(
            color=alt.Color(
                f"{color_field}:N",
                scale=alt.Scale(
                    domain=list(color_map.keys()),
                    range=list(color_map.values()),
                ),
                legend=legend_spec,
            )
        )
    elif legend:
        labels = chart_data[color_field].dropna().unique().tolist()
        color_map = {
            label: CHART_CATEGORICAL[i % len(CHART_CATEGORICAL)]
            for i, label in enumerate(labels)
        }
        base = base.encode(
            color=alt.Color(
                f"{color_field}:N",
                scale=alt.Scale(
                    domain=list(color_map.keys()),
                    range=list(color_map.values()),
                ),
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                    labelFont="Inter, sans-serif",
                    labelColor=PALETTE["royal_blue"],
                    labelFontSize=12,
                    symbolSize=90,
                    symbolType="square",
                ),
            )
        )
    else:
        base = base.encode(color=alt.value(color or PALETTE["royal_blue"]))

    chart = base.properties(
        height=height,
        title=alt.TitleParams(
            title,
            anchor="start",
            fontSize=15,
            fontWeight=700,
            color=PALETTE["royal_blue"],
            dy=8,
        ),
    ).configure_view(stroke=None)

    if show_values and not chart_data.empty:
        text_color = PALETTE["royal_blue"]
        if horizontal:
            text_layer = (
                alt.Chart(chart_data)
                .mark_text(
                    align="left",
                    baseline="middle",
                    dx=6,
                    color=text_color,
                    fontSize=12,
                    fontWeight=700,
                )
                .encode(
                    x=alt.X(value_field, title=None, axis=None),
                    y=alt.Y("etiqueta", title=None, sort=x_sort, axis=None),
                    text=alt.Text(value_field, format=value_format),
                )
            )
        else:
            text_layer = (
                alt.Chart(chart_data)
                .mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-6,
                    color=text_color,
                    fontSize=12,
                    fontWeight=700,
                )
                .encode(
                    x=alt.X("etiqueta", title=None, sort=y_sort, axis=None),
                    y=alt.Y(value_field, title=None, axis=None),
                    text=alt.Text(value_field, format=value_format),
                )
            )
        chart = alt.layer(base, text_layer)

    chart = chart.configure_axis(
        labelColor=PALETTE["royal_blue"],
        titleColor=PALETTE["royal_blue"],
        domainColor="rgba(17, 48, 71, 0.22)",
        tickColor="rgba(17, 48, 71, 0.22)",
        labelFont="Inter, sans-serif",
        titleFont="Inter, sans-serif",
    ).configure_title(
        color=PALETTE["royal_blue"],
        font="Inter, sans-serif",
    ).configure_axisY(grid=False).configure_axisX(grid=False)

    st.altair_chart(chart, width="stretch")


def fmt_pct(value: float) -> str:
    return f"{value*100:.1f}%"


def best_worst_label(per_class: Dict) -> tuple:
    items = [(k, v.get("f1", 0.0)) for k, v in per_class.items()]
    if not items:
        return None, None
    best = max(items, key=lambda kv: kv[1])
    worst = min(items, key=lambda kv: kv[1])
    return best, worst


def build_aspect_insight(block: Dict) -> str:
    best, worst = best_worst_label(block.get("per_class", {}))
    if not best or best[0] == worst[0]:
        return ""
    return (
        f"El modelo identifica mejor el aspecto «{best[0]}» "
        f"(F1 {best[1]:.2f}) que «{worst[0]}» (F1 {worst[1]:.2f})."
    )


def build_category_insight(block: Dict) -> str:
    best, worst = best_worst_label(block.get("per_class", {}))
    if not best or best[0] == worst[0]:
        return ""
    return (
        f"Mayor precisión en la categoría «{best[0]}» "
        f"(F1 {best[1]:.2f}); el reto está en «{worst[0]}» (F1 {worst[1]:.2f})."
    )


def load_cv_results(path: str = "cv5_results_v12.json") -> Dict:
    try:
        import os as _os

        base = _os.path.dirname(_os.path.abspath(__file__))
        full = _os.path.join(base, path)
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.warning(
            f"No se pudo cargar `{path}` con métricas de CV: {exc}. "
            "Ejecuta `python absa_evidence_model.py --cv_folds 5 --cv_output_json cv5_results_v12.json` "
            "para generarlo."
        )
        return {}





def render_page_chrome(cv_data: Dict):
    st.markdown(STYLES_HTML, unsafe_allow_html=True)

    try:
        agg = cv_data.get("aggregate", {}).get("mean", {})
        hero_f1 = f"{agg.get('triplet_strict', {}).get('f1', 0.0) * 100:.1f}%"
    except Exception:
        hero_f1 = "52.2%"

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-badge">Trabajo de grado · NLP en español</div>
            <h1>ABSA con evidencias para reseñas</h1>
            <p>
                Modelo de <b>análisis de sentimiento por aspectos</b> (span-level) sobre reseñas de
                apps en Colombia. Detecta la evidencia textual de cada opinión y la clasifica en
                aspecto, categoría y sentimiento a partir de un Transformer de 110 millones de parámetros ajustado mediante fine-tuning. Funciona con un dataset creado para este propósito.
            </p>
            <div class="hero-metrics">
                <div class="hero-metric"><b>3.493</b><span>reseñas etiquetadas</span></div>
                <div class="hero-metric"><b>14</b><span>aspectos</span></div>
                <div class="hero-metric"><b>6</b><span>categorías</span></div>
                <div class="hero-metric"><b>{hero_f1}</b><span>Rendimiento (F1 cuádrupla estricta)</span></div>
            </div>
            <div class="hero-links">
                <a class="hero-link" href="https://noesis.uis.edu.co/handle/20.500.14071/47741" target="_blank" rel="noopener">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 4-8 4-8-4 8-4z"/><path d="M4 17v4l8 4 8-4v-4"/><path d="M4 13v4l8 4 8-4v-4"/></svg>
                    Documento del trabajo de grado
                </a>
                <a class="hero-link" href="https://github.com/MateoVera12/ABSA-model-for-processing-reviews-in-Spanish-from-Colombia.git" target="_blank" rel="noopener">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>
                    Repositorio en GitHub
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=":material/analytics:", layout="wide")

    cv_data = load_cv_results()
    render_page_chrome(cv_data)

    tab_demo, tab_model = st.tabs(
        ["Demo interactiva", "Rendimiento del modelo"]
    )
    with tab_demo:
        render_demo_tab()
    with tab_model:
        render_model_tab(cv_data)


def render_model_tab(cv_data: Dict):
    st.markdown("#### Rendimiento del modelo")
    st.markdown(
        "<p>Resultados de <b>cross-validation de 5 folds</b> sobre el dataset de "
        "3493 reseñas etiquetadas de apps bancarias.</p>",
        unsafe_allow_html=True,
    )

    if not cv_data:
        return

    agg = cv_data.get("aggregate", {})
    config = cv_data.get("config", {})
    mean = agg.get("mean", {})
    std = agg.get("std", {})

    span_iou = mean.get("span_iou", {})
    triplet = mean.get("triplet_strict", {})

    aspect_macro = mean.get("aspect", {}).get("macro", {})
    category_macro = mean.get("category", {}).get("macro", {})
    sentiment_macro = mean.get("sentiment", {}).get("macro", {})
    pair_macro = mean.get("pair_cat_sent", {}).get("macro", {})

    hero_f1 = triplet.get("f1", 0.0)
    st.markdown(
        f"<div style='display:flex; justify-content:center; margin:1.2rem 0 0.6rem 0;'>"
        f"{render_metric_ring(hero_f1, 'Cuadrupla estricta', 'aspecto + categoría + sentimiento + span coinciden a la vez', color=PALETTE['wood_red'], large=True)}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "F1 de la cuádrupla estricta: el span detectado (IoU ≥ 0.5) y las etiquetas "
        "de aspecto, categoría y sentimiento coinciden simultáneamente con la evidencia real."
    )

    support_metrics = [
        ("F1 Aspecto", aspect_macro.get("f1", 0.0)),
        ("F1 Sentimiento", sentiment_macro.get("f1", 0.0)),
        ("F1 Categoría", category_macro.get("f1", 0.0)),
        ("F1 Detección de span", span_iou.get("f1", 0.0)),
    ]
    ring_columns = st.columns(4, vertical_alignment="center")
    for col, (label, f1) in zip(ring_columns, support_metrics):
        with col:
            st.markdown(
                f"<div style='display:flex; justify-content:center;'>"
                f"{render_metric_ring(f1, label, '', color=PALETTE['royal_blue'], large=False)}"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.caption(
        "F1 de span con IoU ≥ 0.5 respecto a la evidencia real. Aspecto, sentimiento "
        "y categoría se evalúan condicionados a que el span detectado tenga IoU ≥ 0.5."
    )

    st.markdown("##### Desglose por clase")
    st.markdown(
        "<p>Explora el detalle fino de F1 por aspecto, categoría y cobertura "
        "categoría × sentimiento.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("Ver detalle completo"):
        def per_class_rows(block):
            per_class = block.get("per_class", {})
            items = sorted(
                ((k, v.get("f1", 0.0)) for k, v in per_class.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )
            return [(k, f"{v:.2f}") for k, v in items]

        aspect_rows = per_class_rows(aspect_macro)
        category_rows = per_class_rows(category_macro)

        col_a, col_c = st.columns(2)
        with col_a:
            st.markdown("**F1 por aspecto**")
            insight = build_aspect_insight(aspect_macro)
            if insight:
                st.caption(insight)
            st.markdown(
                render_f1_table(["Aspecto", "F1"], aspect_rows, num_cols=[1]),
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown("**F1 por categoría**")
            insight = build_category_insight(category_macro)
            if insight:
                st.caption(insight)
            st.markdown(
                render_f1_table(["Categoría", "F1"], category_rows, num_cols=[1]),
                unsafe_allow_html=True,
            )

        st.markdown("**Cobertura categoría × sentimiento**")
        pair_rows = []
        per_class = pair_macro.get("per_class", {})
        sorted_pairs = sorted(
            per_class.items(),
            key=lambda kv: kv[1].get("f1", 0.0),
            reverse=True,
        )
        for label, vals in sorted_pairs:
            category, _, sent = label.partition("|||")
            badge = build_sentiment_badge(sent) if sent else "—"
            pair_rows.append([category, badge, f"{vals.get('f1', 0.0):.2f}"])
        st.markdown(
            render_f1_table(["Categoría", "Sentimiento", "F1"], pair_rows, num_cols=[2]),
            unsafe_allow_html=True,
        )

        st.markdown("##### Metadatos de entrenamiento")
        config_rows = [
            {"parámetro": "Modelo base", "valor": config.get("model_name", "")},
            {"parámetro": "Pliegues CV", "valor": str(config.get("cv_folds", ""))},
            {"parámetro": "Batch size", "valor": str(config.get("batch_size", ""))},
            {"parámetro": "Max length", "valor": str(config.get("max_len", ""))},
            {"parámetro": "Learning rate", "valor": str(config.get("learning_rate", ""))},
            {"parámetro": "Épocas", "valor": str(config.get("num_epochs", ""))},
            {"parámetro": "Semilla", "valor": str(config.get("seed", ""))},
        ]
        st.dataframe(
            pd.DataFrame(config_rows, columns=["parámetro", "valor"]),
            width="stretch",
            hide_index=True,
        )


def render_demo_tab():
    with st.spinner("Cargando modelo..."):
        try:
            bundle, model_path = load_model()
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")
            return

    with st.container(border=True):
        st.subheader("Entrada de datos")
        st.caption("Elige el formato de entrada: sube un CSV con columna `text` o pega una reseña por línea.")

        input_mode = st.segmented_control(
            "Formato de entrada",
            options=["Archivo CSV", "Texto"],
            default="Archivo CSV",
            label_visibility="collapsed",
        )

        texts: List[str] = []
        if input_mode == "Archivo CSV":
            uploaded = st.file_uploader("Sube un archivo CSV", type=["csv"], label_visibility="collapsed")
            if uploaded is not None:
                df = pd.read_csv(uploaded)
                if "text" not in df.columns:
                    st.error("El CSV debe contener una columna llamada `text`.")
                    return
                texts = df["text"].astype(str).tolist()
        else:
            text_area = st.text_area(
                "Pega reseñas",
                placeholder="Escribe una reseña por línea...",
                label_visibility="collapsed",
            )
            if text_area.strip():
                texts = [line.strip() for line in text_area.splitlines() if line.strip()]

        if not texts:
            render_status_note("Carga un CSV o pega reseñas para continuar.")

        analyze_clicked = st.button("Analizar reseñas", type="primary", disabled=not bool(texts))

    if not texts:
        return

    st.caption(f"Total de reseñas cargadas: {len(texts)}")

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

    if analyze_clicked:

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

    with st.container(border=True):
        st.subheader("Resumen global")
        with st.container(horizontal=True):
            st.metric("Reseñas", f"{len(results)}", border=True)
            st.metric("Con evidencias", f"{reviews_with_evidence}", border=True)
            st.metric("Evidencias", f"{total_evidences}", border=True)
            st.metric("Cobertura", f"{coverage:.1%}", border=True)

        st.caption(f"Promedio de evidencias por reseña: {avg_evidences:.2f}")

        if not evidence_df.empty:
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

            st.markdown("##### Distribución de sentimiento")
            render_distribution_chart(
                sent_counts, "Sentimiento",
                color_map={
                    "POS": PALETTE["green"],
                    "NEU": PALETTE["soft_blue"],
                    "NEG": PALETTE["wood_red"],
                },
                value_field="conteo", value_title="Evidencias",
                height=260,
                legend=True,
            )

            col_chart_1, col_chart_2 = st.columns(2)
            with col_chart_1:
                render_distribution_chart(
                    asp_counts, "Aspecto",
                    value_field="conteo", value_title="Evidencias",
                    height=220,
                    legend=True,
                    horizontal=True,
                )
            with col_chart_2:
                render_distribution_chart(
                    cat_counts, "Categoría",
                    value_field="conteo", value_title="Evidencias",
                    height=220,
                    legend=True,
                    horizontal=True,
                )
        else:
            st.info("No se encontraron evidencias en las reseñas cargadas.")

    with st.container(border=True):
        st.subheader("Resumen por reseña")
        review_display = review_df[[COL_REVIEW_PREVIEW, COL_EVIDENCE_COUNT]].rename(
            columns={
                COL_REVIEW_PREVIEW: "Reseña",
                COL_EVIDENCE_COUNT: "Número de evidencias",
            }
        )
        st.dataframe(review_display, width="stretch")

    with st.container(border=True):
        st.subheader("Tabla de evidencias")
        if evidence_df.empty:
            st.write("No hay evidencias para mostrar.")
        else:
            aspects = sorted(evidence_df[COL_ASPECT].unique().tolist())
            categories = sorted(evidence_df[COL_CATEGORY].unique().tolist())

            sent_order = ["POS", "NEU", "NEG"]
            sent_options = sent_order

            if st.session_state.pop("reset_filters", False):
                st.session_state.pop("filtro_sent", None)
                st.session_state.pop("filtro_asp", None)
                st.session_state.pop("filtro_cat", None)
                st.rerun()

            if st.session_state.get("filtro_asp"):
                asp_label = f"Aspecto: {len(st.session_state['filtro_asp'])} seleccionados"
            else:
                asp_label = f"Aspecto: todos ({len(aspects)})"
            if st.session_state.get("filtro_cat"):
                cat_label = f"Categoría: {len(st.session_state['filtro_cat'])} seleccionados"
            else:
                cat_label = f"Categoría: todos ({len(categories)})"

            filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
            with filter_col1:
                f_sent = st.segmented_control(
                    "Sentimiento",
                    options=sent_options,
                    selection_mode="multi",
                    default=sent_options,
                    label_visibility="collapsed",
                    key="filtro_sent",
                )
            with filter_col2:
                f_asp = st.multiselect(
                    asp_label,
                    aspects,
                    default=[],
                    placeholder=f"Aspecto: todos ({len(aspects)})",
                    key="filtro_asp",
                )
            with filter_col3:
                f_cat = st.multiselect(
                    cat_label,
                    categories,
                    default=[],
                    placeholder=f"Categoría: todos ({len(categories)})",
                    key="filtro_cat",
                )

            cond = evidence_df[COL_SENTIMENT].isin(f_sent)
            if f_asp:
                cond &= evidence_df[COL_ASPECT].isin(f_asp)
            if f_cat:
                cond &= evidence_df[COL_CATEGORY].isin(f_cat)
            filtered = evidence_df[cond]

            filters_active = (
                len(f_sent) != len(sent_options) or len(f_asp) > 0 or len(f_cat) > 0
            )

            st.caption(f"Mostrando **{len(filtered)}** de **{len(evidence_df)}** evidencias")
            if filters_active:
                if st.button("Limpiar filtros", type="secondary"):
                    st.session_state["reset_filters"] = True
                    st.rerun()

            if filtered.empty:
                st.info("No hay evidencias que coincidan con los filtros.")
            else:
                st.markdown(
                    f"<div class='ev-table-scroll'>{render_evidence_table_html(filtered)}</div>",
                    unsafe_allow_html=True,
                )

            st.download_button(
                "Descargar evidencias filtradas en CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="absa_evidencias_filtradas.csv",
                mime="text/csv",
            )

    with st.container(border=True):
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

            st.markdown(render_sentiment_legend(), unsafe_allow_html=True)
            highlighted_text = build_highlight_html(selected_item["text"], selected_item.get("evidences", []))
            st.markdown(f"<div style='line-height:1.8; font-size:1.02rem'>{highlighted_text}</div>", unsafe_allow_html=True)

            render_evidence_cards(selected_item.get("evidences", []))

    st.download_button(
        "Descargar resultados completos en JSON",
        data=json.dumps(results, ensure_ascii=False, indent=2),
        file_name="absa_resultados.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
