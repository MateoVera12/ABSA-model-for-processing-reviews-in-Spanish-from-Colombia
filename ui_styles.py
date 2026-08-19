"""Tokens de diseno y estilos CSS de la aplicacion ABSA Evidence."""

PALETTE = {
    "burnt_red": "#6d120b",
    "wood_red": "#b02a29",
    "macadamia": "#fbf0d8",
    "royal_blue": "#113047",
    "blue_gray": "#4a6b83",
    "green": "#3f7a50",
    "green_soft": "#e6f0e9",
    "navy_soft": "#e8ecf0",
    "red_soft": "#f9e9e8",
    "warm_gray": "#f7f5f2",
    "mist": "#eef1f5",
    "soft_blue": "#41698a",
    "white": "#ffffff",
    "black": "#111111",
}

SENTIMENT_COLORS = {
    "POS": "#3f7a50",
    "NEU": "#41698a",
    "NEG": "#b02a29",
}

SENTIMENT_BG = {
    "POS": "#e6f0e9",
    "NEU": "#e4edf4",
    "NEG": "#f9e9e8",
}

CHART_CATEGORICAL = [
    "#113047",
    "#4a6b83",
    "#2f5d7a",
    "#86a4ba",
    "#345e77",
    "#a9c2d4",
    "#5c7d96",
    "#9bb5ca",
    "#6b8aa3",
    "#b3c8d8",
]

STYLES_HTML = f"""
<style>
        /* ─── Tipografía global (Google Fonts) ─────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        :root {{
            --burnt-red:   {PALETTE["burnt_red"]};
            --wood-red:    {PALETTE["wood_red"]};
            --macadamia:   {PALETTE["macadamia"]};
            --royal-blue:  {PALETTE["royal_blue"]};
            --blue-gray:   {PALETTE["blue_gray"]};
            --soft-blue:   {PALETTE["soft_blue"]};
            --green:       {PALETTE["green"]};
            --green-soft:  {PALETTE["green_soft"]};
            --navy-soft:   {PALETTE["navy_soft"]};
            --red-soft:    {PALETTE["red_soft"]};
            --warm-gray:   {PALETTE["warm_gray"]};
            --mist:        {PALETTE["mist"]};
            --white:       {PALETTE["white"]};
            --black:       {PALETTE["black"]};
        }}

        html, body, [data-testid="stAppViewContainer"] {{
            font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
        }}

        /* ─── App base ───────────────────────────────────────────────── */
        .stApp {{
            background: var(--warm-gray);
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

        .stCaption {{
            color: var(--blue-gray) !important;
            font-size: 0.82rem;
        }}

        /* ─── Headers de sección (azul marino = estructura) ─────────── */
        [data-testid="stHeader"] {{
            background: transparent;
        }}

        /* ─── Tabs (azul marino = estructura) ────────────────────────── */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            background: var(--white) !important;
            border: 1px solid rgba(17, 48, 71, 0.14) !important;
            border-radius: 999px !important;
            padding: 0.25rem !important;
            gap: 0.15rem !important;
            box-shadow: 0 8px 20px rgba(17, 48, 71, 0.05) !important;
            margin-bottom: 1.1rem !important;
        }}
        [data-testid="stTabs"] [data-baseweb="tab"] {{
            border-radius: 999px !important;
            padding: 0.5rem 1.15rem !important;
            font-weight: 600;
            color: var(--royal-blue);
            transition: all 0.18s ease !important;
        }}
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {{
            background: var(--royal-blue) !important;
            color: #ffffff !important;
            box-shadow: 0 8px 18px rgba(17, 48, 71, 0.22) !important;
        }}
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] span,
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] * {{
            color: #ffffff !important;
        }}
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
            display: none !important;
        }}

        /* ─── Hero card (mantiene marca, degradado azul→rojo) ────────── */
        .hero-card {{
            padding: 2.1rem 2.2rem;
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

        .hero-badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.28);
            border-radius: 999px;
            padding: 0.3rem 0.95rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--macadamia) !important;
            margin-bottom: 0.9rem;
        }}

        .hero-metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1.2rem;
        }}
        .hero-metric {{
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 14px;
            padding: 0.55rem 1rem;
            min-width: 130px;
        }}
        .hero-metric b {{
            font-size: 1.25rem;
            font-weight: 800;
            display: block;
        }}
        .hero-metric span {{
            font-size: 0.72rem;
            color: rgba(255, 255, 255, 0.82) !important;
        }}

        .hero-links {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1.4rem;
        }}
        .hero-link {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.32);
            border-radius: 999px;
            padding: 0.55rem 1.15rem;
            color: var(--white) !important;
            font-size: 0.85rem;
            font-weight: 600;
            text-decoration: none !important;
            transition: background 0.2s ease, transform 0.2s ease;
        }}
        .hero-link svg {{
            width: 16px;
            height: 16px;
            flex-shrink: 0;
            opacity: 0.9;
        }}
        .hero-link:hover {{
            background: rgba(255, 255, 255, 0.26);
            transform: translateY(-1px);
        }}

        /* ─── Footer ─────────────────────────────────────────────────── */
        .app-footer {{
            margin-top: 3rem;
            padding-top: 1.2rem;
            border-top: 1px solid rgba(17, 48, 71, 0.14);
            color: var(--blue-gray);
            font-size: 0.85rem;
            text-align: center;
        }}
        .app-footer a {{
            color: var(--royal-blue);
            font-weight: 600;
            text-decoration: none;
        }}

        /* ─── Métricas (azul marino = estructura) ────────────────────── */
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
        [data-testid="stMetricValue"] {{
            font-size: 1.9rem !important;
        }}

        /* ─── Anillos de progreso (donut) ────────────────────────────── */
        .ring-wrap {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.55rem;
            text-align: center;
        }}
        .ring {{
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: conic-gradient(
                var(--ring-color) calc(var(--pct) * 1%),
                rgba(17, 48, 71, 0.08) 0
            );
            position: relative;
            flex-shrink: 0;
        }}
        .ring-inner {{
            border-radius: 50%;
            background: var(--white);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 14px rgba(17, 48, 71, 0.08);
        }}
        .ring-lg {{
            width: 236px;
            height: 236px;
        }}
        .ring-lg .ring-inner {{
            width: 182px;
            height: 182px;
        }}
        .ring-lg .ring-value {{
            font-size: 3.1rem;
        }}
        .ring-sm {{
            width: 132px;
            height: 132px;
        }}
        .ring-sm .ring-inner {{
            width: 102px;
            height: 102px;
        }}
        .ring-sm .ring-value {{
            font-size: 1.55rem;
        }}
        .ring-value {{
            font-weight: 900;
            color: var(--royal-blue);
            line-height: 1;
            letter-spacing: -0.02em;
            font-variant-numeric: tabular-nums;
        }}
        .ring-label {{
            font-weight: 700;
            color: var(--royal-blue);
            font-size: 0.95rem;
            line-height: 1.3;
        }}
        .ring-sublabel {{
            color: var(--blue-gray);
            font-size: 0.8rem;
            line-height: 1.45;
            max-width: 260px;
        }}

        /* ─── Tablas F1 (detalle expandible) ─────────────────────────── */
        .f1-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border: 1px solid rgba(17, 48, 71, 0.14);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(17, 48, 71, 0.05);
            font-size: 0.9rem;
        }}
        .f1-table th {{
            background: var(--navy-soft);
            color: var(--royal-blue);
            text-align: left;
            padding: 0.6rem 0.95rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }}
        .f1-table td {{
            padding: 0.5rem 0.95rem;
            border-bottom: 1px solid rgba(17, 48, 71, 0.08);
            color: var(--royal-blue);
            vertical-align: middle;
        }}
        .f1-table tr:last-child td {{
            border-bottom: none;
        }}
        .f1-table td.num {{
            text-align: right;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }}

        /* ─── Tabla de evidencias (demo interactiva) ─────────────────── */
        .ev-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border: 1px solid rgba(17, 48, 71, 0.14);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(17, 48, 71, 0.05);
            font-size: 0.9rem;
        }}
        .ev-table th {{
            background: var(--navy-soft);
            color: var(--royal-blue);
            text-align: left;
            padding: 0.6rem 0.95rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }}
        .ev-table td {{
            padding: 0.55rem 0.95rem;
            border-bottom: 1px solid rgba(17, 48, 71, 0.08);
            color: var(--royal-blue);
            vertical-align: middle;
        }}
        .ev-table tr:hover td {{
            background: rgba(17, 48, 71, 0.04);
        }}
        .ev-table tr:last-child td {{
            border-bottom: none;
        }}

        /* ─── Evidencias resaltadas (verde/azul/rojo por sentimiento) ── */
        .ev-highlight {{
            padding: 0.15em 0.45em;
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid transparent;
            border-bottom-width: 2px;
            box-shadow: 0 1px 4px rgba(17, 48, 71, 0.10);
            cursor: help;
        }}
        .legend-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--white);
            border: 1px solid rgba(17, 48, 71, 0.12);
            border-radius: 999px;
            padding: 0.25rem 0.8rem;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }}
        .legend-dot {{
            width: 0.65rem;
            height: 0.65rem;
            border-radius: 50%;
        }}
        .sent-badge {{
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        /* ─── Botones: primario azul marino (CTA), secundarios outline ─ */
        [data-testid="stBaseButton-PRIMARY"] button {{
            background: linear-gradient(
                135deg,
                #173a52 0%,
                var(--royal-blue) 55%,
                #214b67 100%
            ) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 0.7rem 1.4rem !important;
            font-weight: 700 !important;
            font-size: 1.02rem !important;
            box-shadow: 0 14px 26px rgba(17, 48, 71, 0.28) !important;
            transition: box-shadow 0.2s ease, transform 0.15s ease !important;
        }}
        [data-testid="stBaseButton-PRIMARY"] button:hover:not(:disabled) {{
            box-shadow: 0 16px 30px rgba(17, 48, 71, 0.38) !important;
            transform: translateY(-1px) !important;
        }}
        [data-testid="stBaseButton-PRIMARY"] button:disabled {{
            background: #c3ccd4 !important;
            color: #f2f4f7 !important;
            box-shadow: none !important;
            transform: none !important;
            cursor: not-allowed !important;
        }}
        [data-testid="stBaseButton-PRIMARY"] button *,
        [data-testid="stBaseButton-PRIMARY"] button span,
        [data-testid="stBaseButton-PRIMARY"] button svg {{
            color: #ffffff !important;
            fill: #ffffff !important;
        }}
        [data-testid="stBaseButton-PRIMARY"] button:disabled *,
        [data-testid="stBaseButton-PRIMARY"] button:disabled span {{
            color: #f2f4f7 !important;
            fill: #f2f4f7 !important;
        }}

        [data-testid="stBaseButton-SECONDARY"] button,
        [data-testid="stBaseButton-TERTIARY"] button {{
            background: var(--white) !important;
            color: var(--royal-blue) !important;
            border: 1.5px solid var(--royal-blue) !important;
            border-radius: 999px !important;
            padding: 0.55rem 1.1rem !important;
            font-weight: 600 !important;
            box-shadow: none !important;
            transition: background 0.15s ease !important;
        }}
        [data-testid="stBaseButton-SECONDARY"] button:hover,
        [data-testid="stBaseButton-TERTIARY"] button:hover {{
            background: var(--navy-soft) !important;
            color: var(--royal-blue) !important;
        }}
        [data-testid="stBaseButton-SECONDARY"] button *,
        [data-testid="stBaseButton-TERTIARY"] button * {{
            color: var(--royal-blue) !important;
            fill: var(--royal-blue) !important;
        }}

        /* ─── File uploader (tono pastel azul marino = input de usuario) ─ */
        [data-testid="stFileUploader"] {{
            background: var(--mist) !important;
            border: 2px dashed rgba(17, 48, 71, 0.28) !important;
            border-radius: 20px !important;
            padding: 0.5rem !important;
            box-shadow: 0 12px 24px rgba(17, 48, 71, 0.05) !important;
            transition: background 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease !important;
        }}
        [data-testid="stFileUploader"] * {{
            color: var(--royal-blue) !important;
        }}
        [data-testid="stFileUploader"]:hover {{
            background: #dce4ee !important;
            border-color: rgba(17, 48, 71, 0.45) !important;
            box-shadow: 0 14px 28px rgba(17, 48, 71, 0.09) !important;
        }}
        [data-testid="stFileUploaderDropzone"]:has(> :nth-child(3)) {{
            background: #dce4ee !important;
            border-color: var(--royal-blue) !important;
           
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background: #dce4ee !important;
            color: var(--white) !important;
            border-radius: 999px !important;
            border: none !important;
            font-weight: 600 !important;
            transition: background 0.2s ease !important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
            background: #c5d1e0 !important;
        }}

        /* ─── Textarea (tono pastel azul marino = input de usuario) ───── */
        [data-testid="stTextArea"] textarea {{
            background: var(--mist) !important;
            border: 1.5px solid rgba(17, 48, 71, 0.22) !important;
            border-radius: 16px !important;
            box-shadow: 0 10px 22px rgba(17, 48, 71, 0.05) !important;
            color: var(--royal-blue) !important;
            transition: background 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease !important;
        }}
        [data-testid="stTextArea"] textarea:hover {{
            background: #e4e9f1 !important;
            border-color: rgba(17, 48, 71, 0.4) !important;
        }}
        [data-testid="stTextArea"] textarea:focus {{
            background: var(--white) !important;
            border-color: var(--royal-blue) !important;
            box-shadow: 0 0 0 3px rgba(17, 48, 71, 0.12) !important;
        }}

        /* ─── Labels de widgets ──────────────────────────────────────── */
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

        /* ─── Checkbox (azul marino) ─────────────────────────────────── */
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] > span {{
            border-radius: 6px !important;
        }}
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] input:checked ~ span {{
            background-color: var(--royal-blue) !important;
            border-color: var(--royal-blue) !important;
        }}
        [data-testid="stCheckbox"] [data-baseweb="checkbox"] svg {{
            color: #ffffff !important;
        }}

        /* ─── Segmented control / toggle (Archivo CSV | Texto) ───────── */
        [data-testid="stButtonGroup"] [data-orientation="horizontal"] {{
            background: var(--white) !important;
            border: 1px solid rgba(17, 48, 71, 0.22) !important;
            border-radius: 999px !important;
            padding: 0.3rem !important;
            gap: 0.3rem !important;
            box-shadow: 0 8px 18px rgba(17, 48, 71, 0.05) !important;
        }}
        [data-testid="stButtonGroup"] button {{
            background: transparent !important;
            color: var(--royal-blue) !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 600 !important;
            padding: 0.45rem 1.3rem !important;
            transition: background 0.1s ease, color 0.1s ease !important;
        }}
        [data-testid="stButtonGroup"] button:hover {{
            background: var(--navy-soft) !important;
        }}
        [data-testid="stButtonGroup"] button[data-selected] {{
            background: var(--royal-blue) !important;
            color: var(--white) !important;
        }}

        /* ─── Toggle-chips de sentimiento (grupo de 3: POS/NEU/NEG) ──── */
        [data-testid="stButtonGroup"]:has(button:nth-child(3)) {{
            width: fit-content !important;
        }}
        [data-testid="stButtonGroup"]:has(button:nth-child(3)) [data-orientation="horizontal"] {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            gap: 0.35rem !important;
        }}
        [data-testid="stButtonGroup"]:has(button:nth-child(3)) [data-orientation="horizontal"] > button {{
            border: 2px solid !important;
            border-radius: 999px !important;
            background: transparent !important;
            padding: 0.22rem 0.9rem !important;
            font-weight: 700 !important;
            font-size: 0.82rem !important;
            transition: none !important;
        }}
        [data-testid="stButtonGroup"]:has(button:nth-child(3)) [data-orientation="horizontal"] > button:nth-child(1) {{
            color: var(--green) !important;
            border-color: var(--green) !important;
        }}
        [data-testid="stButtonGroup"]:has(button:nth-child(3)) [data-orientation="horizontal"] > button:nth-child(2) {{
            color: var(--soft-blue) !important;
            border-color: var(--soft-blue) !important;
        }}
        [data-testid="stButtonGroup"]:has(button:nth-child(3)) [data-orientation="horizontal"] > button:nth-child(3) {{
            color: var(--wood-red) !important;
            border-color: var(--wood-red) !important;
        }}
        [data-testid="stButtonGroup"]:has(button:nth-child(3)) [data-orientation="horizontal"] > button:nth-child(1)[data-selected] {{
            background: var(--green) !important;
            color: #ffffff !important;
            border-color: var(--green) !important;
        }}
        [data-testid="stButtonGroup"]:has(button:nth-child(3)) [data-orientation="horizontal"] > button:nth-child(2)[data-selected] {{
            background: var(--soft-blue) !important;
            color: #ffffff !important;
            border-color: var(--soft-blue) !important;
        }}
        [data-testid="stButtonGroup"]:has(button:nth-child(3)) [data-orientation="horizontal"] > button:nth-child(3)[data-selected] {{
            background: var(--wood-red) !important;
            color: #ffffff !important;
            border-color: var(--wood-red) !important;
        }}

        /* ─── Selectbox / Multiselect (azul marino, fondo blanco) ────── */
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiSelect"] div:has(> [data-testid="stMultiSelectTagsContainer"]) {{
            background: var(--white) !important;
            border: 1px solid rgba(17, 48, 71, 0.22) !important;
            border-radius: 14px !important;
            box-shadow: 0 8px 18px rgba(17, 48, 71, 0.04) !important;
        }}
        [data-testid="stMultiSelect"] div:has(> [data-testid="stMultiSelectTagsContainer"]) {{
            max-height: none !important;
            min-height: 2.6rem !important;
        }}
        [data-testid="stMultiSelect"] input,
        [data-testid="stSelectbox"] input {{
            color: var(--royal-blue) !important;
            background: transparent !important;
        }}
        [data-testid="stMultiSelect"] input::placeholder,
        [data-testid="stSelectbox"] input::placeholder {{
            color: var(--blue-gray) !important;
        }}
        [data-testid="stMultiSelect"] svg,
        [data-testid="stSelectbox"] svg {{
            color: var(--royal-blue) !important;
            fill: var(--royal-blue) !important;
        }}

        /* Tags / chips del multiselect (azul marino) */
        [data-testid="stMultiSelect"] [data-tag] {{
            background-color: var(--navy-soft) !important;
            border: 1px solid rgba(17, 48, 71, 0.18) !important;
            border-radius: 999px !important;
            color: var(--royal-blue) !important;
        }}
        [data-testid="stMultiSelect"] [data-tag] *,
        [data-testid="stMultiSelect"] [data-tag] span,
        [data-testid="stMultiSelect"] [data-tag] svg {{
            color: var(--royal-blue) !important;
            fill: var(--royal-blue) !important;
        }}

        /* Dropdown del select */
        [data-testid="stMultiSelectDropdown"],
        [data-testid="stSelectboxVirtualDropdown"] {{
            background: var(--white) !important;
            border: 1px solid rgba(17, 48, 71, 0.14) !important;
            border-radius: 14px !important;
            box-shadow: 0 12px 28px rgba(17, 48, 71, 0.10) !important;
            overflow-y: auto !important;
        }}
        [data-testid="stMultiSelectDropdown"] li,
        [data-testid="stMultiSelectDropdown"] [role="option"],
        [data-testid="stSelectboxVirtualDropdown"] li,
        [data-testid="stSelectboxVirtualDropdown"] [role="option"] {{
            color: var(--royal-blue) !important;
        }}
        [data-testid="stMultiSelectDropdown"] [role="option"]:hover,
        [data-testid="stMultiSelectDropdown"] [data-hovered],
        [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
        [data-testid="stSelectboxVirtualDropdown"] [data-hovered] {{
            background: var(--navy-soft) !important;
        }}

        /* ─── Dataframe / tablas ─────────────────────────────────────── */
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] > div {{
            border: 1px solid rgba(17, 48, 71, 0.16) !important;
            border-radius: 16px !important;
            overflow: hidden !important;
            box-shadow: 0 10px 24px rgba(17, 48, 71, 0.05) !important;
        }}
        [data-testid="stDataFrame"] [role="columnheader"] {{
            background: var(--navy-soft) !important;
            color: var(--royal-blue) !important;
            font-weight: 700 !important;
        }}

        /* ─── Notificaciones / alertas (neutras, sin beige) ──────────── */
        [data-testid="stAlert"] {{
            background: var(--white) !important;
            border: 1px solid rgba(17, 48, 71, 0.14) !important;
            border-radius: 14px !important;
            color: var(--royal-blue) !important;
            box-shadow: 0 8px 20px rgba(17, 48, 71, 0.04) !important;
        }}
        [data-testid="stAlert"] * {{
            color: var(--royal-blue) !important;
        }}

        /* ─── Nota de estado (mensaje de entrada, tono azul marino) ──── */
        .status-note {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            background: var(--navy-soft) !important;
            border: 1px solid rgba(17, 48, 71, 0.16) !important;
            border-radius: 14px !important;
            color: var(--royal-blue) !important;
            padding: 0.7rem 1rem !important;
            font-size: 0.9rem;
            font-weight: 500;
            box-shadow: 0 8px 20px rgba(17, 48, 71, 0.05) !important;
        }}
        .status-note svg {{
            color: var(--royal-blue) !important;
            fill: var(--royal-blue) !important;
            flex-shrink: 0;
        }}

        /* ─── Tarjetas de evidencia (detalle por reseña) ─────────────── */
        .evidence-card {{
            display: flex;
            align-items: flex-start;
            gap: 0.7rem;
            background: var(--white);
            border: 1px solid rgba(17, 48, 71, 0.12);
            border-left: 4px solid var(--royal-blue);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 8px 18px rgba(17, 48, 71, 0.05);
        }}
        .evidence-card .sent-badge {{
            flex-shrink: 0;
            margin-top: 0.1rem;
        }}
        .evidence-card .ev-text {{
            color: var(--royal-blue);
            font-weight: 500;
            line-height: 1.5;
        }}
        .evidence-card .ev-meta {{
            color: var(--blue-gray);
            font-size: 0.78rem;
            margin-top: 0.2rem;
        }}
        .evidence-card .ev-meta b {{
            color: var(--royal-blue);
        }}

        /* ─── Enfoque de teclado visible ─────────────────────────────── */
        :focus-visible {{
            outline: 2px solid var(--royal-blue) !important;
            outline-offset: 2px !important;
            border-radius: 6px;
        }}
        [data-testid="stTabs"] [data-baseweb="tab"]:focus-visible,
        [data-testid="stBaseButton-PRIMARY"] button:focus-visible,
        [data-testid="stBaseButton-SECONDARY"] button:focus-visible,
        [data-testid="stButtonGroup"] button:focus-visible,
        [data-testid="stMultiSelect"] :focus-visible,
        [data-testid="stSelectbox"] :focus-visible {{
            outline: 2px solid var(--royal-blue) !important;
            outline-offset: 2px !important;
        }}

        /* ─── Contenedor con scroll horizontal (tabla de evidencias) ─── */
        .ev-table-scroll {{
            overflow-x: auto !important;
            border-radius: 14px;
        }}
        .ev-table-scroll .ev-table {{
            min-width: 720px;
        }}

        /* ─── Movimiento reducido ─────────────────────────────────────── */
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                transition-duration: 0.01ms !important;
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
            }}
        }}
        </style>
"""
