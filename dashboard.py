import os
import json
import ast
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ALERTS_PATH = os.path.join(DATA_DIR, "alerts.csv")
LOGS_PATH = os.path.join(DATA_DIR, "access_logs.csv")
FEEDBACK_PATH = os.path.join(DATA_DIR, "analyst_feedback.csv")
CAMPAIGNS_PATH = os.path.join(DATA_DIR, "campaigns.csv")
PR_CURVE_PATH = os.path.join(DATA_DIR, "pr_curve.png")
ROC_CURVE_PATH = os.path.join(DATA_DIR, "roc_curve.png")
CM_PATH = os.path.join(DATA_DIR, "confusion_matrix.png")
EVAL_REPORT_PATH = os.path.join(DATA_DIR, "eval_report.json")

DESIGN = {
    "bg_base": "#0d1117",
    "bg_surface": "#161b22",
    "bg_elevated": "#21262d",
    "border": "#30363d",
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "text_muted": "#6e7681",
    "accent": "#388bfd",
    "accent_hover": "#58a6ff",
    "critical": "#da3633",
    "high": "#d29922",
    "medium": "#9e6a03",
    "low": "#3fb950",
    "font_sans": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "font_mono": "'JetBrains Mono', 'Fira Code', Consolas, monospace",
}

TIER_COLORS = {
    "Critical": DESIGN["critical"],
    "High": DESIGN["high"],
    "Medium": DESIGN["medium"],
    "Low": DESIGN["low"],
}

FEATURE_LABELS = {
    "new_geo": "New Geo",
    "new_resource": "New Resource",
    "new_device": "New Device",
    "hour_deviation": "Off Hours",
    "duration_z": "Duration Z-Score",
    "n_commands": "Command Count",
    "auth_failed": "Auth Failed",
    "resource_hash": "Resource Hash",
    "graph_distance": "Graph Distance",
}


def render_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {{
            --bg-base: {DESIGN['bg_base']};
            --bg-surface: {DESIGN['bg_surface']};
            --bg-elevated: {DESIGN['bg_elevated']};
            --border: {DESIGN['border']};
            --text-primary: {DESIGN['text_primary']};
            --text-secondary: {DESIGN['text_secondary']};
            --text-muted: {DESIGN['text_muted']};
            --accent: {DESIGN['accent']};
            --critical: {DESIGN['critical']};
            --high: {DESIGN['high']};
            --medium: {DESIGN['medium']};
            --low: {DESIGN['low']};
            --font-sans: {DESIGN['font_sans']};
            --font-mono: {DESIGN['font_mono']};
        }}

        html, body, [class*="css"] {{
            font-family: var(--font-sans);
            color: var(--text-primary);
            background-color: var(--bg-base);
        }}

        .stApp {{
            background-color: var(--bg-base);
            overflow-x: hidden !important;
        }}

        header[data-testid="stHeader"], .stAppHeader {{
            display: none !important;
            height: 0 !important;
        }}

        #MainMenu {{
            display: none !important;
        }}

        [data-testid="stDecoration"] {{
            display: none !important;
        }}

        .main .block-container,
        [data-testid="stAppViewBlockContainer"],
        section[data-testid="stMainBlockContainer"],
        .stMainBlockContainer,
        div.block-container {{
            padding-top: 1.5rem !important;
            margin-top: -3.5rem !important;
            padding-bottom: 2rem;
            max-width: 1400px;
            overflow-x: hidden;
        }}

        .cs-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 1rem 0 0.5rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1.25rem;
        }}

        .cs-header h1 {{
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.03em;
            margin: 0;
            line-height: 1;
        }}

        .cs-header .cs-product-tag {{
            font-size: 0.7rem;
            font-weight: 500;
            color: var(--text-muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border: 1px solid var(--border);
            padding: 2px 7px;
            border-radius: 3px;
        }}

        .cs-stat-row {{
            display: flex;
            gap: 1px;
            background: var(--border);
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 1.25rem;
        }}

        .cs-stat {{
            flex: 1;
            background: var(--bg-surface);
            padding: 0.65rem 1rem;
            min-width: 0;
        }}

        .cs-stat-label {{
            font-size: 0.68rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.07em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .cs-stat-value {{
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.2;
            font-variant-numeric: tabular-nums;
        }}

        .cs-stat-value.critical {{ color: var(--critical); }}
        .cs-stat-value.high {{ color: var(--high); }}
        .cs-stat-value.medium {{ color: #e3b341; }}
        .cs-stat-value.low {{ color: var(--low); }}
        .cs-stat-value.accent {{ color: var(--accent); }}

        .tier-badge {{
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 2px 7px;
            border-radius: 3px;
        }}

        .tier-badge.Critical {{ background: rgba(218,54,51,0.15); color: var(--critical); border: 1px solid rgba(218,54,51,0.3); }}
        .tier-badge.High {{ background: rgba(210,153,34,0.15); color: var(--high); border: 1px solid rgba(210,153,34,0.3); }}
        .tier-badge.Medium {{ background: rgba(158,106,3,0.15); color: #e3b341; border: 1px solid rgba(158,106,3,0.3); }}
        .tier-badge.Low {{ background: rgba(63,185,80,0.15); color: var(--low); border: 1px solid rgba(63,185,80,0.3); }}

        .cs-detail-panel {{
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 1rem 1.25rem;
            margin-top: 0.75rem;
        }}

        .cs-detail-title {{
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .cs-kv-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.4rem 1.5rem;
            margin-bottom: 0.75rem;
        }}

        .cs-kv-item {{
            display: flex;
            flex-direction: column;
            gap: 1px;
        }}

        .cs-kv-key {{
            font-size: 0.67rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        .cs-kv-val {{
            font-size: 0.83rem;
            color: var(--text-primary);
            font-family: var(--font-mono);
        }}

        .cs-mitre-block {{
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 4px;
            padding: 0.6rem 0.85rem;
            margin-bottom: 0.75rem;
        }}

        .cs-mitre-id {{
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: var(--accent);
            font-weight: 600;
        }}

        .cs-mitre-name {{
            font-size: 0.82rem;
            color: var(--text-primary);
            margin-top: 1px;
        }}

        .cs-mitre-tactic {{
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 2px;
        }}

        .cs-reason-list {{
            list-style: none;
            padding: 0;
            margin: 0 0 0.75rem 0;
        }}

        .cs-reason-list li {{
            font-size: 0.82rem;
            color: var(--text-secondary);
            padding: 3px 0;
            padding-left: 1rem;
            position: relative;
        }}

        .cs-reason-list li::before {{
            content: ">";
            position: absolute;
            left: 0;
            color: var(--accent);
            font-family: var(--font-mono);
            font-size: 0.75rem;
        }}

        .cs-section-divider {{
            height: 1px;
            background: var(--border);
            margin: 1rem 0;
        }}

        .cs-killchain-bar {{
            display: flex;
            align-items: center;
            gap: 0;
            margin: 0.6rem 0 0.3rem 0;
        }}

        .cs-kc-stage {{
            flex: 1;
            text-align: center;
            font-size: 0.65rem;
            font-weight: 500;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 5px 4px;
            border: 1px solid var(--border);
            color: var(--text-muted);
            background: var(--bg-surface);
            margin-right: -1px;
        }}

        .cs-kc-stage.active {{
            background: var(--accent);
            border-color: var(--accent);
            color: #fff;
            font-weight: 700;
            z-index: 1;
            position: relative;
        }}

        .cs-kc-stage.complete {{
            background: rgba(56,139,253,0.15);
            border-color: var(--accent);
            color: var(--accent);
        }}

        .cs-kc-stage.danger {{
            background: rgba(218,54,51,0.15);
            border-color: var(--critical);
            color: var(--critical);
            font-weight: 700;
        }}

        .cs-escalation-warning {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(218,54,51,0.12);
            border: 1px solid rgba(218,54,51,0.4);
            border-radius: 4px;
            padding: 5px 10px;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--critical);
            margin-top: 0.4rem;
        }}

        .cs-campaign-badge {{
            display: inline-block;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            padding: 2px 7px;
            border-radius: 3px;
            background: rgba(158,106,3,0.2);
            border: 1px solid rgba(210,153,34,0.4);
            color: var(--high);
            font-family: var(--font-mono);
        }}

        .cs-drift-panel {{
            background: rgba(56,139,253,0.06);
            border: 1px solid rgba(56,139,253,0.25);
            border-radius: 5px;
            padding: 0.6rem 0.8rem;
            margin: 0.5rem 0;
        }}

        .cs-drift-label {{
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.3rem;
        }}

        .cs-drift-bar-outer {{
            height: 6px;
            background: var(--bg-elevated);
            border-radius: 3px;
            overflow: hidden;
            margin: 0.3rem 0;
        }}

        .cs-drift-bar-inner {{
            height: 100%;
            border-radius: 3px;
            background: var(--accent);
        }}

        .cs-btn {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            font-size: 0.78rem;
            font-weight: 500;
            border-radius: 4px;
            border: 1px solid var(--border);
            cursor: pointer;
            font-family: var(--font-sans);
            transition: background 0.15s;
        }}

        .cs-btn-accept {{
            background: rgba(63,185,80,0.1);
            color: var(--low);
            border-color: rgba(63,185,80,0.3);
        }}

        .cs-btn-reject {{
            background: rgba(218,54,51,0.1);
            color: var(--critical);
            border-color: rgba(218,54,51,0.3);
        }}

        .stButton > button {{
            font-family: var(--font-sans);
            font-size: 0.78rem;
            font-weight: 500;
            border-radius: 4px;
            padding: 5px 14px;
            border: 1px solid var(--border);
        }}

        div[data-testid="stMetricValue"] {{
            font-size: 1rem !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            background: transparent;
            border-bottom: 1px solid var(--border);
        }}

        .stTabs [data-baseweb="tab"] {{
            font-family: var(--font-sans);
            font-size: 0.82rem;
            font-weight: 500;
            color: var(--text-muted);
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            padding: 0.5rem 1.1rem;
        }}

        .stTabs [aria-selected="true"] {{
            color: var(--text-primary) !important;
            border-bottom: 2px solid var(--accent) !important;
            background: transparent !important;
        }}

        .stSelectbox > div > div, .stMultiSelect > div > div {{
            background-color: var(--bg-surface);
            border-color: var(--border);
            color: var(--text-primary);
            font-size: 0.82rem;
        }}

        @media (max-width: 900px) {{
            .cs-stat-row {{
                flex-wrap: wrap;
            }}
            .cs-stat {{
                flex: 1 1 calc(50% - 1px);
            }}
            .cs-kv-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _tier_badge(tier: str) -> str:
    return f'<span class="tier-badge {tier}">{tier}</span>'


def _load_alerts() -> pd.DataFrame:
    df = pd.read_csv(ALERTS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["severity_score"] = pd.to_numeric(df.get("severity_score", 0), errors="coerce").fillna(0)
    defaults = {
        "severity_tier": "Low",
        "mitre_id": "T0000",
        "mitre_name": "Unknown",
        "mitre_tactic": "Unknown",
        "shap_values": "{}",
        "kill_chain_stage": "dormant",
        "next_stage": "",
        "next_stage_prob": 0.0,
        "escalation_warning": False,
        "campaign_id": "",
        "drift_score": 0.0,
        "geo_location": "",
        "source_ip": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    df["next_stage_prob"] = pd.to_numeric(df["next_stage_prob"], errors="coerce").fillna(0.0)
    df["drift_score"] = pd.to_numeric(df["drift_score"], errors="coerce").fillna(0.0)
    df["escalation_warning"] = df["escalation_warning"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["campaign_id"] = df["campaign_id"].fillna("")
    return df


def _load_feedback() -> pd.DataFrame:
    if os.path.exists(FEEDBACK_PATH):
        return pd.read_csv(FEEDBACK_PATH)
    return pd.DataFrame(columns=["entity_id", "timestamp", "decision", "alert_index"])


def render_summary_strip(alerts: pd.DataFrame, feedback: pd.DataFrame):
    tier_counts = alerts["severity_tier"].value_counts()
    accept_rate = (
        f"{(feedback['decision'] == 'accept').mean() * 100:.0f}%"
        if len(feedback) > 0 else "-"
    )
    n_escalation = int(alerts.get("escalation_warning", pd.Series(dtype=bool)).sum()) if "escalation_warning" in alerts.columns else 0
    n_campaigns = alerts["campaign_id"].astype(str).str.startswith("C-").sum() if "campaign_id" in alerts.columns else 0
    n_drift = (alerts.get("predicted_type", pd.Series()) == "behavioral_drift").sum() if "predicted_type" in alerts.columns else 0
    stats = [
        ("Total Alerts", str(len(alerts)), ""),
        ("Critical", str(tier_counts.get("Critical", 0)), "critical"),
        ("High", str(tier_counts.get("High", 0)), "high"),
        ("Entities Flagged", str(alerts["entity_id"].nunique()), "accent"),
        ("Escalation Warnings", str(n_escalation), "critical" if n_escalation > 0 else ""),
        ("In Campaign", str(n_campaigns), "high" if n_campaigns > 0 else ""),
        ("Behavioral Drift", str(n_drift), "accent" if n_drift > 0 else ""),
        ("Avg Confidence", f"{alerts['confidence'].mean():.2f}", ""),
    ]

    cells = "".join(
        f'<div class="cs-stat">'
        f'<div class="cs-stat-label">{label}</div>'
        f'<div class="cs-stat-value {cls}">{value}</div>'
        f'</div>'
        for label, value, cls in stats
    )
    st.markdown(f'<div class="cs-stat-row">{cells}</div>', unsafe_allow_html=True)


def render_alert_queue(alerts: pd.DataFrame) -> pd.DataFrame:
    st.markdown("#### Alert Queue")

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    with filter_col1:
        tier_filter = st.multiselect(
            "Severity Tier",
            options=["Critical", "High", "Medium", "Low"],
            default=[],
            key="filter_tier",
        )
    with filter_col2:
        type_filter = st.multiselect(
            "Predicted Type",
            options=sorted(alerts["predicted_type"].unique()),
            default=[],
            key="filter_type",
        )
    with filter_col3:
        sort_by = st.selectbox(
            "Sort By",
            options=["severity_score", "anomaly_score", "confidence", "timestamp"],
            index=0,
            key="sort_by",
        )

    view = alerts.copy()
    if tier_filter:
        view = view[view["severity_tier"].isin(tier_filter)]
    if type_filter:
        view = view[view["predicted_type"].isin(type_filter)]

    view = view.sort_values(sort_by, ascending=(sort_by == "timestamp")).reset_index(drop=True)

    display_cols = ["severity_tier", "mitre_id", "entity_id", "entity_type",
                    "severity_score", "confidence", "predicted_type", "timestamp", "true_label"]
    display_df = view[display_cols].copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["severity_score"] = display_df["severity_score"].map(lambda x: f"{x:.1f}")
    display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.2f}")

    def color_tier(val):
        color = TIER_COLORS.get(val, DESIGN["text_muted"])
        return f"color: {color}; font-weight: 600;"

    def color_score(val):
        try:
            score = float(val)
        except ValueError:
            return ""
        if score >= 80:
            return f"color: {DESIGN['critical']};"
        if score >= 60:
            return f"color: {DESIGN['high']};"
        if score >= 35:
            return "color: #e3b341;"
        return f"color: {DESIGN['low']};"

    styled = (
        display_df.style
        .applymap(color_tier, subset=["severity_tier"])
        .applymap(color_score, subset=["severity_score"])
        .set_properties(**{
            "font-size": "0.8rem",
            "font-family": DESIGN["font_mono"],
        })
    )

    st.dataframe(styled, use_container_width=True, height=340)

    return view


def _parse_shap(shap_str) -> dict:
    if not shap_str or pd.isna(shap_str):
        return {}
    try:
        return json.loads(str(shap_str))
    except Exception:
        try:
            return ast.literal_eval(str(shap_str))
        except Exception:
            return {}


def render_shap_chart(shap_dict: dict):
    if not shap_dict:
        st.markdown(
            '<p style="font-size:0.78rem; color: var(--text-muted);">No SHAP data available.</p>',
            unsafe_allow_html=True,
        )
        return

    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]))
    features = [FEATURE_LABELS.get(k, k) for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(5, max(2.2, len(features) * 0.32)))
    colors = [DESIGN["critical"] if v > 0 else DESIGN["accent"] for v in values]
    ax.barh(features, values, color=colors, height=0.55)
    ax.axvline(x=0, color=DESIGN["border"], linewidth=0.8)
    ax.set_facecolor(DESIGN["bg_surface"])
    fig.patch.set_facecolor(DESIGN["bg_surface"])
    for spine in ax.spines.values():
        spine.set_color(DESIGN["border"])
    ax.tick_params(colors=DESIGN["text_secondary"], labelsize=8)
    ax.set_xlabel("SHAP Impact on Prediction", color=DESIGN["text_secondary"], fontsize=8)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_detail_panel(row: pd.Series, logs: pd.DataFrame, feedback: pd.DataFrame, row_index: int):
    tier = str(row.get("severity_tier", "Low"))
    tier_color = TIER_COLORS.get(tier, DESIGN["text_muted"])

    st.markdown(
        f"""
        <div class="cs-detail-panel">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.75rem;">
            <span style="font-size:0.95rem; font-weight:600; color:{DESIGN['text_primary']};">
                {row['entity_id']}
            </span>
            {_tier_badge(tier)}
            <span style="font-size:0.75rem; color:{DESIGN['text_muted']}; margin-left:auto;">
                {str(row['entity_type']).upper()}
            </span>
        </div>
        <div class="cs-kv-grid">
            <div class="cs-kv-item">
                <span class="cs-kv-key">Timestamp</span>
                <span class="cs-kv-val">{str(row['timestamp'])[:19]}</span>
            </div>
            <div class="cs-kv-item">
                <span class="cs-kv-key">Severity Score</span>
                <span class="cs-kv-val" style="color:{tier_color};">{row.get('severity_score', '-')}</span>
            </div>
            <div class="cs-kv-item">
                <span class="cs-kv-key">Anomaly Score</span>
                <span class="cs-kv-val">{float(row.get('anomaly_score', 0)):.5f}</span>
            </div>
            <div class="cs-kv-item">
                <span class="cs-kv-key">Classifier Confidence</span>
                <span class="cs-kv-val">{float(row.get('confidence', 0)):.2f}</span>
            </div>
            <div class="cs-kv-item">
                <span class="cs-kv-key">Predicted Type</span>
                <span class="cs-kv-val">{row.get('predicted_type', '-')}</span>
            </div>
            <div class="cs-kv-item">
                <span class="cs-kv-key">Ground Truth</span>
                <span class="cs-kv-val">{row.get('true_label', '-')}</span>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mitre_id = str(row.get("mitre_id", "T0000"))
    mitre_name = str(row.get("mitre_name", "Unknown"))
    mitre_tactic = str(row.get("mitre_tactic", "Unknown"))
    campaign_id = str(row.get("campaign_id", ""))
    campaign_badge = ""
    if campaign_id.startswith("C-"):
        campaign_badge = (
            f'&nbsp;<span style="font-size:0.65rem;font-weight:700;letter-spacing:0.06em;'
            f'padding:2px 7px;border-radius:3px;background:rgba(158,106,3,0.2);'
            f'border:1px solid rgba(210,153,34,0.4);color:{DESIGN["high"]};'
            f'font-family:var(--font-mono);">{campaign_id}</span>'
        )
    st.markdown(
        f'<div style="background:var(--bg-elevated);border:1px solid var(--border);'
        f'border-radius:5px;padding:0.6rem 0.8rem;margin:0.75rem 0;'
        f'word-break:break-word;overflow-wrap:break-word;">'
        f'<div style="font-family:var(--font-mono);font-size:0.9rem;font-weight:700;'
        f'color:var(--accent);margin-bottom:0.25rem;">{mitre_id}{campaign_badge}</div>'
        f'<div style="font-size:0.82rem;font-weight:600;color:var(--text-primary);'
        f'margin-bottom:0.15rem;">{mitre_name}</div>'
        f'<div style="font-size:0.72rem;color:var(--text-muted);">Tactic: {mitre_tactic}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


    _render_kill_chain_panel(row)
    _render_drift_panel(row)

    reasons_raw = str(row.get("reasons", ""))
    reason_items = [r.strip() for r in reasons_raw.split(";") if r.strip()]
    items_html = "".join(f"<li>{r}</li>" for r in reason_items)
    st.markdown(
        f"""
        <div class="cs-detail-title">Contributing Factors (SHAP)</div>
        <ul class="cs-reason-list">{items_html}</ul>
        """,
        unsafe_allow_html=True,
    )

    shap_dict = _parse_shap(row.get("shap_values", "{}"))
    render_shap_chart(shap_dict)

    st.markdown('<div class="cs-section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cs-detail-title">Entity Session History (last 30)</div>',
        unsafe_allow_html=True,
    )
    entity_history = logs[logs["entity_id"] == row["entity_id"]].sort_values("timestamp")
    history_cols = ["timestamp", "geo_location", "resource_accessed",
                    "auth_method", "session_duration", "label"]
    available = [c for c in history_cols if c in entity_history.columns]
    st.dataframe(
        entity_history[available].tail(30),
        use_container_width=True,
        height=200,
    )

    st.markdown('<div class="cs-section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="cs-detail-title">Analyst Decision</div>', unsafe_allow_html=True)

    acc_col, rej_col, _ = st.columns([1, 1, 3])
    with acc_col:
        if st.button("Accept - True Positive", key=f"accept_{row_index}"):
            _log_feedback(row, "accept")
            st.success("Logged: True Positive")
    with rej_col:
        if st.button("Reject - False Positive", key=f"reject_{row_index}"):
            _log_feedback(row, "reject")
            st.info("Logged: False Positive")


def _log_feedback(row: pd.Series, decision: str):
    existing = _load_feedback()
    new_entry = pd.DataFrame([{
        "entity_id": row["entity_id"],
        "timestamp": str(row["timestamp"]),
        "decision": decision,
    }])
    updated = pd.concat([existing, new_entry], ignore_index=True)
    updated.to_csv(FEEDBACK_PATH, index=False)


def _render_kill_chain_panel(row: pd.Series):
    stage = str(row.get("kill_chain_stage", "dormant"))
    next_stage = str(row.get("next_stage", ""))
    next_prob = float(row.get("next_stage_prob", 0.0))
    escalation = bool(row.get("escalation_warning", False))
    confidence = float(row.get("confidence", 0.0))

    STAGES = ["dormant", "initial_access", "lateral_movement", "exfiltration"]
    STAGE_LABELS = ["Dormant", "Initial Access", "Lateral Movement", "Exfiltration"]
    stage_rank = {s: i for i, s in enumerate(STAGES)}
    current_rank = stage_rank.get(stage, 0)

    def stage_class(i):
        if i < current_rank:
            return "complete"
        if i == current_rank:
            return "active danger" if stage == "exfiltration" else "active"
        return ""

    bars = "".join(
        f'<div class="cs-kc-stage {stage_class(i)}">{label}</div>'
        for i, label in enumerate(STAGE_LABELS)
    )

    warning_html = ""
    if escalation:
        warning_html = (
            f'<div class="cs-escalation-warning">'
            f'Escalation Warning - {int(next_prob*100)}% likely: {next_stage.replace("_", " ").title()}'
            f'</div>'
        )

    conf_pct = int(confidence * 100)
    st.markdown(
        f"""
        <div class="cs-section-divider"></div>
        <div class="cs-detail-title">Kill Chain Position</div>
        <div class="cs-killchain-bar">{bars}</div>
        <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.2rem;">
            Stage confidence: {conf_pct}%
        </div>
        {warning_html}
        """,
        unsafe_allow_html=True,
    )


def _render_drift_panel(row: pd.Series):
    drift = float(row.get("drift_score", 0.0))
    predicted_type = str(row.get("predicted_type", ""))
    if drift <= 0.0 and predicted_type != "behavioral_drift":
        return

    pct = min(int(drift * 400), 100)
    color = "#da3633" if pct > 70 else "#d29922" if pct > 40 else "#388bfd"
    label = "Account Takeover Risk" if pct > 70 else "Behavioral Drift Detected" if pct > 40 else "Fingerprint Drift"

    st.markdown(
        f"""
        <div class="cs-section-divider"></div>
        <div class="cs-drift-panel">
            <div class="cs-drift-label">{label}</div>
            <div style="font-size:0.78rem; color:var(--text-secondary);">
                Cosine drift from 14-day behavioral baseline: <strong style="color:{color};">{drift:.4f}</strong>
            </div>
            <div class="cs-drift-bar-outer">
                <div class="cs-drift-bar-inner" style="width:{pct}%; background:{color};"></div>
            </div>
            <div style="font-size:0.7rem; color:var(--text-muted);">
                Reconstruction error below anomaly threshold - evading per-event detection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_campaigns_tab():
    st.markdown("#### Active Campaigns")
    if not os.path.exists(CAMPAIGNS_PATH):
        st.info("No campaigns.csv found. Run main.py to detect coordinated campaigns.")
        return

    camps = pd.read_csv(CAMPAIGNS_PATH)
    if len(camps) == 0:
        st.info("No coordinated campaigns detected in this dataset.")
        return

    camps["t_start"] = pd.to_datetime(camps["t_start"])
    camps["t_end"] = pd.to_datetime(camps["t_end"])

    for _, camp in camps.iterrows():
        risk = int(camp.get("campaign_risk_score", 0))
        risk_color = (
            DESIGN["critical"] if risk >= 80
            else DESIGN["high"] if risk >= 50
            else "#e3b341"
        )
        t0 = camp["t_start"].strftime("%m-%d %H:%M")
        t1 = camp["t_end"].strftime("%m-%d %H:%M")
        duration = camp.get("duration_minutes", "?")
        entities = str(camp.get("entity_list", ""))
        entity_chips = "".join(
            f'<span style="font-family:var(--font-mono);font-size:0.7rem;'
            f'background:var(--bg-elevated);border:1px solid var(--border);'
            f'border-radius:3px;padding:1px 6px;margin:2px;display:inline-block;">{e.strip()}</span>'
            for e in entities.split(";")[:10] if e.strip()
        )
        st.markdown(
            f"""
            <div style="background:var(--bg-surface);border:1px solid var(--border);
                        border-left:3px solid {risk_color};
                        border-radius:5px;padding:0.8rem 1rem;margin-bottom:0.75rem;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.4rem;">
                    <span style="font-family:var(--font-mono);font-weight:700;
                                font-size:0.85rem;color:{risk_color};">{camp['campaign_id']}</span>
                    <span style="font-size:0.8rem;font-weight:600;
                                color:var(--text-primary);">{camp.get('campaign_type','')}</span>
                    <span style="margin-left:auto;font-size:0.72rem;color:var(--text-muted);">
                        Risk Score: <strong style="color:{risk_color};">{risk}</strong>
                    </span>
                </div>
                <div style="font-size:0.72rem;color:var(--text-muted);margin-bottom:0.4rem;">
                    {t0} &rarr; {t1} &nbsp;|&nbsp; {duration} min window &nbsp;|&nbsp;
                    {camp.get('entities_involved',0)} entities &nbsp;|&nbsp;
                    {camp.get('n_alerts',0)} alerts
                </div>
                <div style="font-size:0.72rem;color:var(--text-secondary);margin-bottom:0.3rem;">
                    Attack types: {camp.get('attack_types','')}
                </div>
                <div>{entity_chips}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_analytics():
    st.markdown("#### Detection Performance")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        if os.path.exists(PR_CURVE_PATH):
            st.image(PR_CURVE_PATH, caption="Precision-Recall Curve", use_container_width=True)
        else:
            st.info("Run main.py to generate pr_curve.png")
    with img_col2:
        if os.path.exists(ROC_CURVE_PATH):
            st.image(ROC_CURVE_PATH, caption="ROC-AUC Curve", use_container_width=True)
        else:
            st.info("Run main.py to generate roc_curve.png")

    st.markdown("#### Anomaly-Type Classifier")
    if os.path.exists(CM_PATH):
        st.image(CM_PATH, caption="Confusion Matrix - predicted vs true attack type", use_container_width=True)
    else:
        st.info("Run main.py to generate confusion_matrix.png")

    if os.path.exists(EVAL_REPORT_PATH):
        with open(EVAL_REPORT_PATH) as f:
            report = json.load(f)

        st.markdown("#### Evaluation Report")

        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        metrics_col1.metric("ROC-AUC", f"{report.get('roc_auc', '-')}")
        metrics_col2.metric("Avg Precision", f"{report.get('average_precision', '-')}")
        metrics_col3.metric("Precision @ 1%", f"{report.get('precision_at_budget', '-')}")
        metrics_col4.metric("Recall @ 1%", f"{report.get('recall_at_budget', '-')}")

        fp_data = report.get("concept_drift_fp_comparison", {})
        if fp_data:
            st.markdown("#### Concept Drift: Rolling Re-baselining Impact")
            fp_col1, fp_col2, fp_col3 = st.columns(3)
            fp_col1.metric("FP Rate (Fixed Baseline)", f"{fp_data.get('fp_rate_fixed_baseline', '-')}")
            fp_col2.metric("FP Rate (Rolling 30-day)", f"{fp_data.get('fp_rate_rolling_30day_baseline', '-')}")
            fp_col3.metric("FP Reduction", f"{fp_data.get('fp_reduction_pct', '-')}%")

        budget_data = report.get("precision_recall_by_budget", {})
        if budget_data:
            st.markdown("#### Precision / Recall at Alert Budgets")
            budget_rows = []
            for pct, vals in budget_data.items():
                budget_rows.append({
                    "Budget %": f"{float(pct)*100:.0f}%",
                    "Alerts Raised": vals.get("n_alerts", "-"),
                    "Precision": f"{vals.get('precision', 0):.3f}",
                    "Recall": f"{vals.get('recall', 0):.3f}",
                })
            st.dataframe(pd.DataFrame(budget_rows), use_container_width=True, hide_index=True)


def _render_severity_trend(alerts: pd.DataFrame):
    st.markdown("#### Severity Distribution Over Time")
    alerts_sorted = alerts.sort_values("timestamp").copy()
    alerts_sorted["date"] = alerts_sorted["timestamp"].dt.date
    daily = alerts_sorted.groupby(["date", "severity_tier"]).size().unstack(fill_value=0)

    tier_order = [t for t in ["Critical", "High", "Medium", "Low"] if t in daily.columns]
    daily = daily[tier_order]

    tier_plot_colors = {
        "Critical": DESIGN["critical"],
        "High": DESIGN["high"],
        "Medium": "#e3b341",
        "Low": DESIGN["low"],
    }

    fig, ax = plt.subplots(figsize=(10, 3.5))
    bottom = np.zeros(len(daily))
    for tier in tier_order:
        vals = daily[tier].values
        ax.bar(
            range(len(daily)),
            vals,
            bottom=bottom,
            color=tier_plot_colors.get(tier, DESIGN["text_muted"]),
            label=tier,
            width=0.7,
        )
        bottom += vals

    ax.set_xticks(range(len(daily)))
    ax.set_xticklabels([str(d) for d in daily.index], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Alert Count", fontsize=8)
    ax.legend(fontsize=7, facecolor=DESIGN["bg_surface"], labelcolor=DESIGN["text_secondary"])
    ax.set_facecolor(DESIGN["bg_surface"])
    fig.patch.set_facecolor(DESIGN["bg_base"])
    for spine in ax.spines.values():
        spine.set_color(DESIGN["border"])
    ax.tick_params(colors=DESIGN["text_secondary"], labelsize=7)
    ax.yaxis.label.set_color(DESIGN["text_secondary"])
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def main():
    st.set_page_config(
        page_title="CogniShield - SOC Analyst Console",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_css()

    st.markdown(
        """
        <div class="cs-header">
            <h1 style="font-size:2.1rem;font-weight:900;color:#e6edf3;letter-spacing:-0.04em;margin:0;line-height:1;font-family:'Inter',-apple-system,sans-serif;">CogniShield</h1>
            <span class="cs-product-tag">SOC Analyst Console</span>
            <span class="cs-product-tag" style="margin-left:4px;">Behavioral Anomaly Detection</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.path.exists(ALERTS_PATH):
        st.error("No alerts.csv found. Run `python main.py` to generate pipeline output.")
        st.stop()

    alerts = _load_alerts()
    logs = pd.read_csv(LOGS_PATH, parse_dates=["timestamp"]) if os.path.exists(LOGS_PATH) else pd.DataFrame()
    feedback = _load_feedback()

    render_summary_strip(alerts, feedback)

    tab_queue, tab_analytics = st.tabs(["Alert Queue", "Analytics"])

    with tab_queue:
        left_col, right_col = st.columns([3, 2], gap="medium")

        with left_col:
            view = render_alert_queue(alerts)

            st.markdown("#### Investigate Alert")
            if len(view) == 0:
                st.info("No alerts match the current filter.")
            else:
                selected_idx = st.selectbox(
                    "Select alert row index",
                    options=list(range(len(view))),
                    format_func=lambda i: (
                        f"[{view.iloc[i]['severity_tier']}] "
                        f"{view.iloc[i]['entity_id']} - "
                        f"{view.iloc[i]['predicted_type']} - "
                        f"Score {view.iloc[i]['severity_score']}"
                    ),
                    key="alert_selector",
                )
                selected_row = view.iloc[int(selected_idx)]

        with right_col:
            if len(view) > 0:
                render_detail_panel(selected_row, logs, feedback, int(selected_idx))

    with tab_analytics:
        sub_performance, sub_campaigns, sub_trend = st.tabs(
            ["Detection Performance", "Campaigns", "Severity Trend"]
        )
        with sub_performance:
            render_analytics()
        with sub_campaigns:
            render_campaigns_tab()
        with sub_trend:
            if len(alerts) > 0:
                _render_severity_trend(alerts)


main()
