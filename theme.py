"""Thème visuel : CSS injecté, palette, composants HTML et style Plotly."""
from __future__ import annotations

from html import escape

import streamlit as st

PALETTE = {
    "primary": "#1F4E8C", "primary_dark": "#0F2A4A", "text": "#16202E", "muted": "#6B7A90",
    "border": "#E3E8EF", "grid": "#EDF1F6", "success": "#1E8E5A", "warning": "#D98E04", "danger": "#C0392B",
}
FONT_FAMILY = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#F5F7FA;--card:#FFFFFF;--border:#E3E8EF;--text:#16202E;--muted:#6B7A90;--primary:#1F4E8C;
--primary-dark:#0F2A4A;--success:#1E8E5A;--warning:#D98E04;--danger:#C0392B;}
html,body,.stApp,[class*="css"]{font-family:Inter,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--text);}
.stApp{background:var(--bg);}
#MainMenu,footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1280px;}

/* Sidebar */
section[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid var(--border);}
section[data-testid="stSidebar"] .block-container{padding-top:1.5rem;}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:.4rem;}
.brand-logo{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#0F2A4A,#1F4E8C);
color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;}
.brand-name{font-weight:700;font-size:1.02rem;letter-spacing:-.01em;}
.brand-sub{color:var(--muted);font-size:.78rem;margin-bottom:1rem;}
.side-title{font-size:.7rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
margin:1.2rem 0 .5rem;padding-bottom:.4rem;border-bottom:1px solid var(--border);}
label[data-testid="stWidgetLabel"] p{font-size:.82rem;font-weight:500;color:var(--text);}
div[data-baseweb="input"],div[data-baseweb="select"]>div{border-radius:8px;border-color:var(--border);background:#FBFCFE;}

/* Header */
.app-header{background:linear-gradient(120deg,#0F2A4A 0%,#1F4E8C 100%);border-radius:14px;padding:22px 28px;
margin-bottom:1.4rem;display:flex;justify-content:space-between;align-items:center;
box-shadow:0 6px 18px rgba(15,42,74,.18);}
.app-header h1{color:#fff;font-size:1.5rem;font-weight:700;margin:0;letter-spacing:-.02em;padding:0;}
.app-header p{color:#BFD0E6;margin:.25rem 0 0;font-size:.88rem;}
.badge{background:rgba(255,255,255,.14);color:#fff;border:1px solid rgba(255,255,255,.25);border-radius:999px;
padding:6px 14px;font-size:.76rem;font-weight:500;white-space:nowrap;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{height:46px;padding:0 18px;font-weight:500;color:var(--muted);background:transparent;}
.stTabs [aria-selected="true"]{color:var(--primary);font-weight:600;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--primary);height:3px;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.3rem;}

/* KPI cards */
.kpi{background:var(--card);border:1px solid var(--border);border-left:4px solid var(--accent,#1F4E8C);
border-radius:12px;padding:16px 18px;box-shadow:0 1px 2px rgba(16,24,40,.04);height:100%;}
.kpi-label{font-size:.7rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);}
.kpi-value{font-size:1.75rem;font-weight:700;letter-spacing:-.02em;margin:.2rem 0 .1rem;line-height:1.15;}
.kpi-sub{font-size:.78rem;color:var(--muted);}

/* Content cards (st.container(key="card_*")) */
div[class*="st-key-card_"]{background:var(--card);border:1px solid var(--border);border-radius:14px;
padding:1.15rem 1.3rem;box-shadow:0 1px 3px rgba(16,24,40,.05);}
.sec-title{font-size:1rem;font-weight:600;margin:0;letter-spacing:-.01em;}
.sec-sub{font-size:.8rem;color:var(--muted);margin:.15rem 0 .6rem;}

/* Chips & key/value */
.chip{display:inline-block;padding:5px 11px;border-radius:999px;font-size:.78rem;font-weight:500;margin:0 6px 6px 0;
border:1px solid transparent;}
.chip.up{background:#FBE9E7;color:#9B2C20;border-color:#F4C9C3;}
.chip.down{background:#E3F4EB;color:#17724A;border-color:#BFE3D0;}
.kv{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px dashed var(--border);font-size:.86rem;}
.kv:last-child{border-bottom:none;}
.kv span:first-child{color:var(--muted);} .kv span:last-child{font-weight:600;}
.note{font-size:.84rem;color:#3B4A5E;line-height:1.55;}

/* Widgets */
.stDownloadButton button{border-radius:8px;border:1px solid var(--border);background:#fff;font-weight:500;}
.stDownloadButton button:hover{border-color:var(--primary);color:var(--primary);}
div[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:10px;overflow:hidden;}
div[data-testid="stExpander"]{border:1px solid var(--border);border-radius:10px;background:#fff;}
</style>
"""


def inject_css() -> None:
    """Injecte la feuille de style globale."""
    st.markdown(_CSS, unsafe_allow_html=True)


def card(name: str):
    """Conteneur stylé 'carte' (la clé produit la classe CSS st-key-card_<name>)."""
    return st.container(key=f"card_{name}")


def header(title: str, subtitle: str, badge: str) -> None:
    st.markdown(
        f'<div class="app-header"><div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>'
        f'<div class="badge">{escape(badge)}</div></div>', unsafe_allow_html=True)


def section_title(title: str, subtitle: str = "") -> None:
    sub = f'<div class="sec-sub">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="sec-title">{escape(title)}</div>{sub}', unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", accent: str | None = None) -> str:
    accent = accent or PALETTE["primary"]
    return (f'<div class="kpi" style="--accent:{accent}"><div class="kpi-label">{escape(label)}</div>'
            f'<div class="kpi-value">{escape(value)}</div><div class="kpi-sub">{escape(sub)}</div></div>')


def chip(text: str, direction: str) -> str:
    """direction : 'up' (augmente le risque) ou 'down' (le réduit)."""
    arrow = "▲" if direction == "up" else "▼"
    return f'<span class="chip {direction}">{arrow} {escape(text)}</span>'


def kv_rows(items: list[tuple[str, str]]) -> str:
    return "".join(f'<div class="kv"><span>{escape(k)}</span><span>{escape(v)}</span></div>' for k, v in items)


def style_figure(fig, height: int = 360):
    """Applique le thème du dashboard à une figure Plotly (fonds transparents, police nette)."""
    p = PALETTE
    fig.update_layout(
        template="none", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=p["text"], size=12), margin=dict(l=8, r=24, t=10, b=8),
        hoverlabel=dict(bgcolor="white", bordercolor=p["border"], font=dict(family=FONT_FAMILY, size=12)),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["border"], tickfont=dict(color=p["muted"]))
    fig.update_yaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["border"], tickfont=dict(color=p["text"]))
    return fig
