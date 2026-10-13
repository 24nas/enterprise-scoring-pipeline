"""Credit Risk Studio : dashboard de scoring et d'explicabilité locale (XGBoost + SHAP).

Lancement : streamlit run app.py
"""
from __future__ import annotations

import json
import logging
from html import escape

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

import config
from config import pretty
from theme import (PALETTE, card, chip, header, inject_css, kpi_card, kv_rows, section_title, style_figure)

log = logging.getLogger("app")
P = PALETTE

# --------------------------------------------------------------------------- #
# Exceptions métier
# --------------------------------------------------------------------------- #
class ModelLoadError(RuntimeError):
    """Artefacts du modèle absents ou illisibles."""


class InputValidationError(ValueError):
    """Saisies utilisateur invalides."""


# --------------------------------------------------------------------------- #
# Chargement (mis en cache)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_bundle() -> dict:
    if not config.BUNDLE_PATH.exists():
        raise ModelLoadError(f"Modèle introuvable ({config.BUNDLE_PATH.name}). Lancez d'abord : python train.py")
    try:
        bundle = joblib.load(config.BUNDLE_PATH)
    except Exception as exc:  # fichier corrompu, versions incompatibles…
        raise ModelLoadError(f"Impossible de lire le modèle : {exc}. Ré-entraînez avec python train.py") from exc
    required = {"preprocessor", "model", "feature_names", "schema", "threshold", "metrics", "metadata"}
    if not required <= set(bundle):
        raise ModelLoadError(f"Bundle incomplet, clés manquantes : {sorted(required - set(bundle))}")
    return bundle


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame | None:
    """Échantillon de référence (optionnel : l'app reste fonctionnelle sans lui)."""
    try:
        return pd.read_csv(config.SAMPLE_PATH)
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        log.warning("Échantillon de référence indisponible : %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def get_explainer(_model) -> shap.TreeExplainer:
    return shap.TreeExplainer(_model)


# --------------------------------------------------------------------------- #
# Logique métier
# --------------------------------------------------------------------------- #
def nice_step(span: float, integer: bool) -> float | int:
    """Pas de saisie adapté à l'amplitude de la variable."""
    step = 10 ** int(np.floor(np.log10(max(span, 1e-9) / 100)))
    return int(max(1, step)) if integer else float(max(0.01, step))


def validate_inputs(values: dict, schema: dict) -> pd.DataFrame:
    """Valide les saisies et construit le DataFrame d'une ligne attendu par le préprocesseur."""
    num, cat = list(schema["numeric"]), list(schema["categorical"])
    missing = [c for c in num + cat if c not in values]
    if missing:
        raise InputValidationError(f"Champs manquants : {', '.join(map(pretty, missing))}")
    df = pd.DataFrame([values], columns=num + cat)
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    if num and not np.isfinite(df[num].to_numpy(dtype=float)).all():
        raise InputValidationError("Une variable numérique est vide ou invalide.")
    for c in cat:
        if df.at[0, c] not in schema["categorical"][c]["categories"]:
            raise InputValidationError(f"Modalité inconnue pour « {pretty(c)} » : {df.at[0, c]}")
    return df


def score(bundle: dict, df: pd.DataFrame) -> tuple[float, np.ndarray]:
    """Retourne (probabilité de défaut, vecteur de features transformé)."""
    X = bundle["preprocessor"].transform(df)
    return float(bundle["model"].predict_proba(X)[0, 1]), X


def shap_values(explainer: shap.TreeExplainer, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Valeurs SHAP (log-odds) classe positive + valeur de base, robuste aux versions de SHAP."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    return sv, float(np.ravel(explainer.expected_value)[-1])


def display_name(feat: str, schema: dict) -> str:
    """Rend lisible un nom de feature issu du one-hot / indicateur de manquants."""
    if feat.startswith("missingindicator_"):
        return f"{pretty(feat.removeprefix('missingindicator_'))} (manquant)"
    for col in schema["categorical"]:
        if feat.startswith(col + "_"):
            return f"{pretty(col)} : {feat[len(col) + 1:]}"
    return pretty(feat)


def contributions(bundle: dict, X: np.ndarray, sv: np.ndarray) -> pd.DataFrame:
    schema = bundle["schema"]
    df = pd.DataFrame({"feature": bundle["feature_names"], "value": X[0], "shap": sv[0]})
    df["label"] = df["feature"].map(lambda f: display_name(f, schema))
    return df.reindex(df["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)


@st.cache_data(show_spinner="Calcul de l'importance globale…")
def global_importance(_bundle: dict, features: pd.DataFrame) -> pd.Series:
    """Importance moyenne |SHAP| sur un échantillon de référence."""
    X = _bundle["preprocessor"].transform(features.head(300))
    sv, _ = shap_values(get_explainer(_bundle["model"]), X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=_bundle["feature_names"])
    imp.index = [display_name(f, _bundle["schema"]) for f in imp.index]
    return imp.sort_values(ascending=False)


def risk_level(proba: float, threshold: float) -> tuple[str, str, str]:
    """(niveau, décision recommandée, couleur) selon le seuil optimisé du modèle."""
    if proba < 0.5 * threshold:
        return "Faible", "Accord recommandé", P["success"]
    if proba < threshold:
        return "Modéré", "Revue manuelle", P["warning"]
    return "Élevé", "Refus recommandé", P["danger"]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(schema: dict) -> dict:
    values: dict = {}
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-logo">C</div><div class="brand-name">Credit Risk Studio</div></div>'
                    '<div class="brand-sub">Paramètres du dossier à évaluer</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-title">Profil financier</div>', unsafe_allow_html=True)
        for col, s in schema["numeric"].items():
            if s["is_integer"]:
                lo, hi = int(np.floor(s["min"])), int(np.ceil(s["max"]))
                med, fmt = int(round(s["median"])), "%d"
            else:
                lo, hi, med, fmt = float(s["min"]), float(s["max"]), float(s["median"]), "%.2f"
            values[col] = st.number_input(pretty(col), min_value=lo, max_value=hi, value=min(max(med, lo), hi),
                                          step=nice_step(hi - lo, s["is_integer"]), format=fmt, key=f"in_{col}")
        if schema["categorical"]:
            st.markdown('<div class="side-title">Profil qualitatif</div>', unsafe_allow_html=True)
        for col, s in schema["categorical"].items():
            cats = s["categories"]
            values[col] = st.selectbox(pretty(col), cats, index=cats.index(s["default"]), key=f"in_{col}")
    return values


# --------------------------------------------------------------------------- #
# Graphiques
# --------------------------------------------------------------------------- #
def gauge_chart(proba: float, threshold: float, color: str) -> go.Figure:
    low = threshold * 0.5 * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=proba * 100,
        number={"suffix": " %", "valueformat": ".1f", "font": {"size": 46, "color": P["text"]}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0, "ticksuffix": "%", "tickcolor": P["muted"]},
               "bar": {"color": color, "thickness": 0.3}, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, low], "color": "#DDF3E8"},
                         {"range": [low, threshold * 100], "color": "#FCEFD2"},
                         {"range": [threshold * 100, 100], "color": "#F8DCD8"}],
               "threshold": {"line": {"color": P["text"], "width": 3}, "thickness": 0.85, "value": threshold * 100}}))
    return style_figure(fig, 290).update_layout(margin=dict(l=20, r=20, t=20, b=0))


def distribution_chart(sample: pd.DataFrame, proba: float) -> go.Figure:
    fig = go.Figure(go.Histogram(x=sample["pred_proba"] * 100, nbinsx=40, marker=dict(color=P["primary"], opacity=.78,
                                   line=dict(color="white", width=1)), hovertemplate="%{x:.1f} % : %{y} dossiers<extra></extra>"))
    fig.add_vline(x=proba * 100, line=dict(color=P["danger"], width=2, dash="dash"),
                  annotation_text="Ce dossier", annotation_font=dict(color=P["danger"], size=11))
    fig.update_xaxes(title="Probabilité de défaut (%)", ticksuffix="%")
    fig.update_yaxes(title="Dossiers")
    return style_figure(fig, 280)


def local_shap_chart(contrib: pd.DataFrame, n: int) -> go.Figure:
    top = contrib.head(n).iloc[::-1]
    labels = [f"{l} = {v:g}" if not (f.count("_") and v in (0, 1) and l != f) else l
              for l, v, f in zip(top["label"], top["value"], top["feature"])]
    fig = go.Figure(go.Bar(
        x=top["shap"], y=labels, orientation="h", text=[f"{v:+.2f}" for v in top["shap"]], textposition="outside",
        cliponaxis=False, marker=dict(color=[P["danger"] if v > 0 else P["success"] for v in top["shap"]]),
        textfont=dict(size=11, color=P["muted"]), hovertemplate="%{y}<br>Contribution : %{x:+.3f}<extra></extra>"))
    fig.update_xaxes(title="Contribution au log-odds de défaut", zeroline=True, zerolinecolor=P["muted"])
    return style_figure(fig, max(320, 34 * n + 70)).update_layout(margin=dict(l=8, r=50, t=10, b=8))


def global_chart(imp: pd.Series, n: int = 12) -> go.Figure:
    top = imp.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker=dict(color=P["primary"], opacity=.85),
                           hovertemplate="%{y}<br>|SHAP| moyen : %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Importance moyenne |SHAP|")
    return style_figure(fig, max(320, 30 * n + 70))


# --------------------------------------------------------------------------- #
# Onglets
# --------------------------------------------------------------------------- #
def tab_overview(bundle, df, proba, base_proba, contrib, sample) -> None:
    thr = bundle["threshold"]
    level, decision, color = risk_level(proba, thr)
    m = bundle["metrics"]

    c = st.columns(4, gap="medium")
    cards = [("Probabilité de défaut", f"{proba:.1%}", f"Portefeuille de référence : {base_proba:.1%}", color),
             ("Niveau de risque", level, f"Seuil de décision : {thr:.1%}", color),
             ("Décision suggérée", decision, "Aide à la décision, non contraignante", color),
             ("Qualité du modèle", f"AUC {m['roc_auc']:.3f}", f"Évalué sur {bundle['metadata']['n_test']:,} dossiers test".replace(",", " "), P["primary"])]
    for col, args in zip(c, cards):
        col.markdown(kpi_card(*args), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1, 1], gap="medium")
    with left, card("gauge"):
        section_title("Score de risque", "Zones : faible · modérée · élevée (trait noir = seuil)")
        st.plotly_chart(gauge_chart(proba, thr, color), use_container_width=True, config={"displayModeBar": False})
    with right, card("drivers"):
        section_title("Facteurs déterminants", "Principales contributions SHAP pour ce dossier")
        chips = "".join(chip(r.label, "up" if r.shap > 0 else "down") for r in contrib.head(5).itertuples())
        st.markdown(chips, unsafe_allow_html=True)
        direction = "supérieur" if proba > base_proba else "inférieur"
        st.markdown(f'<div class="note">Le risque estimé de <b>{proba:.1%}</b> est {direction} à la moyenne du portefeuille '
                    f'(<b>{base_proba:.1%}</b>). Les éléments en rouge ▲ augmentent le risque, ceux en vert ▼ le réduisent. '
                    f'Détail complet dans l\'onglet « Explicabilité SHAP ».</div>', unsafe_allow_html=True)
        st.write("")
        rows = [(pretty(k), f"{v:,.2f}".rstrip("0").rstrip(".").replace(",", " ") if isinstance(v, (int, float, np.number)) else str(v))
                for k, v in df.iloc[0].items()]
        st.markdown(kv_rows(rows[:6]), unsafe_allow_html=True)

    if sample is not None and "pred_proba" in sample:
        st.write("")
        with card("dist"):
            section_title("Positionnement dans le portefeuille", "Distribution des scores sur l'échantillon de référence")
            st.plotly_chart(distribution_chart(sample, proba), use_container_width=True, config={"displayModeBar": False})


def tab_shap(bundle, contrib, base_proba, proba, sample) -> None:
    left, right = st.columns([3, 2], gap="medium")
    with left, card("local"):
        section_title("Explication locale", "Comment chaque variable déplace le score de la valeur de base à la prédiction")
        n = st.slider("Nombre de variables affichées", 5, min(20, len(contrib)), min(10, len(contrib)), key="top_n")
        st.plotly_chart(local_shap_chart(contrib, n), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Valeur de base : {base_proba:.1%} (taux moyen) → prédiction : {proba:.1%}. "
                   "Les contributions sont exprimées en log-odds (additives).")
    with right, card("global"):
        section_title("Importance globale", "Moyenne de |SHAP| sur l'échantillon de référence")
        if sample is None:
            st.info("Échantillon de référence indisponible : importance globale non calculée.")
        else:
            try:
                feats = sample[list(bundle["schema"]["numeric"]) + list(bundle["schema"]["categorical"])]
                st.plotly_chart(global_chart(global_importance(bundle, feats)), use_container_width=True,
                                config={"displayModeBar": False})
            except Exception as exc:
                log.exception("Importance globale")
                st.warning(f"Importance globale indisponible : {exc}")

    st.write("")
    with card("table"):
        section_title("Détail des contributions", "Toutes les variables, triées par impact absolu")
        view = contrib[["label", "value", "shap"]].rename(columns={"label": "Variable", "value": "Valeur", "shap": "Contribution (log-odds)"})
        st.dataframe(view, use_container_width=True, hide_index=True, height=300,
                     column_config={"Contribution (log-odds)": st.column_config.NumberColumn(format="%+.4f")})


def tab_data(bundle, df, sample) -> None:
    m, meta = bundle["metrics"], bundle["metadata"]
    c = st.columns(4, gap="medium")
    for col, args in zip(c, [("ROC AUC", f"{m['roc_auc']:.3f}", f"CV : {meta['cv_best_auc']:.3f}"),
                             ("PR AUC", f"{m['pr_auc']:.3f}", f"Taux de base : {m['base_rate']:.1%}"),
                             ("F1 @ seuil", f"{m['f1']:.3f}", f"P {m['precision']:.2f} · R {m['recall']:.2f}"),
                             ("Brier", f"{m['brier']:.4f}", "Calibration (plus bas = mieux)")]):
        col.markdown(kpi_card(*args), unsafe_allow_html=True)
    st.write("")

    with card("sample"):
        section_title("Échantillon de référence", "Dossiers de test avec score du modèle")
        if sample is None:
            st.info("Aucun échantillon de référence trouvé dans artifacts/.")
        else:
            st.dataframe(sample, use_container_width=True, hide_index=True, height=340)
            st.download_button("Télécharger l'échantillon (CSV)", sample.to_csv(index=False).encode("utf-8"),
                               "reference_sample.csv", "text/csv")
    st.write("")
    a, b = st.columns(2, gap="medium")
    with a, card("input"):
        section_title("Dossier saisi", "Valeurs envoyées au modèle")
        st.dataframe(df.T.rename(columns={0: "Valeur"}).astype(str), use_container_width=True)
    with b, card("meta"):
        section_title("Métadonnées d'entraînement")
        st.markdown(kv_rows([("Entraîné le", meta["trained_at"][:19].replace("T", " ") + " UTC"),
                             ("Train / Val / Test", f"{meta['n_train']} / {meta['n_val']} / {meta['n_test']}"),
                             ("Arbres retenus", str(meta["best_iteration"])),
                             ("XGBoost", meta["versions"]["xgboost"])]), unsafe_allow_html=True)
        with st.expander("Hyperparamètres"):
            st.code(json.dumps(meta["best_params"], indent=2), language="json")


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Credit Risk Studio", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    try:
        bundle = load_bundle()
    except ModelLoadError as exc:
        header("Credit Risk Studio", "Scoring & explicabilité locale", "Modèle indisponible")
        st.error(str(exc))
        st.stop()

    sample = load_sample()
    values = render_sidebar(bundle["schema"])

    try:
        df = validate_inputs(values, bundle["schema"])
        proba, X = score(bundle, df)
        explainer = get_explainer(bundle["model"])
        sv, base_value = shap_values(explainer, X)
    except InputValidationError as exc:
        st.error(f"Saisie invalide : {exc}")
        st.stop()
    except Exception as exc:  # erreur inattendue : message propre + trace dans les logs
        log.exception("Échec du scoring")
        st.error(f"Erreur lors du scoring : {exc}")
        st.stop()

    base_proba = float(1 / (1 + np.exp(-base_value)))
    contrib = contributions(bundle, X, sv)

    header("Credit Risk Studio", "Scoring de défaut et explicabilité locale · XGBoost + SHAP",
           f"Modèle du {bundle['metadata']['trained_at'][:10]}")
    t1, t2, t3 = st.tabs(["Vue d'ensemble", "Explicabilité SHAP", "Données & modèle"])
    with t1:
        tab_overview(bundle, df, proba, base_proba, contrib, sample)
    with t2:
        tab_shap(bundle, contrib, base_proba, proba, sample)
    with t3:
        tab_data(bundle, df, sample)


if __name__ == "__main__":
    main()
