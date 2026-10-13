"""Entraînement d'un modèle XGBoost de scoring de risque de défaut.

Usage :
    python train.py                                   # jeu de données synthétique
    python train.py --data data.csv --target default  # jeu de données réel
    python train.py --n-iter 40 --output-dir artifacts

Pipeline : nettoyage -> split train/val/test -> préprocessing (imputation +
one-hot) -> recherche aléatoire d'hyperparamètres (CV stratifiée) -> refit avec
early stopping -> seuil de décision (F1 sur validation) -> évaluation test
-> sauvegarde des artefacts.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def generate_synthetic_credit_data(n: int = 20_000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Génère un jeu de données de crédit réaliste (mixte, avec valeurs manquantes)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 71, n)
    income = np.clip(rng.lognormal(10.9, 0.45, n), 15_000, 400_000).round(-2)
    loan_amount = np.clip(income * rng.uniform(0.1, 1.2, n), 1_000, 300_000).round(-2)
    term = rng.choice([12, 24, 36, 48, 60, 84], n)
    history = np.clip(age - 18 - rng.exponential(5, n), 0, 45).round(1)
    open_acc = rng.poisson(4, n)
    late = rng.poisson(np.clip(1.2 - history / 40, 0.1, None))
    dti = np.clip(rng.normal(0.28, 0.10, n) + 0.15 * loan_amount / income, 0.02, 0.9).round(3)
    employment = rng.choice(["Salarié CDI", "Salarié CDD", "Indépendant", "Fonctionnaire", "Sans emploi"],
                            n, p=[0.50, 0.10, 0.20, 0.15, 0.05])
    housing = rng.choice(["Propriétaire", "Locataire", "Hébergé"], n, p=[0.45, 0.45, 0.10])
    purpose = rng.choice(["Immobilier", "Automobile", "Consommation", "Travaux", "Professionnel"], n)
    region = rng.choice(["Île-de-France", "Auvergne-Rhône-Alpes", "PACA", "Occitanie", "Grand Est"], n)

    emp_effect = pd.Series(employment).map({"Salarié CDI": -0.3, "Salarié CDD": 0.4, "Indépendant": 0.3,
                                            "Fonctionnaire": -0.5, "Sans emploi": 1.4}).to_numpy()
    house_effect = pd.Series(housing).map({"Propriétaire": -0.3, "Locataire": 0.1, "Hébergé": 0.3}).to_numpy()
    z = (3.0 * dti + 0.45 * late + 1.2 * loan_amount / income - 0.04 * history
         + 0.004 * term + emp_effect + house_effect + rng.normal(0, 0.4, n))

    # Calibrage de l'intercept pour obtenir ~12 % de défauts
    lo, hi = -15.0, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > 0.12:
            hi = mid
        else:
            lo = mid
    default = rng.random(n) < 1 / (1 + np.exp(-(z + (lo + hi) / 2)))

    df = pd.DataFrame({
        "age": age, "annual_income": income, "loan_amount": loan_amount, "loan_term_months": term,
        "credit_history_years": history, "num_open_accounts": open_acc, "debt_to_income": dti,
        "num_late_payments": late, "employment_type": employment, "housing_status": housing,
        "loan_purpose": purpose, "region": region, config.DEFAULT_TARGET: default.astype(int),
    })
    for col, rate in [("annual_income", 0.03), ("credit_history_years", 0.05), ("debt_to_income", 0.02)]:
        df.loc[rng.random(n) < rate, col] = np.nan
    return df


def load_dataset(path: str | None, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Charge, nettoie et valide le jeu de données. Retourne (X, y binaire)."""
    if path is None:
        log.info("Aucun --data fourni : génération du jeu de données synthétique.")
        df = generate_synthetic_credit_data()
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable : {p}")
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' absente. Colonnes : {list(df.columns)}")

    n0 = len(df)
    df = df.drop_duplicates().dropna(subset=[target]).reset_index(drop=True)
    log.info("Lignes : %d -> %d après suppression des doublons / cibles manquantes", n0, len(df))

    classes = sorted(df[target].unique())
    if len(classes) != 2:
        raise ValueError(f"La cible doit être binaire, {len(classes)} classes trouvées : {classes}")
    y = (df[target] == classes[1]).astype(int)

    X = df.drop(columns=[target])
    useless = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]  # vides ou constantes
    if useless:
        log.warning("Colonnes inutiles supprimées : %s", useless)
        X = X.drop(columns=useless)
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
    for c in cat_cols:  # normalisation des manquants pour SimpleImputer
        X[c] = X[c].astype("object").where(X[c].notna(), np.nan)
    return X, y


def infer_schema(X: pd.DataFrame) -> dict:
    """Décrit les colonnes (bornes, médiane, modalités) pour piloter l'UI."""
    schema: dict = {"numeric": {}, "categorical": {}}
    for c in X.select_dtypes(include=[np.number]).columns:
        s = X[c].dropna()
        schema["numeric"][c] = {"min": float(s.min()), "max": float(s.max()), "median": float(s.median()),
                                "is_integer": bool(np.allclose(s, s.round()))}
    for c in X.select_dtypes(exclude=[np.number]).columns:
        vc = X[c].dropna().astype(str).value_counts()
        schema["categorical"][c] = {"categories": sorted(vc.index.tolist()), "default": str(vc.index[0])}
    return schema


# --------------------------------------------------------------------------- #
# Modèle
# --------------------------------------------------------------------------- #
def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Imputation médiane (+ indicateurs de manquants) ; mode + one-hot robuste aux modalités inconnues."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)],
                             verbose_feature_names_out=False)


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Seuil maximisant le F1 (calculé sur le jeu de validation, jamais sur le test)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1))])


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba), "log_loss": log_loss(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
        "threshold": threshold, "base_rate": float(np.mean(y_true)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None, help="CSV/Parquet d'entraînement (défaut : données synthétiques)")
    p.add_argument("--target", default=config.DEFAULT_TARGET, help="Nom de la colonne cible binaire")
    p.add_argument("--n-iter", type=int, default=25, help="Itérations de la recherche d'hyperparamètres")
    p.add_argument("--cv", type=int, default=4, help="Nombre de folds de validation croisée")
    p.add_argument("--output-dir", default=str(config.ARTIFACTS_DIR), help="Dossier de sortie des artefacts")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data, args.target)
    schema = infer_schema(X)
    num_cols, cat_cols = list(schema["numeric"]), list(schema["categorical"])
    log.info("Features : %d numériques, %d catégorielles | taux positif : %.2f%%",
             len(num_cols), len(cat_cols), 100 * y.mean())

    # Split stratifié 70 / 15 / 15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=config.RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp,
                                                random_state=config.RANDOM_STATE)

    # 1) Recherche d'hyperparamètres (pipeline complet : pas de fuite entre folds)
    search_pipe = Pipeline([
        ("prep", build_preprocessor(num_cols, cat_cols)),
        ("model", xgb.XGBClassifier(n_estimators=300, tree_method="hist", eval_metric="auc",
                                    n_jobs=-1, random_state=config.RANDOM_STATE)),
    ])
    space = {
        "model__max_depth": randint(3, 9), "model__learning_rate": uniform(0.02, 0.18),
        "model__subsample": uniform(0.6, 0.4), "model__colsample_bytree": uniform(0.5, 0.5),
        "model__min_child_weight": randint(1, 10), "model__gamma": uniform(0, 2),
        "model__reg_lambda": uniform(0.5, 5), "model__reg_alpha": uniform(0, 1),
    }
    search = RandomizedSearchCV(
        search_pipe, space, n_iter=args.n_iter, scoring="roc_auc",
        cv=StratifiedKFold(args.cv, shuffle=True, random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE, n_jobs=1, verbose=1,
    )
    log.info("Recherche d'hyperparamètres (%d itérations x %d folds)...", args.n_iter, args.cv)
    search.fit(X_tr, y_tr)
    best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    log.info("Meilleur AUC CV : %.4f | params : %s", search.best_score_, best)

    # 2) Refit final avec early stopping sur la validation
    prep = build_preprocessor(num_cols, cat_cols)
    A_tr, A_val, A_te = prep.fit_transform(X_tr), prep.transform(X_val), prep.transform(X_te)
    feature_names = [str(f) for f in prep.get_feature_names_out()]
    model = xgb.XGBClassifier(**best, n_estimators=2000, early_stopping_rounds=50, tree_method="hist",
                              eval_metric="auc", n_jobs=-1, random_state=config.RANDOM_STATE)
    model.fit(A_tr, y_tr, eval_set=[(A_val, y_val)], verbose=False)
    log.info("Arbre(s) retenus : %d (best iteration)", model.best_iteration + 1)

    # 3) Seuil (validation) puis évaluation honnête sur le test
    threshold = best_f1_threshold(y_val.to_numpy(), model.predict_proba(A_val)[:, 1])
    proba_te = model.predict_proba(A_te)[:, 1]
    metrics = evaluate(y_te.to_numpy(), proba_te, threshold)
    log.info("TEST | AUC=%.4f | PR-AUC=%.4f | Brier=%.4f | F1=%.3f @ seuil %.3f",
             metrics["roc_auc"], metrics["pr_auc"], metrics["brier"], metrics["f1"], threshold)

    # 4) Sauvegarde
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": args.target, "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_te),
        "best_iteration": int(model.best_iteration + 1), "cv_best_auc": float(search.best_score_),
        "best_params": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in best.items()},
        "versions": {"xgboost": xgb.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }
    bundle = {"preprocessor": prep, "model": model, "feature_names": feature_names, "schema": schema,
              "threshold": threshold, "metrics": metrics, "metadata": metadata}
    joblib.dump(bundle, out / config.BUNDLE_PATH.name, compress=3)
    model.save_model(out / config.XGB_NATIVE_PATH.name)
    (out / config.METRICS_PATH.name).write_text(
        json.dumps({"metrics": metrics, "metadata": metadata}, indent=2, ensure_ascii=False), encoding="utf-8")

    sample = X_te.sample(n=min(1000, len(X_te)), random_state=config.RANDOM_STATE).copy()
    sample["target"] = y_te.loc[sample.index].to_numpy()
    sample["pred_proba"] = model.predict_proba(prep.transform(sample[num_cols + cat_cols]))[:, 1]
    sample.to_csv(out / config.SAMPLE_PATH.name, index=False)

    log.info("Artefacts écrits dans %s", out.resolve())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)
