"""Configuration centralisée : chemins, constantes et libellés métier."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.joblib"   # préprocesseur + modèle + schéma
XGB_NATIVE_PATH = ARTIFACTS_DIR / "xgb_model.json"    # format natif XGBoost (portable)
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"         # métriques lisibles par la CI
SAMPLE_PATH = ARTIFACTS_DIR / "reference_sample.csv"  # échantillon de référence

RANDOM_STATE = 42
DEFAULT_TARGET = "default"

# Libellés d'affichage (fallback automatique pour toute autre colonne)
FEATURE_LABELS = {
    "age": "Âge",
    "annual_income": "Revenu annuel (€)",
    "loan_amount": "Montant du prêt (€)",
    "loan_term_months": "Durée du prêt (mois)",
    "credit_history_years": "Ancienneté crédit (ans)",
    "num_open_accounts": "Comptes ouverts",
    "debt_to_income": "Taux d'endettement",
    "num_late_payments": "Retards de paiement",
    "employment_type": "Situation professionnelle",
    "housing_status": "Logement",
    "loan_purpose": "Objet du prêt",
    "region": "Région",
}


def pretty(col: str) -> str:
    """Retourne un libellé lisible pour une colonne."""
    return FEATURE_LABELS.get(col, col.replace("_", " ").capitalize())
