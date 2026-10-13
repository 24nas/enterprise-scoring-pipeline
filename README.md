# Credit Risk Studio
https://enterprise-scoring-pipeline.streamlit.app/

Scoring de risque de défaut (XGBoost) avec explicabilité locale et globale (SHAP), servi par un dashboard Streamlit.

## Arborescence

```
credit-risk-studio/
├── .streamlit/config.toml               
├── app.py                   
├── config.py                
├── theme.py                 
├── train.py                 
├── requirements.txt
└── README.md
```

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py                                    # données synthétiques
python train.py --data mes_donnees.csv --target defaut --n-iter 40

streamlit run app.py
```

## Choix techniques

- **Préprocessing** : imputation médiane + indicateurs de manquants, mode + one-hot tolérant aux modalités inconnues.
- **Optimisation** : `RandomizedSearchCV` sur pipeline complet (pas de fuite), refit avec early stopping.
- **Seuil** : choisi sur la validation (F1), métriques finales sur un test jamais utilisé pour l'ajustement.
- **Probabilités** : pas de `scale_pos_weight`, afin de conserver des probabilités calibrées pour l'affichage.
- **Artefacts** : bundle `joblib` (préprocesseur, modèle, schéma, seuil, métriques), modèle natif `.json`, `metrics.json`, échantillon de référence.
