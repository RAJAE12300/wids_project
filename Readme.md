# 🏥 ICU Mortality Prediction — WiDS Datathon 2020

> Prédiction de la mortalité hospitalière en soins intensifs à partir de données cliniques massives, avec un ensemble hybride qui dépasse le score clinique de référence (APACHE 4A) de +6.2 points d'AUC.

![badge-python](https://img.shields.io/badge/-Python%203.10-3776AB?style=flat-square&logo=python&logoColor=white)
![badge-xgboost](https://img.shields.io/badge/-XGBoost-005571?style=flat-square)
![badge-lightgbm](https://img.shields.io/badge/-LightGBM-9ACD32?style=flat-square)
![badge-pytorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![badge-shap](https://img.shields.io/badge/-SHAP-8A2BE2?style=flat-square)
![badge-status](https://img.shields.io/badge/status-completed-brightgreen?style=flat-square)

## 🎯 Le problème

Les systèmes de scoring clinique classiques (APACHE II/III/4A) utilisés pour prédire la mortalité en soins intensifs reposent sur des modèles linéaires entraînés sur de petites cohortes historiques du début des années 2000. Ils ne capturent pas les interactions non-linéaires entre variables cliniques et n'exploitent pas la richesse des dossiers médicaux électroniques (EHR) modernes. Une prédiction précoce et fiable de la mortalité permet une escalade clinique à temps, une meilleure allocation des ressources, et des discussions pronostiques mieux informées avec les familles.

## 💡 La solution

Un pipeline Big Data complet comparant **7 modèles** (Machine Learning et Deep Learning) sur le dataset **WiDS Datathon 2020** (Stanford / MIT / GOSSIS), avec pour contribution principale un **Hybrid Stacked Ensemble** combinant XGBoost, LightGBM et Random Forest via un méta-apprenant en régression logistique.

## 📊 Dataset

- **Source** : WiDS Datathon 2020 (Stanford University / MIT Laboratory for Computational Physiology, initiative GOSSIS)
- **Volume** : 91 713 admissions ICU labellisées, 184 variables cliniques (valeurs de laboratoire, signes vitaux, covariables APACHE, démographie, comorbidités)
- **Déséquilibre de classe** : 91.4% survivants vs 8.6% décès — traité avec **SMOTE** (sampling_strategy=0.3) et pondération de classe
- **Split** : 80% entraînement (73 370) / 20% validation stratifiée (18 343), seed=42

## 🧠 Modèles comparés

| Modèle | Type | AUC-ROC |
|---|---|---|
| **Hybrid Stacked Ensemble** ⭐ | XGBoost + LightGBM + RF (stacking) | **0.9026** |
| XGBoost | Gradient Boosting | 0.9018 |
| MLP (4 couches) | Deep Learning | 0.8893 |
| Random Forest | Ensemble ML | 0.8879 |
| Régression Logistique | ML classique | 0.8876 |
| LightGBM | Gradient Boosting | 0.8728 |
| TabTransformer | Deep Learning (attention) | 0.8656 |
| *APACHE 4A (référence clinique)* | *Score clinique* | *0.8407* |

**Tous les modèles dépassent la baseline clinique APACHE 4A**, l'ensemble hybride apportant un gain de +6.2 points d'AUC.

## 🔬 Feature engineering clinique

7 variables composites dérivées pour capturer des patterns physiologiques :

- **Shock Index** = FC max / PAS min → capture le compromis hémodynamique
- **Pulse Pressure** = PAS max − PAD min
- **Vital Ranges** (FC, PAS, PAD, FR, Temp, SpO2) → instabilité physiologique sur 24h
- **GCS total** (score de Glasgow, 3–15) → statut neurologique agrégé
- **Ratio BUN/Créatinine** → marqueur de fonction rénale

➡️ `shock_index_d1` s'est classé **5ᵉ prédicteur global** en importance SHAP, validant l'intérêt du feature engineering clinique.

## 🔍 Explicabilité (SHAP)

Analyse SHAP (TreeExplainer sur XGBoost) :
- **Top-5 prédicteurs globaux** : `apache_4a_hospital_death_prob`, `apache_4a_icu_death_prob`, `age`, `ventilated_apache`, `shock_index_d1`
- Graphiques waterfall patient-par-patient pour 3 cas représentatifs (survivant bas-risque, décès haut-risque, cas limite) — permettant une explication clinique de chaque prédiction individuelle

## ⚠️ Limites identifiées

- Risque de fuite temporelle : certaines variables agrégées sur 24h peuvent refléter une dégradation déjà en cours plutôt que la prédire
- LightGBM présente un problème de calibration (F1=0 à seuil 0.5 malgré un bon AUC) — nécessite une calibration post-hoc (Platt scaling / isotonic regression)
- Prédiction statique (fenêtre d'admission) plutôt que mise à jour dynamique pendant le séjour

## 🚀 Pistes d'amélioration

- Validation externe sur MIMIC-IV et eICU
- Calibration post-hoc pour LightGBM et XGBoost
- Optimisation du seuil via Decision Curve Analysis
- Intégration de notes cliniques (NLP) pour une prédiction multimodale
- Évaluation de l'équité entre sous-groupes démographiques (âge, sexe, origine ethnique)

## 🛠️ Stack technique

- **Langage** : Python 3.10
- **ML** : scikit-learn 1.2, XGBoost 1.7, LightGBM 3.3
- **Deep Learning** : PyTorch 2.0 (MLP, TabTransformer)
- **Explicabilité** : SHAP 0.42
- **Prétraitement** : imputation médiane, SMOTE, one-hot encoding, StandardScaler

## 🚀 Installation

```bash
git clone https://github.com/TON_USERNAME/icu-mortality-prediction.git
cd icu-mortality-prediction
pip install -r requirements.txt
```

Dataset disponible publiquement sur Kaggle : [WiDS Datathon 2020](https://www.kaggle.com/c/widsdatathon2020)

## 👥 Auteure

- Rajae Fdili — Département Big Data Engineering, UEMF, Fès, Maroc

## 📄 Licence

MIT