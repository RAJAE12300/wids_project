# 🏥 WiDS Datathon 2020 — ICU Mortality Prediction
## Big Data Engineering · PhD Research Project

---

## 📁 Project Folder Structure

```
your_project/
│
├── data/                          ← PUT YOUR CSV FILES HERE
│   ├── training_v2.csv            (63 MB  — labeled training data)
│   ├── unlabeled.csv              (27 MB  — test data to predict)
│   ├── WiDS_Datathon_2020_Dictionary.csv  (feature descriptions)
│   ├── solution_template.csv      (39,308 encounter IDs to fill)
│   └── samplesubmission.csv      (format reference)
│
├── WiDS_2020_Full_Pipeline.ipynb  ← MAIN NOTEBOOK (start here)
├── requirements.txt               ← Python dependencies
├── README.md                      ← This file
│
└── outputs/                       ← Generated automatically
    ├── submission_hybrid_ensemble.csv
    ├── eda_class_imbalance.png
    ├── eda_missing_data.png
    ├── eda_distributions.png
    ├── eda_correlation.png
    ├── eda_categorical.png
    ├── roc_curves_all_models.png
    ├── shap_beeswarm_xgb.png
    ├── shap_bar_xgb.png
    ├── shap_waterfall_survivor.png
    ├── shap_waterfall_deceased.png
    └── calibration_curves.png
```

---

## ⚡ Quick Start (3 steps)

### Step 1 — Create data folder and add your CSV files
```bash
mkdir data
# Copy your 5 CSV files into the data/ folder
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Open and run the notebook
```bash
jupyter notebook WiDS_2020_Full_Pipeline.ipynb
```
Run cells top to bottom. Each phase is clearly labeled.

---

## 📓 Notebook Structure

| Phase | Section | What it does |
|-------|---------|--------------|
| 1 | Setup & Loading | Loads all 5 CSV files, builds feature registry |
| 2 | EDA | Class imbalance, missing data, distributions, correlations |
| 3 | Preprocessing | Imputation, encoding, SMOTE, feature engineering |
| 4A | ML Models | Logistic Regression, Random Forest, LightGBM, XGBoost |
| 4B | DL Models | MLP (4-layer), TabTransformer (attention-based) |
| 4C | Hybrid Ensemble | Stacked XGBoost + LightGBM + MLP — novel contribution |
| 5 | SHAP / XAI | Beeswarm, waterfall, calibration plots |
| 6 | Submission | Generate submission_hybrid_ensemble.csv |

---

## 🎯 Task Summary

- **Goal:** Predict `hospital_death` (0=survived, 1=died) for 39,308 ICU patients
- **Metric:** AUC-ROC (NOT accuracy — dataset is imbalanced ~8-12% positive)
- **Baseline to beat:** APACHE 4A clinical score (column `apache_4a_hospital_death_prob`)
- **Input:** 188 clinical features — vitals, labs, APACHE scores, demographics, comorbidities
- **Output:** `submission_hybrid_ensemble.csv` — encounter_id + hospital_death probability

---

## 🔧 Feature Engineering Created

| Feature | Formula | Clinical Meaning |
|---------|---------|-----------------|
| `shock_index_d1` | HR_max / SBP_min | High = shock risk |
| `pulse_pressure_d1` | SBP_max - DBP_min | Low = poor perfusion |
| `heartrate_range_d1` | HR_max - HR_min | Variability = instability |
| `sysbp_range_d1` | SBP_max - SBP_min | BP variability |
| `resprate_range_d1` | RR_max - RR_min | Respiratory instability |
| `gcs_total` | GCS eyes + motor + verbal | Total neurological score |
| `bun_creatinine_ratio` | BUN_max / Creatinine_max | Kidney function |

---

## 🤖 Models Trained

| # | Model | Type | Imbalance Handling |
|---|-------|------|--------------------|
| 1 | Logistic Regression | Classic ML | class_weight='balanced' |
| 2 | Random Forest | Ensemble ML | class_weight='balanced' |
| 3 | LightGBM | Gradient Boosting | is_unbalance=True |
| 4 | XGBoost | Gradient Boosting | scale_pos_weight |
| 5 | MLP (4-layer) | Deep Learning | BCEWithLogitsLoss + pos_weight |
| 6 | TabTransformer | Deep Learning | BCEWithLogitsLoss + pos_weight |
| 7 ⭐ | Hybrid Ensemble | Stacked (Novel) | Inherits from level-1 models |

---

## 💡 Important Notes

1. **NEVER apply SMOTE to test data** — only train
2. **NEVER use accuracy as metric** — always AUC-ROC for imbalanced data
3. **scale_pos_weight** for XGBoost = (# negatives) / (# positives)
4. **Scaler fit on train only**, then transform test — prevents data leakage
5. **Early stopping** on XGBoost/LightGBM uses validation AUC
6. **SHAP** works best with XGBoost (TreeExplainer) — fastest & most accurate
7. Set `DATA_DIR = './data/'` in cell 1.2 to match your folder

---

## 📊 Expected Results

| Model | Expected Val AUC |
|-------|-----------------|
| APACHE 4A (baseline) | ~0.87 |
| Logistic Regression | 0.82–0.85 |
| Random Forest | 0.84–0.87 |
| LightGBM | 0.86–0.89 |
| XGBoost | 0.87–0.90 |
| MLP | 0.85–0.88 |
| TabTransformer | 0.87–0.90 |
| **Hybrid Ensemble ⭐** | **> 0.91** |

---

## 📄 For Your Article (TRIPOD)

Your paper reports:
- Dataset: WiDS Datathon 2020 (Stanford / MIT Lab)
- Train size, test size, positive rate
- Preprocessing steps (imputation, SMOTE, encoding)
- All 7 model results in a comparison table
- SHAP figures as article figures
- Comparison vs APACHE 4A baseline

**Target journal:** Computer Methods and Programs in Biomedicine (Elsevier)
