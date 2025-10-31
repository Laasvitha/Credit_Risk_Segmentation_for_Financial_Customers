# Deep Learning-Based Credit Risk Segmentation for Financial Customers

## Overview

This project implements a **deep learning-based credit risk segmentation model** using **CatBoost** (a gradient boosting framework optimized for tabular data), trained on the Statlog (German Credit Data) dataset. The model estimates **Probability of Default (PD)** for financial customers and segments them into three risk tiers: **Low Risk**, **Sloppy Payer**, and **High Risk**. This enables automated decision-making in lending, such as auto-approval for low-risk applicants, manual review for sloppy payers, and decline for high-risk ones.

The segmentation is based on calibrated PD thresholds optimized for expected misclassification cost (with FN=5×FP, per Statlog standards). The model incorporates calibration (Isotonic or Platt scaling), fairness checks (e.g., ΔTPR/ΔFPR by sex proxy), and stability monitoring (PSI for drift).

### Project Objectives
- **Preprocess** the credit dataset, handling mixed categorical/numerical features and class imbalance.
- **Train** a CatBoost model to predict PD (P(bad credit outcome)).
- **Calibrate** probabilities and select thresholds to minimize expected cost on validation data.
- **Segment** customers into risk tiers: 
  1. **Low Risk**: High creditworthiness, no missed payments (PD ≤ low threshold, e.g., ~0.05).
  2. **Sloppy Payer**: Occasional late payments but not defaulting (low < PD ≤ high threshold, e.g., ~0.20).
  3. **High Risk**: Frequent late payments or defaults (PD > high threshold).
- **Evaluate** performance via AUC, expected cost, PSI (stability), and fairness metrics.
- **Generate** a model card for governance and compute tiered expected loss (EL) using EAD (Exposure at Default) and LGD (Loss Given Default) assumptions.

This is a **tabular classification task** with a focus on interpretability, calibration, and production readiness. The implementation is in a Jupyter Notebook (`Deep Learning-Based Credit Risk Segmentation for Financial Customers.ipynb`), allowing easy experimentation, metric visualization, and extension (e.g., SHAP for explainability).

### Key Features
- **Dataset**: Statlog German Credit (1,000 rows, 20 features; ~30% bad outcomes).
- **Model**: CatBoost with early stopping, class weights for imbalance, and categorical feature support.
- **Calibration & Thresholding**: Automatic selection of Isotonic/Platt and cost-minimizing thresholds.
- **Segmentation**: Three-tier policy with monotonic default rates.
- **Monitoring**: PSI for drift, fairness snapshot, auto-generated model card.
- **Extensibility**: Includes EL computation and governance plan; add SHAP or challengers (e.g., LightGBM).

### Why CatBoost?
CatBoost excels on tabular data with native handling of categoricals, robust to overfitting via ordered boosting, and strong out-of-the-box performance. It's more efficient than deep learning (e.g., neural nets) for small datasets like this, while providing built-in calibration and explainability—ideal for regulated domains like credit risk.

## Dataset

### Source
- **Name**: Statlog (German Credit Data)
- **UCI Link**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Kaggle Link** (CSV ready): [Kaggle - German Credit](https://www.kaggle.com/datasets/uciml/german-credit)
- **Format**: CSV (via `ucimlrepo` or direct download); 1,000 rows × 20 features + target.
- **Size**: Small (1K samples), balanced for quick prototyping; scalable to larger credit portfolios.
- **Structure** (Anonymized attributes; target: 1=good, 2=bad → mapped to 0=good, 1=bad):
  | Column          | Description                          | Type     | Example Value |
  |-----------------|--------------------------------------|----------|---------------|
  | `Attribute1`   | Status of existing checking account | Categorical | A11 (≤0 DM) |
  | `Attribute2`   | Duration in months                  | Numerical | 6            |
  | `Attribute3`   | Credit history                      | Categorical | A34 (critical) |
  | ... (up to 20) | ... (e.g., purpose, credit amount, employment, etc.) | Mixed   | ...          |
  | Target (y)     | Credit risk (1=good, 2=bad)         | Binary   | 1            |

- **Content**: Real anonymized German credit applications from 1994. Features include account status, loan purpose, employment duration, etc. ~70% good (0), 30% bad (1).
- **Preprocessing Notes**:
  - 13/20 features are categorical (object dtype); auto-detected for CatBoost.
  - No missing values; stratified splits preserve class balance.
  - Sample: `Attribute1=A11, Attribute2=6, Attribute3=A34, ...` (low-risk profile).
- **Ethical Considerations**: Dataset is anonymized and public domain (UCI). Fairness checked via sex proxy (Attribute9: personal status/sex). Monitor for bias in production; complies with GDPR-like standards.

### Data Loading
The notebook uses `ucimlrepo.fetch_ucirepo(id=144)` for seamless loading. Download CSV from Kaggle if preferred and load via `pd.read_csv()`.

## Requirements

### Dependencies
The project uses Python 3.10+ and the following libraries:

| Library          | Version (as used) | Purpose |
|------------------|-------------------|---------|
| `numpy`         | (latest)         | Numerical computations. |
| `pandas`        | (latest)         | Data loading/manipulation (CSV parsing, grouping). |
| `ucimlrepo`     | (latest)         | UCI dataset fetching. |
| `catboost`      | (latest)         | Gradient boosting model with categorical support. |
| `sklearn`       | (latest)         | Splits, metrics (AUC, confusion matrix), calibration (Isotonic, LogisticRegression). |

### Environment Setup
- **Python Version**: 3.10.11 (tested).
- **No GPU Required**: CatBoost runs on CPU; enable GPU for large-scale training.

## Installation

1. **Clone or Download the Project**:
   ```
   git clone <your-repo-url>  # If hosted on GitHub
   # Or download the ZIP and extract.
   ```

2. **Download the Dataset** (Optional; `ucimlrepo` fetches automatically):
   - Use Kaggle link for CSV; place in project root as `german_credit.csv`.

3. **Set Up Virtual Environment** (Recommended):
   ```
   python -m venv credit-risk-env
   source credit-risk-env/bin/activate  # On macOS/Linux
   # Or on Windows: credit-risk-env\Scripts\activate
   ```

4. **Install Dependencies**:
   ```
   pip install ucimlrepo catboost scikit-learn pandas numpy
   ```
   - If using Jupyter:
     ```
     pip install notebook
     ```

5. **Verify Installation**:
   - Run `python -c "from ucimlrepo import fetch_ucirepo; print('OK')"` (fetches dataset without errors).

## Usage

### Running the Notebook
1. **Launch Jupyter**:
   ```
   jupyter notebook
   ```
   - Open `Deep Learning-Based Credit Risk Segmentation for Financial Customers.ipynb`.

2. **Execute Cells Step-by-Step**:
   - **Cell 1**: Loads dataset via UCI; prints summary and previews X (features).
   - **Cell 2**: Stratified split (70/15/15); detects categoricals (13 indices).
   - **Cell 3**: Trains CatBoost (2000 iterations, early stopping); prints val/test AUC.
   - **Cell 4-5**: Imports calibration tools.
   - **Cell 6**: Fits Isotonic/Platt on val; selects best by val cost.
   - **Cell 7**: Optimizes thresholds (t_low, t_high) for 3-tier segmentation via val cost.
   - **Cell 8**: Plots calibration curves (Brier decomposition).
   - **Cell 9**: Computes PSI (stability); prints summary.
   - **Cell 10**: Fairness snapshot (ΔTPR/ΔFPR/selection by sex proxy).
   - **Cell 11**: Generates/saves model card Markdown.
   - **Cell 12**: Computes tiered EL (assumes LGD=0.6, EAD=Attribute5).
   - **Cell 13**: Project summary (metrics, policy).

3. **Expected Outputs**:
   - Dataset: 1,000 rows, 700 good/300 bad.
   - AUC: Val ~0.7865, Test ~0.8195.
   - Thresholds: e.g., t=0.1632 (cost=77 on test: FP=52, FN=5).
   - Tiers: Low (n=34, PD~4.6%), Sloppy (n=47, PD~16.0%), High (n=69, PD~45.5%).
   - PSI: ~0.0888 (stable).
   - Fairness: ΔTPR=0.081, ΔFPR=0.071, ΔSelection=0.054.
   - Model Card: Saved to `reports/model_card.md`.
   - EL: Portfolio total ~97K (sum across tiers).

### Training the Model
- **Customization**:
  - Hyperparams: Edit `depth=6`, `learning_rate=0.05`, `iterations=2000` in CatBoost.
  - Costs: Change `C_FP=1`, `C_FN=5` for different risk appetites.
  - Imbalance: Class weights auto-computed; add `scale_pos_weight` for tweaks.
  - Challengers: Replace with LightGBM or XGBoost for comparison.
- **Hardware Notes**: Trains in <1 min on CPU. For larger datasets, use CatBoost GPU.

### Segmenting Customers (Risk Tiers)
Post-training, segment via calibrated PDs:
```python
# Example (add after Cell 7)
def segment_customer(pd, t_low=0.05, t_high=0.20):
    if pd <= t_low:
        return "Low Risk"  # Auto-approve
    elif pd <= t_high:
        return "Sloppy Payer"  # Manual review
    else:
        return "High Risk"  # Decline

# Batch example
pd_test_seg = np.vectorize(segment_customer)(pd_test_cal)
print(pd.Series(pd_test_seg).value_counts())
```
- **Policy**:
  - **Low**: Approve at standard rates/terms.
  - **Sloppy**: Review/caps/adders.
  - **High**: Decline/alternatives.
- **EL Computation**: Uses `EL = PD * LGD * EAD`; customize LGD/EAD sources.

### Evaluation
- **Quantitative**:
  - **AUC**: Discrimination (~0.82 test).
  - **Expected Cost**: Misclassification (77 test; FN-heavy).
  - **PSI**: Stability (<0.1 = stable).
  - **Brier**: Calibration (via plot; low decomposition error).
- **Qualitative**:
  - Tiers show monotonic default rates (2.9% → 21.3% → 49.3%).
  - Fairness: Low disparities; monitor thresholds.
- **Metrics in Notebook**: Prints/tables for AUC, cost, PSI, fairness, EL.

## Model Architecture

CatBoostClassifier for binary PD prediction:

- **Params**:
  - `loss_function="Logloss"`, `eval_metric="AUC"`.
  - `depth=6`, `learning_rate=0.05`, `l2_leaf_reg=10`.
  - `iterations=2000`, `od_wait=200` (early stopping).
  - `class_weights=[1.0, ~2.33]` (imbalance: 700/300).
  - Categorical handling: Native via indices.

- **Data Flow**:
  1. **Input**: Raw features (700 train samples).
  2. **Boosting**: Ordered trees with categorical splits.
  3. **Output**: Raw PDs → Calibrated → Thresholded tiers.
  - Total Trees: ~500-1000 (early stop).
  - Explainability: Add `shap_values = cbc.get_feature_importance(type='ShapValues')`.

## Results

- **Performance** (Test Set):
  - AUC: 0.8195
  - Threshold: 0.1632
  - Cost: 77 (FP=52, FN=5; ~39% acceptance)
  - PSI (Train→Test): 0.0888 (stable)
- **Segmentation** (Test, n=150):
  - Low Risk (n=34): Avg PD=4.6%, Avg EAD=2118, Sum EL=2143
  - Sloppy Payer (n=47): Avg PD=16.0%, Avg EAD=2311, Sum EL=10702
  - High Risk (n=69): Avg PD=45.5%, Avg EAD=4302, Sum EL=84192
  - Portfolio EL: ~97K (LGD=0.6)
- **Fairness** (Sex Proxy):
  | Group | n   | AUC    | TPR    | FPR    | Selection Rate |
  |-------|-----|--------|--------|--------|----------------|
  | Female| 54 | 0.8180| 0.8421| 0.5429| 0.6481        |
  | Male  | 96 | 0.8176| 0.9231| 0.4714| 0.5938        |
  - ΔTPR: 0.081, ΔFPR: 0.071, ΔSelection: 0.054

    <img width="458" height="470" alt="image" src="https://github.com/user-attachments/assets/b6091308-4b19-4243-a703-59fab78019b6" />
    <img width="540" height="393" alt="image" src="https://github.com/user-attachments/assets/ca97a9df-6c2b-421e-bb9b-6d02071d3bf6" />


## Limitations and Improvements

### Limitations
- **Small Dataset**: 1K samples; may overfit—use cross-validation for larger data.
- **Anonymized Features**: Limits real-world interpretability; map to business terms.
- **Static Assumptions**: LGD=0.6 fixed; EAD=Attribute5 proxy—integrate true values.
- **Binary Target**: No granularity (e.g., days-past-due); extend to multi-class.

---
