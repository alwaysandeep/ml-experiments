# Plan: Diagnose & Improve Logistic Regression for Kaggle S6E2

## Context
- **Competition:** Playground Series S6E2 — Heart Disease binary classification (AUC metric)
- **Current state:** Simple logistic regression, Public LB AUC = **0.95090**, local val AUC = **0.9537**
- **Goal:** Identify WHERE the model fails, then improve via feature engineering — staying within logistic regression only
- **Gap to close:** Top LB is 0.95405 → need +0.00315 on public LB

## Notebook
`C:\Users\schitta\Downloads\ml-experiments\experiments\kaggle\playground-feb-26-edition.ipynb`

## Data
- `experiments/kaggle/input/competitions/playground-series-s6e2/train.csv` (630K rows)
- `experiments/kaggle/input/competitions/playground-series-s6e2/test.csv` (270K rows)
- 8 categorical features, 5 continuous features, no missing values, balanced target (55/45)

---

## PART 1: Diagnostics (7 cells to add)

### 1.1 Per-Feature AUC Analysis
Train a single-feature LR for each of the 13 features. Rank by val AUC.
- **Insight:** Identifies which features carry most signal individually
- **Action:** Top 3-5 features become priority targets for interactions in Part 2
- **Good:** Clear separation (some >0.80, some <0.60) | **Bad:** All features clustered near same AUC

### 1.2 Feature Coefficient Analysis
Extract and visualize LR coefficients from the trained pipeline (after OneHotEncoding).
- **Insight:** Which encoded features have strongest positive/negative weights
- **Action:** Near-zero coefficients = candidates for removal; large coefficients = interaction targets
- Plot top 15 by absolute value, color by sign (red=risk, blue=protective)

### 1.3 Calibration Curve
Plot predicted probability vs actual fraction of positives (10-20 bins).
- **Insight:** Are predicted probabilities reliable? LR is usually well-calibrated but check extremes
- **Action:** If poorly calibrated → add `CalibratedClassifierCV` wrapper
- Include Brier score and log loss metrics

### 1.4 Error Analysis — FP vs FN Patterns
At threshold=0.5, categorize val predictions into Correct/FP/FN. Compare feature distributions across groups.
- **Insight:** What feature patterns do misclassified samples share?
- **Action:** High-error subgroups reveal where to add interactions or bins
- Print error rates, boxplots for continuous features, crosstabs for categorical features
- Flag any subgroup with >15% error rate

### 1.5 Subgroup ROC Analysis
Compute separate ROC curves for subgroups: Sex (0 vs 1), Chest pain type (1-4), Exercise angina (0 vs 1), Age bins (<45, 45-54, 55-64, 65+).
- **Insight:** Does model perform equally across subgroups?
- **Action:** If AUC gap >0.05 between subgroups → add interaction between that feature and top predictors
- Plot overlaid ROC curves per subgroup feature

### 1.6 Residual Analysis
Plot residuals (actual - predicted prob) vs predicted probability. Histogram of predictions. Count samples in "uncertain zone" (prob 0.4-0.6).
- **Insight:** Are errors systematic or random? How many borderline cases exist?
- **Action:** Systematic patterns → need polynomial/interaction terms; many uncertain samples → model lacks discriminating features for those cases
- Compare feature distributions of uncertain vs confident samples

### 1.7 Confusion Matrix at Multiple Thresholds
Show confusion matrices at thresholds 0.3, 0.4, 0.5, 0.6, 0.7. Plot F1 vs threshold to find optimal.
- **Insight:** Note — this is informational since Kaggle uses AUC (threshold-independent), but useful for understanding the decision boundary
- **Action:** Mainly for understanding, not for submission improvement

---

## PART 2: Feature Engineering (6 cells to add)

Each step builds on the previous — features accumulate.

### 2.1 Interaction Features (highest priority)
Create interactions between top features from Diagnostic 1.1. Starting candidates:
- `ST depression * Exercise angina` (continuous * binary)
- `Thallium + Number of vessels fluro` (concatenated categorical)
- `Age * ST depression` (continuous * continuous)
- `Chest pain type + Exercise angina` (concatenated categorical)
- `Max HR * Age` (continuous * continuous)

Evaluate: compare val AUC to baseline 0.9537.

### 2.2 Polynomial & Ratio Features
- Squared terms: Age, BP, Cholesterol, Max HR, ST depression
- Log transforms: log1p for all positive continuous features
- Domain ratios: `Max HR / Age` (age-adjusted heart rate), `BP / Age`

Evaluate: compare val AUC to step 2.1.

### 2.3 Binning Continuous Features (clinical thresholds)
Keep original continuous features AND add bins as additional categoricals:
- Age: <40, 40-49, 50-59, 60-69, 70+
- BP: Normal(<120), Elevated(120-130), Stage1(130-140), Stage2+(>140)
- Cholesterol: Desirable(<200), Borderline(200-240), High(>240)
- Max HR: quartile bins
- ST depression: None(0), Mild(0-1), Moderate(1-2), Severe(>2)

Evaluate: compare val AUC to step 2.2.

### 2.4 Regularization Tuning
GridSearchCV over:
- `C`: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
- `penalty`: ['l1', 'l2'] with solver='saga'
- Scoring: roc_auc, 5-fold stratified CV

Evaluate: report best params and CV AUC.

### 2.5 5-Fold Cross-Validation
Run best configuration with StratifiedKFold(5) to get robust AUC estimate with std.
- If std < 0.005 → stable model
- Compare mean CV AUC to single-split val AUC

### 2.6 Final Comparison & Submission
- Table comparing all model variants (baseline, +interactions, +poly, +bins, +regularization)
- Generate test predictions with best model
- Save as `improved_logistic_submission.csv`

---

## Verification
1. Run all Part 1 diagnostic cells — visually inspect plots, read printed insights
2. Run Part 2 cells sequentially — confirm each step's val AUC >= previous step
3. Final model's val AUC should be > 0.9537 (baseline)
4. Submit `improved_logistic_submission.csv` to Kaggle — target Public LB > 0.9509
5. Expected combined improvement: +0.003 to +0.010 AUC over baseline
