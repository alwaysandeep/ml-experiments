# Part 1 Diagnostics — Deep Reference Guide

## Context
Picking up PLAN-diagnose-improve-lr.md to implement Part 1 diagnostics for the Kaggle S6E2 heart disease logistic regression model (AUC=0.9537). This document captures the deep understanding of each diagnostic approach before implementation.

**Notebook:** `experiments/kaggle/playground-feb-26-edition.ipynb`

---

## 1.1 Per-Feature AUC Analysis

### What It Actually Measures
Train a single-feature LR for each of 13 features, compute validation AUC per feature.

When you train LR on one feature and compute AUC, you're answering: **"If I pick a random patient WITH heart disease and a random patient WITHOUT, what's the probability that this single feature alone ranks the sick patient higher?"**

This is mathematically equivalent to the **Mann-Whitney U statistic**. AUC = 0.75 literally means 75% of the time, a random positive scores higher than a random negative on that feature.

### How It Differs from Other Measures
- **Correlation (Pearson):** Only captures linear relationships, scale-dependent. A perfect monotonic but non-linear relationship could have low correlation but high AUC.
- **Mutual information:** Captures any statistical dependence (including non-monotonic) — more general than AUC but requires binning/tuning.
- **Single-feature AUC:** Captures monotonic ranking separation. Doesn't care about exact shape, just whether higher values generally mean higher disease probability.

### Why AUC Over Accuracy
1. **Threshold-independent** — no arbitrary cutoff needed
2. **Class imbalance robust** — doesn't reward predicting majority class
3. **Directly measures ranking quality** — matches Kaggle's scoring metric
4. **Clean comparability** — categorical and continuous features on the same 0.5–1.0 scale

### Interpretation Guide
| Individual AUC | Interpretation |
|---|---|
| 0.50 | Random — zero discriminative power |
| 0.55–0.65 | Weak signal |
| 0.65–0.75 | Moderate — useful but not standalone |
| 0.75–0.85 | Strong individual predictor |
| 0.85+ | Very strong — rare for single feature |

For this dataset (full-model AUC 0.9537, 13 features), expect best single feature around 0.80–0.85.

### Your Actual Results (from notebook)
| Feature | Val AUC |
|---|---|
| Thallium | 0.80 |
| Chest pain type | 0.77 |
| Max HR | 0.76 |
| ST depression | 0.74 |
| Number of vessels fluro | 0.72 |
| Slope of ST | 0.72 |
| Exercise angina | 0.70 |
| Sex | 0.66 |
| Age | 0.63 |
| EKG results | 0.61 |
| Cholesterol | 0.55 |
| FBS over 120 | 0.51 |
| BP | 0.50 |

### Limitations
1. **No redundancy awareness** — correlated features both rank high but adding both may give nothing over one
2. **Interaction-only features invisible** — XOR-like patterns show AUC ≈ 0.50 for both features individually
3. **Categorical features with rare levels** can overfit (mitigated by 630K rows)
4. **Individual power ≠ marginal contribution** — high AUC feature might add nothing if a correlated feature is already in the model

### Feature Engineering Guidance
Common heuristic: "create interactions between top-ranked features." **This is folklore, not proven.** With only 13 features (78 pairs), test all pairwise interactions rather than pre-selecting. Use rankings for intuition, not hard pruning.

---

## 1.2 Feature Coefficient Analysis

### What LR Coefficients Mean
```
log(p / (1-p)) = B0 + B1*x1 + B2*x2 + ... + Bn*xn
```
Each coefficient Bi means: **a 1-unit increase in xi changes the log-odds of heart disease by Bi, holding all else constant.** e^Bi is the odds ratio.

- **Positive coefficient** → increases risk
- **Negative coefficient** → protective
- **Near-zero** → contributes little

### Why It's Useful
1. **Multivariate** — unlike per-feature AUC, coefficients reflect contribution within the full model
2. **Identifies dead weight** — near-zero coefficients = candidates for removal or transformation
3. **Identifies dominant features** — largest |coefficients| = highest-leverage targets for feature engineering
4. **Has direction** — risk factor vs protective factor

### Critical Caveat: Scale Dependence
Raw coefficients are NOT directly comparable unless features are on the same scale. Age (20-80) vs ST_depression (0-6) will have different coefficient magnitudes for the same predictive power. **Solution:** standardize features first (your pipeline uses StandardScaler for continuous features). One-hot encoded categoricals are naturally on 0/1 scale.

### Your Actual Results (from notebook)
Top coefficients (already standardized for continuous):
- Chest pain type 4: +2.24 (strongest risk)
- Thallium 7: +1.95
- Number of vessels fluro 3: +1.90
- Number of vessels fluro 2: +1.84
- Thallium 6: +1.43
- Exercise angina 1: +1.19
- Max HR: -0.83 (strongest protective)
- Near-zero: BP (0.00), FBS over 120 (-0.03), EKG results 1 (-0.06)

### Limitation: U-Shaped Relationships
If a feature has a U-shaped relationship with disease (both extremes are risky), the linear coefficient averages the opposing effects and appears near-zero — **underestimating true importance**.

**Three fixes:**
1. **Polynomial (squared) terms** — add x^2 so model can fit U-shape. Pro: simple. Con: only captures quadratic nonlinearity.
2. **Bin into categories** — each bin gets its own coefficient, no linearity constraint. Pro: captures any shape. Con: loses within-bin information, requires choosing boundaries.
3. **Keep both original + bins** (best practice) — continuous captures linear trend, bins capture nonlinear deviations.

### How 1.1 and 1.2 Complement Each Other
| Per-Feature AUC (1.1) | Coefficients (1.2) |
|---|---|
| Univariate | Multivariate |
| Any monotonic relationship | Only linear contribution |
| No redundancy awareness | Splits weight among correlated features |
| Scale-independent | Scale-dependent (unless standardized) |

A feature with high AUC but near-zero coefficient = another correlated feature already captures its signal. Moderate AUC but large coefficient = provides unique information.

---

## 1.3 Calibration Curve

### What Calibration Means
Model is well-calibrated when predicted probabilities match reality. If model says 70% chance for 100 patients, ~70 should actually have disease.

**Different from discrimination (AUC).** Perfect AUC + terrible calibration is possible (perfect ranking but wrong probabilities).

### How It Works
1. Sort predictions into 10-20 bins
2. For each bin: X = average predicted probability, Y = actual fraction of positives
3. Plot. Perfectly calibrated = diagonal line (y=x)

### Deviation Patterns
| Pattern | Name | Meaning |
|---|---|---|
| Above diagonal | Under-confident | Says 40%, reality is 60% |
| Below diagonal | Over-confident | Says 80%, reality is 60% |
| S-shaped | Sigmoid miscalibration | Mixed |
| Flat in middle | Poor resolution | Predictions cluster around 0.5 |

### LR Is Usually Well-Calibrated
LR optimizes log-loss, a proper scoring rule — loss is minimized when predicted probabilities match true probabilities. **Breaks down when:** model is misspecified (nonlinear truth + linear model), feature distribution mismatch, heavy regularization.

### Brier Score vs Log Loss
Both measure calibration + discrimination.

| Predicted p (true label=1) | Brier = (1-p)^2 | Log Loss = -log(p) |
|---|---|---|
| 0.9 | 0.01 | 0.105 |
| 0.5 | 0.25 | 0.693 |
| 0.1 | 0.81 | 2.303 |
| 0.01 | 0.98 | **4.605** |
| 0.001 | 0.998 | **6.908** |

**Log loss penalizes confident wrong predictions MORE harshly** — unbounded (goes to infinity). Brier is bounded (max 1.0). This is why LR (which optimizes log loss) tends to be well-calibrated: infinite penalty for confident wrong answers forces hedging when uncertain.

### For Your Case
With 630K rows, balanced classes, no extreme regularization — **expect good calibration**. This is a sanity check, not a discovery tool. More important for poorly-calibrated models (random forests, naive Bayes).

---

## 1.4 Error Analysis — FP vs FN Patterns

### What It Is
At threshold 0.5, classify validation patients into TP/TN/FP/FN. Then examine: **what do FP and FN patients look like? Do they share feature patterns?**

### Why It's Valuable
Errors are NOT random. The model fails on specific subpopulations whose feature combinations confuse the linear boundary. Identifying those patterns tells you exactly what features or transformations to add.

### Your Actual Results
- **11.3% error rate** — 14,239 errors out of 126K validation
- **6,393 FPs, 7,846 FNs** — roughly balanced
- **Max HR is the most telling:** FNs have MaxHR=156 (looks healthy), TPs have 141, TNs have 162
- **ST depression:** FNs have 0.48 (low), TPs have 1.28 (high) — FNs are sick patients with low ST depression
- **BP and Cholesterol:** virtually identical across all 4 groups — useless for distinguishing errors
- **No category exceeds 15% error rate** — but Males (13%) > Females (8%), Chest pain type 4 (13%), Thallium 3 has 8% FN rate

### Key Insight
**FNs are sick patients with "healthy-looking" profiles** — high Max HR, low ST depression, normal Thallium. To catch them, need interactions that capture "even though Max HR is high, if Thallium=X and ChestPainType=Y, still risky."

**However:** You already tried all pairwise interactions (cells 52-53) and only gained +0.0001 AUC. Linear interactions aren't enough — may need nonlinear transformations (bins, polynomials).

### FP/FN Rate Computation Note
The notebook computes rates per category: FP_rate = FP_count_in_category / total_in_category. This is NOT the classical FPR (FP / actual_negatives). Both valid — notebook version answers "if I'm a male patient, what's my chance of being misclassified?"

### Limitations
1. **Threshold-dependent** (0.5 is arbitrary for AUC evaluation)
2. **Correlation ≠ causation** in error patterns
3. **Descriptive, not prescriptive** — shows WHERE model fails, not automatically HOW to fix

---

## 1.5 Subgroup ROC Analysis

### What It Is
Compute separate ROC curves and AUC for subgroups: Sex (0 vs 1), Chest pain type (1-4), Exercise angina (0 vs 1), Age bins (<45, 45-54, 55-64, 65+).

### How It Differs from Error Analysis (1.4)
Error analysis gives threshold-dependent error rates. Subgroup ROC gives **threshold-independent discrimination** per subgroup. Since Kaggle scores on AUC, this is more relevant.

A subgroup could have low error rate at 0.5 but terrible AUC (lucky threshold), or high error rate but excellent AUC (wrong threshold for that group).

### AUC Gap Interpretation
| Gap | Meaning |
|---|---|
| < 0.02 | Uniform performance — no subgroup problem |
| 0.02–0.05 | Minor discrepancy |
| > 0.05 | Significant underperformance — feature representation insufficient for this population |

### Fix for AUC Gaps
Add interaction terms between the subgroup feature and top predictors (e.g., Sex x ST depression). This lets the model learn different slopes for different groups.

### Limitations
1. Subgroups are pre-selected by you — may miss unexpected subgroups
2. Overlapping subgroups — hard to disentangle which feature causes the gap
3. AUC needs both classes — near-pure subgroups have unreliable AUC

---

## 1.6 Residual Analysis

### What It Is
Residual = actual outcome (0 or 1) - predicted probability. Plot residuals to find systematic patterns.

### Three Plots
1. **Residuals vs predicted probability** — look for systematic deviations from the expected two-stripe pattern
2. **Histogram of predictions** — should be bimodal (peaks near 0 and 1) for good model
3. **Uncertain zone count (0.4-0.6)** — fewer = better

### Key Difference from Error Analysis
Error analysis is binary (right/wrong). Residual analysis is continuous — distinguishes "barely wrong" from "catastrophically wrong."

### The Real Power: Residuals vs Features
Plot residuals against each feature:
- **Residuals curve up/down with a feature** → nonlinear relationship (needs polynomial/bins)
- **Residuals large for specific feature range** → model can't handle that region
- **No pattern** → errors are random, model is well-specified

This directly detects U-shaped relationships — residuals positive at both extremes of a feature.

### Limitations
1. Binary residuals are inherently discrete (two stripes) — use smoothing or binning
2. 126K points = scatter plots become blobs — use hexbin or sampled subsets
3. Descriptive, not prescriptive

---

## 1.7 Confusion Matrix at Multiple Thresholds

### What It Is
Confusion matrices at thresholds 0.3, 0.4, 0.5, 0.6, 0.7. Plot F1 vs threshold.

### Why It's Least Important for Your Goal
Kaggle uses AUC (threshold-independent). Optimizing for a specific threshold doesn't change your score. This is purely informational.

### What It Still Teaches
1. **Decision boundary sharpness** — flat F1 across thresholds = fuzzy boundary, sharp peak = clean
2. **Error symmetry** — optimal threshold near 0.5 = balanced model (expected with 55/45 split)
3. **Sensitivity analysis** — how many sick patients caught at different operating points

### Limitations
1. F1 assumes equal cost of FP/FN (not realistic in medicine)
2. Not actionable for AUC improvement
3. With balanced classes and good calibration, likely uninteresting

---

## Actionability Summary

| Diagnostic | Actionability for AUC | What It Drives |
|---|---|---|
| 1.1 Per-Feature AUC | **High** | Identifies strongest features for interactions |
| 1.2 Coefficients | **High** | Shows model's actual feature usage, dead weight |
| 1.3 Calibration | **Low** | Sanity check only |
| 1.4 Error Analysis | **High** | Reveals failure patterns, specific subgroups |
| 1.5 Subgroup ROC | **Medium** | Identifies underperforming populations |
| 1.6 Residual Analysis | **High** | Reveals model misspecification, nonlinearity |
| 1.7 Multi-Threshold CM | **Low** | Informational only |

---

## Key Finding So Far
Pairwise linear interactions (all 78 pairs tested) only improved AUC by +0.0001. This suggests the model needs **nonlinear transformations** (polynomials, bins) rather than just multiplicative interactions between features.
