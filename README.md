# MOF Synthesis Property Prediction — NU-1000 / NU-901

Machine learning workflow predicting **particle size** and **crystalline phase (PXRD ratio)** of
Zr-based metal-organic frameworks from their synthesis conditions.

Built as part of the **Aixelo × DataLab Applied Data Science program** (team of 4, May–July 2026).
This repository contains the data-analysis and modeling half of the project; my role was data
cleaning, feature engineering, model selection, and validation.

---

## Problem

MOF synthesis is highly sensitive to reaction conditions — modulator loading, co-modulator
volume, temperature, and dilution all shift the resulting crystal morphology. Controlling
particle size is required for device integration, but the relationship between recipe and
outcome is non-linear and largely established by trial and error in the lab.

**Goal:** predict, from the synthesis recipe alone, (a) the resulting particle size in nm and
(b) which MOF phase forms (NU-1000 vs NU-901, indicated by a PXRD reflection ratio) — so that
target-driven recipe recommendation becomes possible.

---

## Data

Two experimental datasets from the partner lab, one per modulator system:

| Dataset | Modulator | Raw runs | After dropping failed syntheses |
|---|---|---|---|
| **BPh** | biphenyl-4-carboxylic acid (→ NU-1000) | 83 | 83 |
| **BA** | benzoic acid (→ NU-901) | 67 | 63 |
| **Merged** | both, with modulator dummy | 150 | **146** |

**Inputs (6 raw):** linker mass, modulator mass, co-modulator (TFA) volume, linker/node
solution volumes, reaction temperature, addition temperature.

**Targets:** `particle_size_nm` (18–5200 nm range), `pxrd_ratio` (phase indicator),
`particle_size_stdv_pct` (dispersity — dropped after Run 3 as unpredictable).

Class balance is uneven: the merged set splits into Small 52 / Medium 44 / Large 50 by size
regime, and Mixed 84 / NU-1000 36 / NU-901 26 by phase.

> **Note:** the raw CSVs are lab-generated data belonging to the project partner and are not
> included in this repository. The notebooks document every preprocessing step so the pipeline
> is auditable without them.

---

## Approach

### 1. Preprocessing — chemistry-aware normalization

Raw masses and volumes are not comparable across the two protocols, so everything is converted
to **molar equivalents relative to the Zr₆-oxo cluster** (69.0 mg, MW 1388.4 g/mol = 1.00 eq.):

- modulator and linker mass → eq. via molecular weight
- TFA volume (µL) → eq. via density and MW, then `log1p`
- addition temperature → binary `addition_hot` flag (≥ 100 °C)
- explicit `TFA_was_added` flag to distinguish *measured zero* from *missing*
- `IsolationForest` (contamination 5%) to flag, not remove, candidate outliers

This step is what makes the two datasets mergeable at all.

### 2. Feature engineering — five progressive tiers

Features were added in batches and evaluated against the previous tier, rather than generated
in bulk:

| Tier | Added | Rationale |
|---|---|---|
| Run 1 (9) | molar equivalents, `tfa_ratio`, `modulation_effect` | baseline chemistry encoding |
| Run 2 (12) | `linker_conc`, `mod_x_T`, `TFA_present` | concentration and temperature interactions |
| Run 4 (16) | `mod_to_linker`, `TFA_to_linker`, `total_cap_to_linker`, `supersaturation` | stoichiometric competition for Zr₆ coordination sites |
| Run 5 (22) | `acidity_score` (pKa-weighted), `TFA_acidity_boost`, `Zr6_conc_mM`, `node_linker_coverage` | nucleation theory and coordination chemistry |
| Run 8/10 (23) | `regime_score` | continuous input-derived proxy for size regime |

**Target engineering:** particle size is log-normally distributed over two orders of magnitude,
so models train on `log10(particle_size)` and `log1p(pxrd_ratio)`. All reported scores are
**back-transformed to original units** (nm, ratio) — log-space R² is systematically optimistic
because a 10 nm error at 100 nm and a 1000 nm error at 10 000 nm are the same log residual.

### 3. Feature selection

Per-dataset Spearman |r| thresholds (BPh 0.20, BA 0.40, Merged 0.40), later extended to a
**union rule** — keep a feature if Spearman |r| ≥ threshold **or** normalized mutual
information ≥ 0.10 — to rescue non-linear relationships that rank correlation misses.

### 4. Model selection

16 model families benchmarked per dataset × target: linear (OLS, Ridge, Lasso, ElasticNet,
Bayesian Ridge, Huber), kernel (Kernel Ridge with RBF and Laplacian kernels, SVR, Gaussian
Process with Matérn), neighbours (KNN), trees and ensembles (Random Forest, Extra Trees,
Gradient Boosting, AdaBoost, XGBoost, LightGBM), and MLP.

Hyperparameters tuned with `GridSearchCV` / `RandomizedSearchCV` (50 iterations) inside a
`Pipeline` (median imputation → `RobustScaler` → model).

**Evaluation is Leave-One-Out.** With 63–83 samples per dataset, 5-fold CV leaves folds of only
12–16 points and the variance of the estimate swamps the differences between models.

---

## Results

Best model per dataset × target, **R² in original units** from LOO predictions:

| Dataset | Target | Best model | R² (original) | MAE |
|---|---|---|---|---|
| BA | particle size (nm) | **Kernel Ridge (Laplacian)** | **0.674** | 333 nm |
| Merged | PXRD ratio | KNN | 0.632 | 0.44 |
| Merged | particle size (nm) | SVR | 0.612 | 360 nm |
| BA | PXRD ratio | ElasticNet | 0.556 | 0.30 |
| BPh | particle size (nm) | SVR | 0.507 | 405 nm |
| BPh | PXRD ratio | SVR | 0.299 | 0.54 |

**Kernel methods win consistently.** With ~100 samples and smooth-but-non-linear chemical
response surfaces, Kernel Ridge and SVR outperform both regularized linear models and tree
ensembles — trees fragment the space too aggressively at this sample size, and the Laplacian
kernel handles the sharp transitions around modulator-concentration thresholds better than RBF.

### Uncertainty

Point estimates on n = 63–146 are not stable, so R² is reported with a **paired percentile
bootstrap 95% CI** (2000 resamples over LOO predictions). For the best model
(BA, particle size): **R² = 0.674, 95% CI [0.350, 0.769]**.

The width of that interval is a finding in itself: at this sample size the performance estimate
carries substantial uncertainty, which argues for expanded and more evenly distributed data
collection before any of these models drives lab decisions. The interval is reported rather
than the point estimate alone for exactly this reason.

### Data leakage — found and corrected

The `regime_score` and `is_likely_large` features are built by searching threshold cut-points
that best separate the *size regime*, which is a binning of the actual target. In the original
runs this search — and the Spearman/MI feature selection — was fit on the **full** dataset
before LOO began, so each held-out point's own label influenced the features used to predict it.

`TopModels_DeepDive_LeakCorrected.ipynb` §17 implements a **fold-safe nested LOO** where
threshold search, feature selection, and imputation statistics are all refit on the training
fold only, and reports the leaky and corrected scores side by side.

---

## Repository structure

```
MOF_ML.ipynb                            Main workflow — preprocessing, EDA, Runs 1-10, post-hoc analysis
TopModels_DeepDive_LeakCorrected.ipynb  Reproduces the 6 winners, deep-dive diagnostics,
                                        fold-safe nested LOO, bootstrap CIs
```

**`MOF_ML.ipynb`** — the experimental log. Ten numbered runs, each one hypothesis about what
should improve the model, with the outcome recorded whether or not it worked (Run 6, the size-regime
feature, is kept in the notebook specifically because it *is* the leakage case that motivated
Runs 7–10).

**`TopModels_DeepDive_LeakCorrected.ipynb`** — takes only the winning configurations forward and
adds: actual-vs-predicted with ±20% bands, residual analysis, MAE broken down by size regime,
permutation importance, partial dependence plots, model internals, the leak-corrected nested LOO,
and bootstrap confidence intervals.

---

## Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `SciPy` · `XGBoost` · `LightGBM` · `SHAP`
· `matplotlib` · `seaborn` · `joblib`

---

## What this project taught me

Most of the work was not model selection — it was deciding what an honest number looks like on
a small experimental dataset. Log-space R² flattered the models; original-unit R² did not.
A point estimate looked decisive; the bootstrap interval showed it wasn't. A feature that
improved every score turned out to be leaking the target. Each of those was a step toward
reporting a result the lab could actually act on.
