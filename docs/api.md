# API Reference

## `LSATBinary` / `LSATCox`: Fixed Parameters

| Argument | Default | Meaning |
|---|---|---|
| `lam` | 0.05 | Overall elastic-net regularization strength |
| `alpha` | 0.5 | L1 fraction in [0,1]; zero is ridge |
| `tau` | 1.0 | Nonnegative scalar or layer-to-weight dictionary |
| `source_lam`, `source_alpha` | 0.05, 0.5 | Source-only anchor regularization |
| `strategy` | `"lsat"` | `"lsat"`, `"pooled"`, or `"target"` |
| `max_iter`, `tol` | 2000, 1e-6 | SciPy optimization controls |

All features are used. Neither class accepts screening/budget settings.

## `LSATBinaryCV` / `LSATCoxCV`: Select Target Parameters

These use the same `source_lam`, `source_alpha`, `strategy`, `max_iter` and
`tol`, plus:

| Argument | Default | Meaning |
|---|---|---|
| `lambdas` | (0.01, 0.1) | Nonnegative candidate regularization strengths |
| `alphas` | (0.5,) | Candidate L1 fractions |
| `tau_grid` | (0, 0.1, 1) | Shared candidate values, or dictionary of grids per layer |
| `cv` | 5 | Target folds, stratified by class/event status |
| `tau_search` | `"coordinate"` | Coordinate search or full Cartesian `"grid"` |
| `coordinate_passes` | 2 | Layer-weight coordinate-search passes |
| `max_candidates` | 10000 | Cap on tried parameter configurations |
| `random_state` | 42 | Inner fold reproducibility |

### Fit Inputs

```python
model.fit(X_source, y_source, X_target_train, y_target_train,
          groups=layer_per_feature, feature_names=optional_names)
```

X matrices have shape (patients, predictors), with identical column order.
Use numeric arrays or matching pandas DataFrames. Encode categorical data
outside the package using a training-only encoder. NaNs are permitted;
infinity is not. Supply one nonempty `groups` label per feature. For binary,
labels are 0/1 with both present; for Cox, use `survival_outcome(time,event)`.
Do not pass held-out target test data to `fit()`.

### Methods and Fitted Outputs

| Name | Output |
|---|---|
| `predict_proba(X)` (binary) | n x 2 array: probabilities of 0 and 1 |
| `predict(X)` (binary) | 0/1 at probability threshold 0.5 |
| `predict(X)` (Cox) | Relative log hazard; higher is higher event risk |
| `predict_survival_function(X,times)` (Cox) | n x len(times) survival probabilities |
| `feature_table()` | All features, layers, nonzero flags, raw and standardized coefficients, source anchors and tau |
| `coef_`, `coef_standardized_` | Full-length coefficient vectors in input and standardized units |
| `n_features_in_`, `n_nonzero_` | Supplied feature count and number above absolute 1e-6 |
| `best_params_` | Selected or supplied lambda, alpha, tau |
| `best_validation_loss_` (CV) | Mean validation loss of selected converged configuration |
| `cv_results_` (CV) | Tried settings, fold losses, mean loss and convergence |
| `cv_splits_`, `fold_preprocessing_` (CV) | Audit of fold indices and preprocessing statistics |
| `converged_` | Both source and target optimizers converged, where applicable |
| `optimization_result_`, `source_optimization_result_` | Native SciPy optimizer diagnostics |

`source_anchor` in feature tables is in standardized units. Large tau pulls
coefficients toward that anchor. Zero tau removes anchoring for that layer,
but the source may still influence shared preprocessing. To fit an entirely
source-independent baseline, use `strategy="target"`.

## Data, Evaluation, and Biomarkers

| Function | Key inputs | Output |
|---|---|---|
| `make_binary_transfer()` | Sizes, feature/block counts, correlations, `sigma2_tl`, seed | Synthetic source, target-training and independent test arrays, true coefficients and groups |
| `make_survival_transfer()` | Same plus event/censoring rates | Illustrative proportional-hazards data, times in years |
| `survival_outcome(time,event)` | Positive follow-up; event 0/1 | Structured array with fields `event`, `time` |
| `binary_at_horizon(y,horizon)` | Censored outcome, horizon | Known-outcome mask and binary survival labels |
| `binary_metrics(y,probability)` | Labels and P(y=1) | AUC and log loss |
| `survival_metrics(y_train,y_test,risk,times,on_error)` | Target-training censoring data; test risk | C-index, IPCW AUC dictionary, error reasons |
| `stable_features(models,...)` | Repeated fits with identical input feature identities | Frequencies, sign consistency, median coefficients, priorities and stability flags |
| `priority_overlap(tables,top_n=50)` | Named stable-feature tables | Boolean membership matrix for priority-set comparison |

Use `sigma2_tl` as a **variance**, not a standard deviation. Simulation
replicates are independent generated datasets; repeated train/test splits of
one cohort are not independent simulation datasets. This package provides
building blocks and examples, not a claim to regenerate manuscript tables.

