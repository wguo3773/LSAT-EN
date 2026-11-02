# Tutorial: All-Feature Source-to-Target Modeling

## 1. Prepare Matching Predictors

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from lsaten import LSATBinaryCV, LSATCoxCV, survival_outcome

# Numeric CSVs: rows=patients, columns=the same features in the same order.
Xs = pd.read_csv("source_features.csv", index_col=0)
Xt = pd.read_csv("target_features.csv", index_col=0)
source_outcomes = pd.read_csv("source_outcomes.csv", index_col=0).loc[Xs.index]
target_outcomes = pd.read_csv("target_outcomes.csv", index_col=0).loc[Xt.index]
feature_info = pd.read_csv("feature_layers.csv").set_index("feature")
groups = feature_info.loc[Xs.columns, "layer"].to_numpy()
assert Xs.columns.equals(Xt.columns)
```

The feature metadata has one row per column, e.g. `feature,layer` with layer
values mRNA, miRNA, methylation, CNV, protein, mutation. The labels are
arbitrary: a single layer and other study settings are also supported.
Identifiers are matched explicitly by you; raw sequencing normalization,
categorical encoding and patient/sample deduplication are upstream tasks.
No raw TCGA patient data are included or fetched by the package.

**All columns enter fitting. There is no screening, ranking cutoff or
feature-budget search.** Imputation and standardization are learned from
training data only. For very large datasets, provision sufficient memory
for dense matrices and tune a feasible candidate grid.

## 2. Binary Outcome and Held-Out Target Test

```python
ys = source_outcomes["survival_4yr"].to_numpy()  # 1=alive through 4 years
yt = target_outcomes["survival_4yr"].to_numpy()
train, test = train_test_split(
    range(len(yt)), test_size=0.30, stratify=yt, random_state=42,
)
model = LSATBinaryCV(
    lambdas=(0.01, 0.05, 0.1), alphas=(0.25, 0.5, 0.75),
    tau_grid=(0.0, 0.1, 1.0), cv=5,
)
model.fit(Xs, ys, Xt.iloc[train], yt[train], groups=groups)
probability = model.predict_proba(Xt.iloc[test])[:, 1]
```

Never call `fit()` on the target-test patients. CV uses the provided training
subset only. The selected source anchor is refitted in the final training
coordinate system. Fold-specific preprocessing statistics are inspectable in
`fold_preprocessing_`. For an external test cohort, preserve feature order
and call prediction without refitting the model or preprocessing.

## 3. Cox Outcome and Time Units

```python
ys = survival_outcome(source_outcomes["time"], source_outcomes["death"])
yt = survival_outcome(target_outcomes["time"], target_outcomes["death"])
train, test = train_test_split(
    range(len(yt)), test_size=0.30, stratify=yt["event"], random_state=42,
)
model = LSATCoxCV(cv=5, lambdas=(0.01, 0.1), tau_grid=(0.0, 0.1, 1.0))
model.fit(Xs, ys, Xt.iloc[train], yt[train], groups=groups)
risk = model.predict(Xt.iloc[test])
```

Evaluate times in the same units as the input. Use `(2,3,4,5)` for years,
or corresponding day values for days. Harrell C-index uses higher score as
higher hazard, unlike positive binary survival coefficients.

```python
from lsaten import survival_metrics
metrics = survival_metrics(yt[train], yt[test], risk, times=(2.,3.,4.,5.), on_error="nan")
print(metrics["c_index"])
print(metrics["ipcw_auc"])
print(metrics["errors"])  # always inspect missing/unsupported evaluations
```

## 4. Layer Weights and Coefficients

```python
print(model.best_params_)
coefficients = model.feature_table()
coefficients.to_csv("feature_coefficients.csv", index=False)
nonzero = coefficients.loc[coefficients["nonzero"]]
print(nonzero.sort_values("standardized_coefficient"))
```

Large tau pulls target slopes toward the source anchor. Zero tau removes
that penalty for the layer but not the source's role in shared preprocessing.
For a wholly target-only baseline, use `strategy="target"`.
Zero coefficients are selected by the elastic net, not an earlier screening step.

## 5. Refit Stability and Comparisons

Run repeated matched outer splits; keep every original predictor column in
every refit. Store each fitted model, then call `stable_features(models)`.
Use identical outer splits for all methods, tuning each independently on
training data. See `examples/repeated_target_splits.py`.

`priority_overlap()` compares top lists. It does not identify a feature as
uniquely prognostic, establish an interaction, or prove another method missed
its biology. Predicted miRNA-target pairs need an external target database
and biological validation; coefficient signs alone do not demonstrate regulation.

## 6. Reproduce the Packaged Example

```bash
python examples/quickstart.py
```

The script generates only synthetic records, runs all-feature fits, prints
metrics, and saves coefficients, CV results, selected parameters and survival
probabilities to `docs/example-output/`. The README numeric tables come from
this script, not from redrawing or reading values off manuscript figures.

