# Mathematical Specification

## All-Feature Input

All supplied predictors enter fitting. There is **no SIS, univariate feature
filtering, top-K selection, or feature-budget CV step**. Supply the same feature
columns in the same order for source, target training, and prediction data.
Features are assigned to prespecified layers through `groups`.

Missing values are filled with training medians. A feature missing throughout
the available training data receives zero. Predictors are centered and scaled
with training statistics; a zero-variance scale is set to one. No predictor
column is removed. A constant or entirely missing column contains no estimable
outcome information and will ordinarily have a zero coefficient under a
positive elastic-net penalty.

## Source Anchor and Target Objective

The source-only model uses elastic-net settings `source_lam` and
`source_alpha`. These are explicit fixed inputs, defaulting to 0.05 and 0.5;
they are not silently selected using held-out target outcomes. Target CV
selects `lam`, `alpha`, and layer-specific `tau` values.

Both binary and Cox models minimize

$$
L_T(\beta) + \lambda\left[\alpha\|\beta\|_1
+\frac{1-\alpha}{2}\|\beta\|_2^2\right]
+\frac12\sum_g\tau_g\|\beta_g-\widehat\beta_{Sg}\|_2^2.
$$

Binary loss is mean logistic log loss. Its target intercept is unpenalized
and is not anchored to the source intercept. `predict_proba(X)[:,1]` predicts
the probability of label 1. For fixed-horizon survival, label 1 must mean
alive through the stated horizon. `binary_at_horizon()` excludes censoring
before that horizon instead of labeling those patients alive.

Cox loss is negative Breslow partial log-likelihood divided by the number
of target events. Tied event times use the complete risk set at that time.
There is no Cox intercept parameter. Higher `predict()` scores mean higher
event risk. The Breslow baseline hazard is estimated from target-training
outcomes, even when using the pooled-coefficient baseline. Survival estimates
are flat beyond the last training event, not validated extrapolation.

Lambda, alpha, and tau therefore depend on the documented likelihood
normalization. They need not equal regularization parameters from another
library. The implementation uses SciPy's L-BFGS-B optimizer with beta=u-v,
u,v>=0, and L1 term lambda*alpha*sum(u+v). This is an exact nonnegative
variable representation, not a smooth approximation to the L1 penalty.
Convergence is exposed rather than silently assumed.

## Cross-Validation Without Validation Leakage

For each inner target fold:

1. Keep the validation fold completely outside imputation and scaling.
2. Fit preprocessing using all source and the remaining target-training folds
   (target-only strategy uses only those target-training folds).
3. Fit a source-only anchor in that fold's standardized coordinate system.
4. Hold this anchor fixed across target-parameter candidates for the fold.
5. Fit the target objective and evaluate on the held-out target fold.

Binary CV minimizes mean fold log loss; Cox CV minimizes mean fold
event-normalized negative partial likelihood. Folds are stratified by class
or event status. The package requires enough observations in both strata
for the requested CV count; it does not silently reuse training observations
as validation or invent evaluable folds. The chosen settings are refitted
using all provided target-training data. The user must hold the outer target
test set aside and supply identical outer splits to all comparison methods.

The default layer-weight search is coordinate search (two passes), not a
guaranteed global search of the Cartesian grid. Set `tau_search="grid"` for
an exhaustive Cartesian search when feasible. `max_candidates` prevents an
accidental combinatorial run. `cv_results_` retains tried candidates, fold
losses, and convergence; nonconverged candidates cannot win the search.

## Baselines and Biomarkers

`strategy="pooled"` fits EN on source plus target-training outcomes, with no
anchoring. `strategy="target"` fits EN on target-training data only; its
preprocessing also ignores source data. Both strategies can tune their own
lambda and alpha on identical target folds. No subset is selected to favor
LSAT-EN. Group labels do not alter either unanchored baseline's penalty.

`stable_features()` ranks features by absolute median nonzero standardized
coefficient multiplied by selection frequency. Its threshold and sign rule
are explicit. This is descriptive repeated-fit stability, not statistical
significance, mechanistic proof, or external biomarker validation. Source and
target preprocessing can vary between refits; report this when comparing
standardized coefficient magnitudes. `priority_overlap()` compares top stable
lists, and absence from a list does not imply a zero model coefficient.

## Survival Metrics

Optional evaluation uses scikit-survival's Harrell C-index and
`cumulative_dynamic_auc()`. Censoring is estimated from target-training
outcomes. Evaluation times must be supported by observed follow-up and
censoring support. Errors raise by default; `on_error="nan"` retains missing
values and reasons, never clips unsupported weights or substitutes zero AUC.
The six omics layers in the manuscript are an application: the API supports
any supplied layer labels and does not require age or omics data specifically.

