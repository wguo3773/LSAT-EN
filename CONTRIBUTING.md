# Contributing

Install locally with `python -m pip install -e ".[dev,survival]"` and run
`python -m pytest -q`. Submit changes with focused regression tests and clear
descriptions of objective normalization, censoring assumptions and input shapes.
Use synthetic data in public issues and examples; do not upload patient records.

The current package intentionally fits all supplied predictors. Do not add an
implicit screening, feature-budget selection or held-out-test tuning step.
Changes to the objective, source-anchor construction or tau search must be
documented and tested. Report numerical convergence and unsupported metrics
explicitly rather than silently substituting results.

