"""Run from the repository root after installation. Synthetic data only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lsaten import (LSATBinaryCV, LSATCoxCV, make_binary_transfer,
                    make_survival_transfer, binary_metrics, survival_metrics)


def main():
    out = Path("docs/example-output")
    out.mkdir(parents=True, exist_ok=True)
    layer_names = ["mRNA", "miRNA", "methylation", "CNV", "protein", "mutation"]
    groups = np.repeat(layer_names, 6)
    data = make_binary_transfer(n_source=180, n_target=90, n_test=150,
                                n_features=36, block_size=6, sigma2_tl=.25, random_state=42)
    rows = []
    for strategy in ("lsat", "pooled", "target"):
        model = LSATBinaryCV(lambdas=(.02, .1), alphas=(.5,), tau_grid=(0., .5, 2.),
                             cv=3, coordinate_passes=1, strategy=strategy)
        model.fit(data["X_source"], data["y_source"], data["X_target"], data["y_target"], groups=groups)
        metrics = binary_metrics(data["y_test"], model.predict_proba(data["X_test"])[:, 1])
        rows.append(dict(method=strategy, **metrics, n_input=model.n_features_in_,
                         n_nonzero=model.n_nonzero_, best_validation_loss=model.best_validation_loss_))
        if strategy == "lsat":
            model.feature_table().to_csv(out / "binary-features.csv", index=False)
            model.cv_results_.to_json(out / "binary-cv-results.json", orient="records", indent=2)
            (out / "binary-best-parameters.json").write_text(json.dumps(model.best_params_, indent=2))
    binary = pd.DataFrame(rows)
    binary.to_csv(out / "binary-metrics.csv", index=False)
    print("BINARY EXAMPLE (all 36 predictors)")
    print(binary.to_string(index=False))

    data = make_survival_transfer(n_source=140, n_target=90, n_test=100,
                                  n_features=36, block_size=6, random_state=43)
    rows = []
    for strategy in ("lsat", "pooled", "target"):
        model = LSATCoxCV(lambdas=(.05, .15), alphas=(.5,), tau_grid=(0., .5),
                          cv=3, coordinate_passes=1, strategy=strategy)
        model.fit(data["X_source"], data["y_source"], data["X_target"], data["y_target"], groups=groups)
        metrics = survival_metrics(data["y_target"], data["y_test"], model.predict(data["X_test"]),
                                    times=(2., 3., 4., 5.), on_error="nan")
        rows.append(dict(method=strategy, c_index=metrics["c_index"],
                         **{f"auc_{int(t)}yr": v for t, v in metrics["ipcw_auc"].items()},
                         n_input=model.n_features_in_, n_nonzero=model.n_nonzero_,
                         metric_errors=json.dumps(metrics["errors"])))
        if strategy == "lsat":
            model.feature_table().to_csv(out / "cox-features.csv", index=False)
            (out / "cox-best-parameters.json").write_text(json.dumps(model.best_params_, indent=2))
            survival = model.predict_survival_function(data["X_test"][:5], [2., 3., 4., 5.])
            pd.DataFrame(survival, columns=["2yr", "3yr", "4yr", "5yr"]).to_csv(
                out / "cox-survival-probabilities.csv", index=False)
    cox = pd.DataFrame(rows)
    cox.to_csv(out / "cox-metrics.csv", index=False)
    print("\nCOX EXAMPLE (all 36 predictors; times in years)")
    print(cox.to_string(index=False))


if __name__ == "__main__":
    main()

