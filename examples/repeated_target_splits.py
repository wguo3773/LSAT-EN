"""Matched repeated outer splits; each method tunes itself on training data."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lsaten import LSATBinaryCV, make_binary_transfer, binary_metrics


def run(n_repeats=10):
    data = make_binary_transfer(n_source=180, n_target=140, n_test=50,
                                n_features=30, block_size=10, random_state=2026)
    results = []
    for repeat in range(n_repeats):
        train, test = train_test_split(np.arange(len(data["y_target"])),
                                      test_size=.30, stratify=data["y_target"], random_state=repeat)
        for strategy in ("lsat", "pooled", "target"):
            model = LSATBinaryCV(lambdas=(.02, .1), tau_grid=(0., .5), cv=3,
                                 coordinate_passes=1, strategy=strategy, random_state=repeat)
            model.fit(data["X_source"], data["y_source"], data["X_target"][train],
                      data["y_target"][train], groups=data["groups"])
            score = binary_metrics(data["y_target"][test],
                                   model.predict_proba(data["X_target"][test])[:, 1])
            results.append({"repeat": repeat, "method": strategy, **score})
    return pd.DataFrame(results)


if __name__ == "__main__":
    print(run().groupby("method").auc.agg(["mean", "std", "count"]))

