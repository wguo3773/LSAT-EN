from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "simulation-data.csv"
OUT = ROOT / "assets"

PANELS = [
    ("A", "0", "sigma0"),
    ("B", "0.25", "sigma0p25"),
    ("C", "1", "sigma1"),
    ("D", "4", "sigma4"),
]
METHODS = [
    ("glmtrans EN", "#D85B78"),
    ("Overall EN", "#73777A"),
    ("Individual EN", "#819E00"),
    ("GAS-EN", "#009B7A"),
    ("Group-tau ReCAST-MAP-EN", "#9B55B7"),
]
TARGET_SIZES = [200, 140, 100, 80]

rows = []
with SOURCE.open(newline="") as handle:
    all_rows = list(csv.DictReader(handle))
for letter, sigma, slug in PANELS:
    panel_rows = [r for r in all_rows if r['panel'] == letter]
    for n in TARGET_SIZES:
        for method, _ in METHODS:
            matches = [r for r in panel_rows if int(r["nT"]) == n and r["method"] == method]
            if len(matches) != 1:
                raise ValueError(f"Expected one result for sigma={sigma}, nT={n}, {method}")
            r = matches[0]
            rows.append({
                "panel": letter,
                "sigmaTL2": sigma,
                "nT": n,
                "method": method,
                "auc_pct": float(r["auc_pct"]),
                "se_pct": float(r["se_pct"]),
            })

assert len(rows) == 80

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})
fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
fig.subplots_adjust(left=0.095, right=0.985, top=0.91, bottom=0.15, wspace=0.13, hspace=0.21)

group_y = np.arange(len(TARGET_SIZES))
bar_h = 0.15
offsets = np.linspace(-2, 2, len(METHODS)) * bar_h
for ax, (letter, sigma, _) in zip(axes.flat, PANELS):
    panel = [r for r in rows if r["panel"] == letter]
    for offset, (method, color) in zip(offsets, METHODS):
        data = [next(r for r in panel if r["nT"] == n and r["method"] == method) for n in TARGET_SIZES]
        ax.barh(
            group_y + offset,
            [r["auc_pct"] - 50 for r in data],
            left=50,
            height=bar_h * 0.9,
            color=color,
            edgecolor="none",
            xerr=[r["se_pct"] for r in data],
            error_kw={"ecolor": "#202020", "elinewidth": 0.8, "capsize": 1.5, "capthick": 0.8},
        )
    ax.set_yticks(group_y, [str(n) for n in TARGET_SIZES])
    ax.set_xlim(50, 100.5)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.grid(axis="x", color="#E4E6E8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=5)
    ax.set_title(rf"{letter}    $\sigma_{{TL}}^2 = {sigma}$", loc="left", fontweight="bold", fontsize=13, pad=10)

axes[0, 0].invert_yaxis()
fig.suptitle("Binary prediction performance across source-target heterogeneity", x=0.095, y=0.98, ha="left", fontweight="bold", fontsize=17)
fig.text(0.095, 0.945, "Mean test AUC over 20 simulations; error bars indicate SE", fontsize=10.5, color="#4B5563")
fig.supxlabel("AUC (%)", y=0.105, fontsize=11)
fig.supylabel("Target-training sample size", x=0.028, fontsize=11)
handles = [Patch(facecolor=color, label='LSAT-EN' if method == 'Group-tau ReCAST-MAP-EN' else method) for method, color in METHODS]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.025), ncol=5, frameon=False, fontsize=9.5, handlelength=1.2, columnspacing=1.4)

for suffix in ("png", "svg"):
    path = OUT / f'manuscript-binary-simulation.{suffix}'
    fig.savefig(path, dpi=300, facecolor="white")
plt.close(fig)
print(OUT / "manuscript-binary-simulation.png")
