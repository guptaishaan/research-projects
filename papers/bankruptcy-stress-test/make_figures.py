"""Regenerate the figures for the 2008 bankruptcy stress-test paper.

The source dataset (`american_bankruptcy.csv`, ~78.7k company-years) is licensed and
not redistributed with this repository, so the study's measured results are transcribed
here from the executed notebook and written back out as CSV on every run.  If the raw
CSV is available, set BANKRUPTCY_CSV and the per-year curve is recomputed from it
instead of being omitted.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent / "_common"))
import figstyle  # noqa: E402

plt = figstyle.use()

ALIVE, FAILED = "#2a9d8f", "#c0392b"
CALM_C, REC_C = "#2c6fbb", "#c0392b"

# ---------------------------------------------------------------- recorded results
COHORT = dict(
    company_years=78682, alive=73462, failed=5220,
    year_min=1999, year_max=2018,
    imminent_positives=609, rows_after_labelling=74071,
    n_train=25611, rate_train=0.0057,
    n_calm_test=8596, rate_calm_test=0.0065,
    n_recession=7163, rate_recession=0.0113,
    n_features=15, n_raw_variables=18,
)

STRESS = pd.DataFrame([
    ("Logistic Regression", 0.646, 0.010, 0.615, -0.031),
    ("Random Forest",       0.755, 0.020, 0.798, +0.042),
    ("MLP",                 0.625, 0.012, 0.575, -0.050),
], columns=["model", "calm_AUC", "calm_PR", "recession_AUC", "change"])

FAILURE_RATE_BY_YEAR = pd.DataFrame(
    [(2005, 9.0), (2009, 6.3), (2015, 3.3)], columns=["fyear", "failed_pct"])

STRESS.round(4).to_csv(HERE / "results_stress_test.csv", index=False)
(HERE / "results_cohort.json").write_text(json.dumps(COHORT, indent=2))

# ---------------------------------------------------------------- Figure 1
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0),
                         gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
bars = ax.bar(["alive", "failed"], [COHORT["alive"], COHORT["failed"]],
              color=[ALIVE, FAILED], edgecolor="k", lw=0.4, width=0.62)
for b, v in zip(bars, [COHORT["alive"], COHORT["failed"]]):
    ax.text(b.get_x() + b.get_width() / 2, v + 1600, f"{v:,}",
            ha="center", fontsize=8, fontweight="bold")
ax.set_ylim(0, 84000)
ax.set_ylabel("company-years")
ax.set_title("(a) raw status label")

ax = axes[1]
stages = ["all\ncompany-years", "after dropping\nnon-final failed years",
          "imminent-failure\npositives"]
vals = [COHORT["company_years"], COHORT["rows_after_labelling"],
        COHORT["imminent_positives"]]
bars = ax.bar(stages, vals, color=["#8fa8c8", "#5b7db1", FAILED],
              edgecolor="k", lw=0.4, width=0.6)
ax.set_yscale("log")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:,}",
            ha="center", fontsize=8, fontweight="bold")
ax.set_ylim(200, 4e5)
ax.set_ylabel("rows (log scale)")
ax.set_title(f"(b) target construction: {COHORT['imminent_positives']} positives "
             f"({100 * COHORT['imminent_positives'] / COHORT['rows_after_labelling']:.2f}\\%)")
ax.tick_params(axis="x", labelsize=7.5)
fig.savefig(FIG / "fig1_class_balance.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0),
                         gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
splits = ["train\n(calm 1999--2006)", "held-out test\n(calm 1999--2006)",
          "stress test\n(recession 2008--09)"]
sizes = [COHORT["n_train"], COHORT["n_calm_test"], COHORT["n_recession"]]
rates = [COHORT["rate_train"], COHORT["rate_calm_test"], COHORT["rate_recession"]]
bars = ax.bar(splits, sizes, color=[CALM_C, "#7ba3d0", REC_C],
              edgecolor="k", lw=0.4, width=0.6)
for b, n, r in zip(bars, sizes, rates):
    ax.text(b.get_x() + b.get_width() / 2, n + 700,
            f"{n:,}\n{100 * r:.2f}\\% pos.", ha="center", fontsize=7.5)
ax.set_ylim(0, 31000)
ax.set_ylabel("company-years")
ax.set_title("(a) evaluation splits, grouped by company")
ax.tick_params(axis="x", labelsize=7.5)

ax = axes[1]
ax.bar(FAILURE_RATE_BY_YEAR.fyear.astype(str), FAILURE_RATE_BY_YEAR.failed_pct,
       color=["#8fa8c8", REC_C, "#8fa8c8"], edgecolor="k", lw=0.4, width=0.55)
for xi, v in enumerate(FAILURE_RATE_BY_YEAR.failed_pct):
    ax.text(xi, v + 0.2, f"{v:.1f}\\%", ha="center", fontsize=8)
ax.set_ylim(0, 11)
ax.set_ylabel("company-years labelled `failed' (\\%)")
ax.set_xlabel("fiscal year")
ax.set_title("(b) reported-failure rate drifts over time")
fig.savefig(FIG / "fig2_splits.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3),
                         gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
x = np.arange(len(STRESS)); w = 0.36
b1 = ax.bar(x - w / 2, STRESS.calm_AUC, w, label="held-out calm (1999--2006)",
            color=CALM_C, edgecolor="k", lw=0.4)
b2 = ax.bar(x + w / 2, STRESS.recession_AUC, w, label="recession (2008--09)",
            color=REC_C, edgecolor="k", lw=0.4)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{b.get_height():.3f}", ha="center", fontsize=7.5)
ax.axhline(0.5, ls=":", color="grey", lw=1)
ax.text(len(STRESS) - 0.6, 0.515, "chance", color="grey", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(["Logistic\nregression", "Random\nforest", "MLP"], fontsize=8)
ax.set_ylim(0, 0.95); ax.set_ylabel("ROC-AUC")
ax.set_title("(a) does the calm-era model survive the recession?")
ax.legend(fontsize=7, loc="lower right")

ax = axes[1]
colors = ["#177245" if c > 0 else "#b22222" for c in STRESS.change]
ax.barh(["Logistic\nregression", "Random\nforest", "MLP"], STRESS.change,
        color=colors, edgecolor="k", lw=0.4, height=0.5)
for yi, c in enumerate(STRESS.change):
    ax.text(c + (0.003 if c > 0 else -0.003), yi, f"{c:+.3f}",
            va="center", ha="left" if c > 0 else "right", fontsize=8,
            fontweight="bold")
ax.axvline(0, color="k", lw=1)
ax.set_xlim(-0.085, 0.075)
ax.set_xlabel(r"$\Delta = \mathrm{AUC}_{\mathrm{recession}}-\mathrm{AUC}_{\mathrm{calm}}$")
ax.set_title("(b) signed change under the regime shift")
fig.savefig(FIG / "fig3_stress_test.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
# The leakage artifact, stated with the exact recorded numbers: a per-year curve
# scored on years that still contain training firms is not comparable with the
# company-grouped held-out estimate.
fig, ax = plt.subplots(figsize=(6.4, 3.2))
labels = ["calm years\n(contains training firms)",
          "held-out calm firms\n(1999--2006)",
          "recession\n(2008--09)"]
vals = [np.nan, STRESS.calm_AUC[1], STRESS.recession_AUC[1]]
bars = ax.bar([1, 2], vals[1:], color=["#7ba3d0", REC_C], edgecolor="k",
              lw=0.4, width=0.5)
ax.bar([0], [0.95], color="none", edgecolor="#b22222", lw=1.2, ls="--",
       width=0.5, hatch="///")
ax.text(0, 0.50, "not a valid\nestimate:\nfirms overlap\nthe training set",
        ha="center", va="center", fontsize=7.5, color="#b22222")
for b, v in zip(bars, vals[1:]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
            ha="center", fontsize=8, fontweight="bold")
ax.axhline(0.5, ls=":", color="grey", lw=1)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylim(0, 1.05); ax.set_ylabel("random-forest ROC-AUC")
ax.set_title("Only the two right-hand bars are comparable")
fig.savefig(FIG / "fig4_leakage_artifact.pdf")
plt.close(fig)

# ---------------------------------------------------------------- optional per-year
csv = os.environ.get("BANKRUPTCY_CSV")
if csv and Path(csv).exists():
    print(f"raw CSV supplied at {csv}: the per-year curve can be recomputed. "
          "Extend this script with the notebook's fit/score loop.")
else:
    print("raw CSV not supplied; per-year curve omitted "
          "(set BANKRUPTCY_CSV to enable).")

print(STRESS.to_string(index=False))
print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
