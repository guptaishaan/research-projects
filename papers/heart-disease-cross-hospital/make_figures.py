"""Regenerate the vector figures used by the cross-hospital generalization paper.

Reads the exported result tables in ../../outputs/results/ and writes PDF figures
into ./figures/.  No re-training is performed: the tables are the study record.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "outputs" / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent / "_common"))
import figstyle  # noqa: E402

plt = figstyle.use()


SITES = ["Cleveland", "Hungary", "Switzerland", "VA"]
MODELS = ["LogReg", "RandomForest", "MLP"]
PRETTY = {"LogReg": "Logistic regression", "RandomForest": "Random forest", "MLP": "MLP"}

within = pd.read_csv(RESULTS / "within_cleveland.csv")
pairs = pd.read_csv(RESULTS / "pairwise_auc.csv")
gap = pd.read_csv(RESULTS / "generalization_gap.csv")


def matrix(model):
    m = np.full((4, 4), np.nan)
    for _, r in pairs.iterrows():
        m[SITES.index(r.train_site), SITES.index(r.test_site)] = r[model]
    return m


# ---------------------------------------------------------------- Figure 1
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
for ax, model in zip(axes, MODELS):
    M = matrix(model)
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.55, vmax=0.90)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(SITES, rotation=35, ha="right")
    ax.set_yticklabels(SITES if model == "LogReg" else [""] * 4)
    ax.set_xlabel("target hospital")
    if model == "LogReg":
        ax.set_ylabel("source hospital")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if i == j else "normal")
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=1.8))
    ax.set_title(PRETTY[model])
    for s in ax.spines.values():
        s.set_visible(False)
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="ROC-AUC")
fig.savefig(FIG / "fig1_transfer_matrix.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fig, ax = plt.subplots(figsize=(5.4, 3.2))
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w / 2, gap.same_hospital_AUC, w, label="internal (same hospital)",
            color="#2c6fbb", edgecolor="k", lw=0.4)
b2 = ax.bar(x + w / 2, gap.new_hospital_AUC, w, label="external (new hospital)",
            color="#e0a13a", edgecolor="k", lw=0.4)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .008,
                f"{b.get_height():.3f}", ha="center", fontsize=7)
for xi, r in zip(x, gap.itertuples()):
    ax.annotate(rf"$\Delta$={r.gap:+.3f}",
                (xi, max(r.same_hospital_AUC, r.new_hospital_AUC) + .06),
                ha="center", fontsize=8, fontweight="bold",
                color="#b22222" if r.gap > 0 else "#177245")
ax.axhline(0.5, ls=":", c="grey", lw=1)
ax.text(2.35, 0.515, "chance", color="grey", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in MODELS])
ax.set_ylim(0, 1.0); ax.set_ylabel("macro-average ROC-AUC")
ax.legend(loc="lower right", frameon=False)
fig.savefig(FIG / "fig2_generalization_gap.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
off = pairs[pairs.regime == "new hospital"]
fig, ax = plt.subplots(figsize=(6.8, 3.4))
labels = [f"{r.train_site[:4]}$\\rightarrow${r.test_site[:4]}" for _, r in off.iterrows()]
x = np.arange(len(off)); w = 0.26
for k, (model, c) in enumerate(zip(MODELS, ["#2c6fbb", "#3a7d44", "#c0392b"])):
    ax.bar(x + (k - 1) * w, off[model], w, label=PRETTY[model], color=c,
           edgecolor="k", lw=0.3)
ax.axhline(0.5, ls=":", color="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("target-site ROC-AUC"); ax.set_ylim(0.4, 0.95)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
fig.savefig(FIG / "fig3_directed_transfers.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
fig, ax = plt.subplots(figsize=(5.6, 3.0))
x = np.arange(3); w = 0.26
for k, (col, lab, c) in enumerate([("accuracy", "accuracy", "#9ecae1"),
                                   ("ROC_AUC", "ROC-AUC", "#2c6fbb"),
                                   ("F1", "$F_1$", "#123f66")]):
    ax.bar(x + (k - 1) * w, within[col], w, label=lab, color=c, edgecolor="k", lw=0.3)
    for xi, v in zip(x + (k - 1) * w, within[col]):
        ax.text(xi, v + .015, f"{v:.3f}", ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in within.model])
ax.set_ylim(0, 1.12); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
ax.set_ylabel("score")
ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center",
          bbox_to_anchor=(0.5, 1.15), columnspacing=1.6)
fig.savefig(FIG / "fig4_internal_cleveland.pdf")
plt.close(fig)

print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
