"""Regenerate figures and result tables for the CLIP-forgetting forecasting paper.

Reads the cached 48-run dynamics campaign in
../../clip_forgetting_forecasting/outputs/dynamics/ and re-runs the
leave-one-dataset-out forecasting evaluation with the project's own helper module.
Adds one control the original notebook flagged as missing: a persistence baseline
that simply predicts final forgetting from forgetting at the end of the warm-up.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1] / "clip_forgetting_forecasting"
DYN = PROJ / "outputs" / "dynamics"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent / "_common"))
import figstyle  # noqa: E402

plt = figstyle.use()
sys.path.insert(0, str(PROJ))
import forecasting  # noqa: E402


FAMILY_COLOR = {"natural": "#2ca02c", "digits": "#d62728", "medical": "#1f77b4"}

curves = pd.read_csv(DYN / "dynamics_curves.csv")
features = pd.read_csv(DYN / "features.csv")
baseline = pd.read_csv(DYN / "baseline.csv", index_col=0).iloc[:, 0]
FAMILY = dict(zip(features.domain, features.family))
curves["family"] = curves.domain.map(FAMILY)
FINAL_EPOCH = curves.epoch.max()
final = curves[curves.epoch == FINAL_EPOCH]
lrs = sorted(curves.lr.unique())

# ---------------------------------------------------------------- Figure 1
fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9), sharey=True)
for ax, lr in zip(axes, lrs):
    for _, run in curves[curves.lr == lr].groupby("domain"):
        run = run.sort_values("epoch")
        ax.plot(run.epoch, run.forgetting, color=FAMILY_COLOR[run.family.iloc[0]],
                alpha=0.8, lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(rf"$\eta = {lr:g}$")
    ax.set_xlabel("fine-tuning epoch")
axes[0].set_ylabel(r"forgetting $\Phi_t$")
axes[-1].legend([Line2D([0], [0], color=c, lw=2) for c in FAMILY_COLOR.values()],
                list(FAMILY_COLOR), title="family", frameon=False, fontsize=7,
                title_fontsize=7, loc="upper left")
fig.savefig(FIG / "fig1_forgetting_curves.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
strongest = final[final.lr == max(lrs)]
ranked = strongest.set_index("domain").forgetting.sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.1),
                         gridspec_kw={"width_ratios": [2.3, 1]})
axes[0].bar(ranked.index, ranked.values,
            color=[FAMILY_COLOR[FAMILY[d]] for d in ranked.index],
            edgecolor="k", lw=0.3)
axes[0].set_ylabel(r"final forgetting $\Phi_{20}$")
axes[0].tick_params(axis="x", rotation=90, labelsize=7)
axes[0].set_title(r"(a) per dataset at $\eta=10^{-5}$")
groups = [strongest[strongest.family == f].forgetting.values for f in FAMILY_COLOR]
bp = axes[1].boxplot(groups, tick_labels=list(FAMILY_COLOR), patch_artist=True,
                     widths=0.55, medianprops=dict(color="k"))
for patch, colour in zip(bp["boxes"], FAMILY_COLOR.values()):
    patch.set_facecolor(colour); patch.set_alpha(0.55)
axes[1].set_title("(b) by family")
axes[1].set_ylabel(r"final forgetting $\Phi_{20}$")
fig.savefig(FIG / "fig2_final_forgetting.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
static_cols = ["semantic_dist", "clip_embed_dist", "spectral_dist", "frechet_dist"]
merged = strongest.merge(features, on="domain")
corr = merged[static_cols + ["forgetting"]].corr()["forgetting"].drop("forgetting")
fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.9),
                         gridspec_kw={"width_ratios": [1.15, 1]})
axes[0].barh(static_cols, corr.values,
             color=["#2c7fb8" if v > 0 else "#d95f0e" for v in corr.values],
             edgecolor="k", lw=0.3)
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_xlabel(r"Pearson $r$ with final forgetting")
axes[0].set_title("(a) static descriptors, 16 datasets")
run = curves[(curves.domain == "SVHN") & (curves.lr == max(lrs))].sort_values("epoch")
for col, lab, c in [("forgetting", r"$\Phi_t$ (forgetting)", "#d62728"),
                    ("train_acc", "train accuracy", "#2ca02c"),
                    ("param_drift", r"$\delta^{\mathrm{par}}_t$", "#1f77b4"),
                    ("emb_drift", r"$\delta^{\mathrm{emb}}_t$", "#9467bd")]:
    v = run[col].values
    axes[1].plot(run.epoch, (v - v.min()) / (v.max() - v.min() + 1e-12),
                 marker="o", ms=2.5, lw=1.2, color=c, label=lab)
axes[1].axvspan(1, 5, color="grey", alpha=0.15)
axes[1].text(3, 0.06, "warm-up", ha="center", fontsize=7, color="0.3")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("min--max rescaled")
axes[1].set_title("(b) tracked signals, SVHN")
axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
fig.savefig(FIG / "fig3_features.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Evaluation
table = forecasting.build_table(curves, features, K=5).reset_index(drop=True)
results = {}
for label, cols in [("static only", forecasting.STATIC),
                    ("static + warm-up", forecasting.STATIC + forecasting.WARMUP)]:
    results[label] = forecasting.evaluate(table, cols)
rows = [dict(features=label, model=name, **s)
        for label, res in results.items() for name, s in res.items()]
scores = pd.DataFrame(rows)


def metrics(pred, true):
    return dict(MAE=float(np.mean(np.abs(pred - true))),
                Spearman=float(spearmanr(pred, true)[0]),
                R2=float(1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)))


# Persistence control: final forgetting == forgetting observed at the end of warm-up.
persist = metrics(table.f_warm.values, table.target.values)
# Linear extrapolation control: continue the warm-up slope to the final epoch.
extrap = metrics((table.f_warm + table.fslope_warm * (FINAL_EPOCH - 5)).values,
                 table.target.values)
scores = pd.concat([
    scores,
    pd.DataFrame([dict(features="control", model="Persistence", **persist),
                  dict(features="control", model="Slope extrapolation", **extrap)]),
], ignore_index=True)
scores.round(4).to_csv(HERE / "results_lodo.csv", index=False)

# ---------------------------------------------------------------- Figure 4
piv = (scores[scores.features != "control"]
       .pivot_table(index="model", columns="features", values="R2")
       [["static only", "static + warm-up"]])
order = ["Ridge (linear)", "Random Forest", "Grad Boosting", "MLP (neural net)"]
piv = piv.loc[order]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2),
                         gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(piv)); w = 0.36
axes[0].bar(x - w / 2, piv["static only"], w, label="static only",
            color="#bdbdbd", edgecolor="k", lw=0.3)
axes[0].bar(x + w / 2, piv["static + warm-up"], w, label="static + warm-up",
            color="#2c7fb8", edgecolor="k", lw=0.3)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].axhline(persist["R2"], ls="--", color="#d62728", lw=1)
axes[0].text(len(piv) - 0.55, persist["R2"] + 0.03, "persistence control",
             color="#d62728", fontsize=7, ha="right")
axes[0].set_xticks(x)
axes[0].set_xticklabels([m.split(" (")[0] for m in piv.index], rotation=12)
axes[0].set_ylabel(r"leave-one-dataset-out $R^2$")
axes[0].set_title("(a) does the warm-up add information?")
axes[0].legend(frameon=False, loc="lower left")

best_cols = forecasting.STATIC + forecasting.WARMUP
pred = forecasting.lodo(table, best_cols, forecasting.models()["Grad Boosting"])
actual = table.target.values
fam = table.domain.map(FAMILY).values
for family, colour in FAMILY_COLOR.items():
    sel = fam == family
    axes[1].scatter(actual[sel], pred[sel], color=colour, s=26, alpha=0.85,
                    edgecolor="k", lw=0.3, label=family)
lims = [min(actual.min(), pred.min()) - .02, max(actual.max(), pred.max()) + .02]
axes[1].plot(lims, lims, "k--", lw=1)
axes[1].set_xlabel("actual final forgetting")
axes[1].set_ylabel("forecast (dataset held out)")
axes[1].set_title("(b) gradient boosting, LODO")
axes[1].legend(frameon=False, fontsize=7, title="family", title_fontsize=7)
fig.savefig(FIG / "fig4_forecast_quality.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 5
early = forecasting.warmup_curve(curves, features)
persist_by_k, extrap_by_k = [], []
for K in early.K:
    t = forecasting.build_table(curves, features, K=int(K))
    persist_by_k.append(metrics(t.f_warm.values, t.target.values)["R2"])
    extrap_by_k.append(metrics((t.f_warm + t.fslope_warm * (FINAL_EPOCH - K)).values,
                               t.target.values)["R2"])
fig, ax = plt.subplots(figsize=(5.4, 3.1))
ax.plot(early.K, early.held_out_R2, "o-", color="#2c7fb8", ms=3.5,
        label="gradient boosting (static + warm-up)")
ax.plot(early.K, persist_by_k, "s--", color="#d62728", ms=3,
        label=r"persistence: $\hat\Phi_{20}=\Phi_K$")
ax.plot(early.K, extrap_by_k, "^:", color="#7f7f7f", ms=3,
        label="slope extrapolation")
ax.axhline(0, color="k", lw=0.6)
ax.set_ylim(-0.6, 1.05)
ax.set_xlabel("warm-up length $K$ (epochs observed)")
ax.set_ylabel(r"leave-one-dataset-out $R^2$")
ax.legend(frameon=False, loc="lower right", fontsize=7)
ax.grid(alpha=0.25)
fig.savefig(FIG / "fig5_warmup_length.pdf")
plt.close(fig)

early["persistence_R2"] = persist_by_k
early["extrapolation_R2"] = extrap_by_k
early.round(4).to_csv(HERE / "results_warmup_curve.csv", index=False)

summary = {
    "baseline_zero_shot": {k: float(v) for k, v in baseline.items()},
    "n_runs": int(len(table)),
    "n_datasets": int(table.domain.nunique()),
    "final_epoch": int(FINAL_EPOCH),
    "family_final_forgetting": strongest.groupby("family").forgetting
                                        .agg(["mean", "min", "max"]).round(3).to_dict(),
    "static_corr": corr.round(3).to_dict(),
    "persistence": {k: round(v, 4) for k, v in persist.items()},
    "extrapolation": {k: round(v, 4) for k, v in extrap.items()},
}
(HERE / "results_summary.json").write_text(json.dumps(summary, indent=2))
print(scores.round(3).to_string(index=False))
print(json.dumps(summary["family_final_forgetting"], indent=1))
print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
