"""Regenerate figures and result tables for the RadAlert-AT exposure-triage paper.

Re-runs the study's leave-one-study-out benchmark from the harmonized expression
matrix in ../../raddose-phytodosimetry/raddose_data/, so every number in the paper is
recomputed rather than transcribed.  Runs on CPU in well under a minute.
"""
import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             matthews_corrcoef, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "raddose-phytodosimetry" / "raddose_data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent / "_common"))
import figstyle  # noqa: E402

plt = figstyle.use()

SEED = 0
np.random.seed(SEED)
warnings.filterwarnings("ignore")
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#8A8A8A"}

# ---------------------------------------------------------------- data
X = pd.read_parquet(DATA / "X_rnaseq.parquet")
manifest = pd.read_csv(DATA / "manifest_rnaseq.csv")
assert len(X) == len(manifest) == 158 and X.shape[1] == 4002

DOSE = "absorbed_dose_gy"
y_dose = manifest[DOSE].to_numpy(float)
y = (y_dose > 0).astype(int)
groups = manifest["study"].to_numpy()
studies = sorted(manifest["study"].unique())
Xv = X.to_numpy(float)

DDR_PANEL = ["AT2G31320", "AT4G21070", "AT5G66130", "AT4G02390",
             "AT5G20850", "AT4G22960", "AT1G07500"]
ddr_idx = [X.columns.get_loc(g) for g in DDR_PANEL]


def logistic_top(k):
    return make_pipeline(SelectKBest(f_classif, k=k), StandardScaler(),
                         LogisticRegression(C=.1, class_weight="balanced",
                                            max_iter=5000, random_state=SEED))


def rbf_top(k):
    return make_pipeline(SelectKBest(f_classif, k=k), StandardScaler(),
                         SVC(C=1, kernel="rbf", class_weight="balanced",
                             probability=True, random_state=SEED))


def trees_top(k):
    return make_pipeline(SelectKBest(f_classif, k=k),
                         ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2,
                                              max_features="sqrt",
                                              class_weight="balanced", n_jobs=-1,
                                              random_state=SEED))


def ddr_model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=.1, class_weight="balanced",
                                            max_iter=5000, random_state=SEED))


def loso_probability(model, matrix=Xv):
    p = np.full(len(y), np.nan)
    for held in studies:
        test = groups == held
        p[test] = clone(model).fit(matrix[~test], y[~test]).predict_proba(matrix[test])[:, 1]
    assert np.isfinite(p).all()
    return p


predictions = {
    "Always exposed": np.full(len(y), 0.5),
    "DDR-7 logistic": loso_probability(ddr_model(), Xv[:, ddr_idx]),
    "Top-50 logistic": loso_probability(logistic_top(50)),
    "Top-100 RBF-SVM": loso_probability(rbf_top(100)),
    "Top-100 ExtraTrees": loso_probability(trees_top(100)),
}
tree_50 = loso_probability(trees_top(50))
predictions["Stability ensemble"] = .5 * (tree_50 + predictions["Top-100 ExtraTrees"])


def metric_row(name, p):
    pred = (p >= .5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "ROC_AUC": roc_auc_score(y, p),
        "macro_study_AUC": np.mean([roc_auc_score(y[groups == s], p[groups == s])
                                    for s in studies]),
        "average_precision": average_precision_score(y, p),
        "MCC": matthews_corrcoef(y, pred),
        "sensitivity": recall_score(y, pred),
        "specificity": recall_score(y, pred, pos_label=0),
    }


metrics = pd.DataFrame([metric_row(n, p) for n, p in predictions.items()])
metrics = metrics.sort_values(["balanced_accuracy", "ROC_AUC"], ascending=False)
metrics.round(4).to_csv(HERE / "results_models.csv", index=False)

best_p = predictions["Stability ensemble"]
best_y = (best_p >= .5).astype(int)

by_study = pd.DataFrame([{
    "study": s, "n": int((groups == s).sum()),
    "exposed_fraction": y[groups == s].mean(),
    "accuracy": accuracy_score(y[groups == s], best_y[groups == s]),
    "balanced_accuracy": balanced_accuracy_score(y[groups == s], best_y[groups == s]),
    "ROC_AUC": roc_auc_score(y[groups == s], best_p[groups == s]),
} for s in studies])
by_study.round(4).to_csv(HERE / "results_by_study.csv", index=False)

# ---------------------------------------------------------------- Figure 1
dose_counts = manifest[DOSE].value_counts().sort_index()
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0),
                         gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
bars = ax.bar(dose_counts.index.astype(int).astype(str), dose_counts.values,
              color=[C["green"] if d == 0 else C["orange"] for d in dose_counts.index],
              edgecolor="k", lw=0.4, width=0.62)
for b, n in zip(bars, dose_counts.values):
    ax.text(b.get_x() + b.get_width() / 2, n + 2, str(n), ha="center", fontsize=8)
ax.set_xlabel("absorbed dose (Gy)"); ax.set_ylabel("samples")
ax.set_ylim(0, 90)
ax.set_title("(a) the dose axis is bimodal, not a calibration curve")

ax = axes[1]
tab = (manifest.assign(exposed=y).groupby("study")
       .agg(control=("exposed", lambda s: int((s == 0).sum())),
            exposed=("exposed", "sum")))
x = np.arange(len(tab))
ax.bar(x, tab.control, 0.6, label="control (0 Gy)", color=C["green"],
       edgecolor="k", lw=0.4)
ax.bar(x, tab.exposed, 0.6, bottom=tab.control, label="exposed ($>0$ Gy)",
       color=C["orange"], edgecolor="k", lw=0.4)
ax.set_xticks(x); ax.set_xticklabels([s.replace("OSD-", "") for s in tab.index])
ax.set_xlabel("NASA OSDR study"); ax.set_ylabel("samples")
ax.set_title("(b) study composition is highly uneven")
ax.legend(fontsize=7, loc="upper left")
fig.savefig(FIG / "fig1_design_audit.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
plot_cols = ["accuracy", "balanced_accuracy", "ROC_AUC"]
pdf = metrics.set_index("model")[plot_cols].sort_values("balanced_accuracy")
fig, ax = plt.subplots(figsize=(6.8, 3.2))
yy = np.arange(len(pdf)); h = 0.26
for k, (col, lab, c) in enumerate([("accuracy", "accuracy", C["sky"]),
                                   ("balanced_accuracy", "balanced accuracy", C["green"]),
                                   ("ROC_AUC", "ROC-AUC", C["purple"])]):
    ax.barh(yy + (k - 1) * h, pdf[col], h, label=lab, color=c, edgecolor="k", lw=0.3)
ax.axvline(.5, color="k", ls=":", lw=1)
ax.text(.512, -0.62, "chance", fontsize=7, color="0.35")
ax.set_yticks(yy); ax.set_yticklabels(pdf.index, fontsize=8)
ax.set_xlim(0, 1.0); ax.set_xlabel("score")
ax.legend(fontsize=7, loc="lower right")
fig.savefig(FIG / "fig2_model_ladder.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2),
                         gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
cm = confusion_matrix(y, best_y)
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["control", "exposed"]); ax.set_yticklabels(["control", "exposed"])
ax.set_xlabel("predicted"); ax.set_ylabel("true")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_title("(a) pooled LOSO confusion matrix")
for s in ax.spines.values():
    s.set_visible(False)

ax = axes[1]
x = np.arange(len(by_study)); w = 0.26
ax.bar(x - w, by_study.accuracy, w, label="accuracy", color=C["sky"],
       edgecolor="k", lw=0.3)
ax.bar(x, by_study.balanced_accuracy, w, label="balanced accuracy", color=C["green"],
       edgecolor="k", lw=0.3)
ax.bar(x + w, by_study.ROC_AUC, w, label="ROC-AUC", color=C["purple"],
       edgecolor="k", lw=0.3)
ax.axhline(.5, color="k", ls=":", lw=1)
ax.set_xticks(x)
ax.set_xticklabels([s.replace("OSD-", "") for s in by_study.study])
ax.set_xlabel("held-out study"); ax.set_ylim(0, 1.08); ax.set_ylabel("score")
ax.set_title("(b) performance varies sharply across unseen studies")
ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.savefig(FIG / "fig3_error_analysis.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
by_dose = pd.DataFrame([{
    "dose_Gy": d, "n": len(idx),
    "fraction_called_exposed": best_y[np.asarray(list(idx))].mean(),
    "mean_exposure_score": best_p[np.asarray(list(idx))].mean(),
} for d, idx in manifest.groupby(DOSE).groups.items()])
by_dose.round(4).to_csv(HERE / "results_by_dose.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
ax = axes[0]
ax.plot(by_dose.dose_Gy, by_dose.mean_exposure_score, "o-", color=C["orange"], lw=1.6)
for r in by_dose.itertuples():
    ax.annotate(f"$n={r.n}$", (r.dose_Gy, r.mean_exposure_score), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=7)
ax.axhline(.5, color="k", ls=":", lw=1, label="decision threshold")
ax.set_xlabel("absorbed dose (Gy)"); ax.set_ylabel("mean held-out exposure score")
ax.set_ylim(0, 1.05)
ax.set_title("(a) exposure score by dose (descriptive)")
ax.legend(fontsize=7, loc="lower right")

ax = axes[1]
for s in studies:
    take = groups == s
    fpr, tpr, _ = roc_curve(y[take], best_p[take])
    ax.plot(fpr, tpr, lw=1.4,
            label=f"{s.replace('OSD-', 'OSD-')}: {roc_auc_score(y[take], best_p[take]):.2f}")
ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
ax.set_title("(b) ROC per completely held-out study")
ax.legend(fontsize=6.5, loc="lower right")
fig.savefig(FIG / "fig4_dose_and_roc.pdf")
plt.close(fig)

# ---------------------------------------------------------------- uncertainty
rng = np.random.default_rng(SEED)
B = 10000
boot = {k: [] for k in ["accuracy", "balanced_accuracy", "ROC_AUC"]}
for _ in range(B):
    draw = rng.choice(len(by_study), len(by_study), replace=True)
    for k in boot:
        boot[k].append(by_study[k].iloc[draw].mean())
uncertainty = pd.DataFrame([{
    "metric": k, "estimate": by_study[k].mean(),
    "ci_lo": np.quantile(boot[k], .025), "ci_hi": np.quantile(boot[k], .975),
} for k in boot])
uncertainty.round(4).to_csv(HERE / "results_uncertainty.csv", index=False)

# ---------------------------------------------------------------- gene stability
selection = pd.Series(0, index=X.columns, dtype=int)
for held in studies:
    train = groups != held
    sel = SelectKBest(f_classif, k=100).fit(Xv[train], y[train]).get_support(indices=True)
    selection.iloc[sel] += 1
stable = (selection.sort_values(ascending=False).rename("folds").head(20)
          .rename_axis("gene").reset_index())
ann = pd.read_csv(DATA / "tables" / "panel_annotation.csv")
stable = stable.merge(ann[["gene", "symbol", "description"]], on="gene", how="left")
stable.to_csv(HERE / "results_stable_genes.csv", index=False)
ddr_stable = sorted(set(selection[selection == 6].index).intersection(DDR_PANEL))

fig, ax = plt.subplots(figsize=(6.4, 3.6))
top = stable.head(15).sort_values(["folds", "gene"])
labels = [s if isinstance(s, str) and s else g for g, s in zip(top.gene, top.symbol)]
ax.barh(labels, top.folds, color=C["green"], edgecolor="k", lw=0.3, height=0.65)
ax.axvline(6, color="k", ls=":", lw=1)
ax.set_xlim(0, 6.5); ax.set_xlabel("training folds selecting the gene (of 6)")
ax.tick_params(axis="y", labelsize=7)
fig.savefig(FIG / "fig5_stable_genes.pdf")
plt.close(fig)

# ---------------------------------------------------------------- secondary regression
pred_dose = np.full(len(y_dose), np.nan)
for held in studies:
    test = groups == held
    m = HistGradientBoostingRegressor(random_state=SEED).fit(Xv[~test], y_dose[~test])
    pred_dose[test] = np.clip(m.predict(Xv[test]), 0, 100)
dose_mae = float(np.mean(np.abs(y_dose - pred_dose)))
dose_rho = float(spearmanr(y_dose, pred_dose).correlation)

fig, ax = plt.subplots(figsize=(4.4, 3.6))
sc = ax.scatter(y_dose, pred_dose, c=y, cmap="coolwarm", alpha=.7, s=26,
                edgecolor="k", lw=0.3)
ax.plot([0, 100], [0, 100], "k--", lw=1, label="perfect calibration")
ax.set_xlabel("true dose (Gy)"); ax.set_ylabel("predicted dose (Gy)")
ax.set_title(f"MAE $= {dose_mae:.1f}$ Gy, Spearman $\\rho = {dose_rho:.3f}$",
             fontsize=8.5)
ax.legend(fontsize=7, loc="upper left")
fig.savefig(FIG / "fig6_dose_regression.pdf")
plt.close(fig)

summary = {
    "n_samples": int(len(y)), "n_genes": int(X.shape[1]),
    "n_controls": int((y == 0).sum()), "n_exposed": int((y == 1).sum()),
    "headline": {k: round(float(v), 4) for k, v in
                 metrics[metrics.model == "Stability ensemble"].iloc[0].items()
                 if k != "model"},
    "confusion": cm.tolist(),
    "uncertainty": uncertainty.round(4).to_dict("records"),
    "ddr_genes_selected_every_fold": ddr_stable,
    "dose_regression": {"MAE_Gy": round(dose_mae, 2), "spearman": round(dose_rho, 4)},
}
(HERE / "results_summary.json").write_text(json.dumps(summary, indent=2))

print(metrics.round(3).to_string(index=False))
print(by_study.round(3).to_string(index=False))
print(uncertainty.round(3).to_string(index=False))
print("DDR genes in every fold:", ddr_stable)
print(f"dose regression: MAE={dose_mae:.1f} Gy, rho={dose_rho:.3f}")
print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
