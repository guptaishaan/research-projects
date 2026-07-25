"""Regenerate the figures for the exoplanet accuracy--interpretability paper.

The source study ships results inside its executed notebook rather than as data
files, so the measured leaderboards are transcribed here verbatim (see
`results_detection.csv` / `results_characterization.csv`, written on each run) and
the illustrative transit-shape panel is recomputed from the same analytic
Mandel--Agol quadratic limb-darkening model the study used.
"""
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

FAM_COLOR = {"linear": "#2ca02c", "tree": "#ff7f0e", "neural net": "#d62728",
             "KAN": "#1f77b4", "physics-KAN": "#9467bd"}

# ---------------------------------------------------------------- measured results
DET = pd.DataFrame([
    ("MLP",            "neural net", False, 0.975, 0.573),
    ("XGBoost",        "tree",       False, 0.964, 0.822),
    ("RandomForest",   "tree",       False, 0.959, 0.903),
    ("CNN",            "neural net", False, 0.956, 0.768),
    ("Logistic/Ridge", "linear",     True,  0.946, 0.999),
    ("KAN",            "KAN",        True,  0.943, 0.900),
    ("LSTM",           "neural net", False, 0.906, 0.728),
], columns=["model", "family", "readable", "ROC_AUC", "additivity"])

REC = pd.DataFrame([
    ("RandomForest",   "tree",        False, 0.0081, 0.4762),
    ("CNN",            "neural net",  False, 0.0081, 0.4262),
    ("MLP",            "neural net",  False, 0.0084, 0.2977),
    ("XGBoost",        "tree",        False, 0.0087, 0.3144),
    ("LSTM",           "neural net",  False, 0.0087, 0.4841),
    ("Physics-KAN",    "physics-KAN", False, 0.0100, 0.5298),
    ("KAN",            "KAN",         True,  0.0145, 0.9888),
    ("Logistic/Ridge", "linear",      True,  0.0150, 0.9974),
], columns=["model", "family", "readable", "MAE", "additivity"])
RP_LO, RP_HI = 0.03, 0.15
REC["skill"] = 1 - REC.MAE / (RP_HI - RP_LO)

REAL = pd.DataFrame([
    ("Kepler-719 b", 0.080, 0.081), ("Kepler-734 b", 0.042, 0.060),
    ("Kepler-485 b", 0.121, 0.076), ("Kepler-548 b", 0.122, 0.070),
    ("Kepler-550 b", 0.049, 0.060), ("Kepler-170 b", 0.030, 0.052),
    ("Kepler-228 d", 0.040, 0.055), ("Kepler-1624 b", 0.109, 0.108),
    ("Kepler-855 b", 0.074, 0.081), ("Kepler-546 b", 0.056, 0.069),
    ("Kepler-244 b", 0.031, 0.054), ("Kepler-12 b", 0.119, 0.046),
], columns=["planet", "published", "recovered"])

DET.to_csv(HERE / "results_detection.csv", index=False)
REC.to_csv(HERE / "results_characterization.csv", index=False)
REAL.to_csv(HERE / "results_real_kepler.csv", index=False)

# ---------------------------------------------------------------- transit model
PHASE_W, L, U_LD = 0.10, 201, (0.3, 0.2)


def phase_grid():
    return np.linspace(-PHASE_W, PHASE_W, L)


def _overlap(d, R, rp):
    """Area of intersection between a stellar annulus of radius R and the planet disc."""
    d = np.maximum(d, 1e-7)
    sep, small = R + rp, min(R, rp)
    x1 = np.clip((d ** 2 + R ** 2 - rp ** 2) / (2 * d * R), -1 + 1e-9, 1 - 1e-9)
    x2 = np.clip((d ** 2 + rp ** 2 - R ** 2) / (2 * d * rp), -1 + 1e-9, 1 - 1e-9)
    tri = np.maximum((-d + R + rp) * (d + R - rp) * (d - R + rp) * (d + R + rp), 1e-12)
    part = R ** 2 * np.arccos(x1) + rp ** 2 * np.arccos(x2) - 0.5 * np.sqrt(tri)
    return np.where(d >= sep, 0.0, np.where(d <= abs(R - rp), np.pi * small ** 2, part))


def transit_flux(t, rp, b, a, u=U_LD, n_rings=256):
    """Quadratically limb-darkened transit light curve by annulus decomposition."""
    theta = 2 * np.pi * np.asarray(t)
    d = np.sqrt(a ** 2 * np.sin(theta) ** 2 + b ** 2 * np.cos(theta) ** 2)
    edges = np.linspace(0, 1, n_rings + 1)
    rmid = 0.5 * (edges[1:] + edges[:-1])
    mu = np.sqrt(np.maximum(1 - rmid ** 2, 0.0))
    I = 1 - u[0] * (1 - mu) - u[1] * (1 - mu) ** 2
    F0 = (I * np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)).sum()
    prev = _overlap(d, max(edges[0], 1e-6), rp)
    occ = np.zeros_like(d)
    for k in range(n_rings):
        nxt = _overlap(d, edges[k + 1], rp)
        occ += I[k] * (nxt - prev)
        prev = nxt
    return 1.0 - occ / F0


# ---------------------------------------------------------------- Figure 1
t = phase_grid()
cases = [((0.12, 0.0, 12), "large planet\n$R_p/R_*=0.12$"),
         ((0.05, 0.0, 12), "small planet\n$R_p/R_*=0.05$"),
         ((0.12, 0.9, 12), "grazing\n$b=0.9$"),
         ((0.12, 0.0, 8),  "tighter orbit\n$a/R_*=8$")]
fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.4), sharey=True)
for ax, ((rp, b, a), title) in zip(axes, cases):
    ax.plot(t, transit_flux(t, rp, b, a), lw=1.5, color="#1f77b4")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("orbital phase")
axes[0].set_ylabel("relative flux")
fig.savefig(FIG / "fig1_transit_shapes.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2),
                         gridspec_kw={"width_ratios": [1, 1.05]})
d = DET.sort_values("ROC_AUC")
axes[0].barh(d.model, d.ROC_AUC, color=[FAM_COLOR[f] for f in d.family],
             edgecolor="k", lw=0.3)
for y, v in enumerate(d.ROC_AUC):
    axes[0].text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=7)
axes[0].set_xlim(0.85, 1.0)
axes[0].set_xlabel("detection ROC-AUC")
axes[0].set_title("(a) detection leaderboard")

r = REC.sort_values("MAE", ascending=False)
axes[1].barh(r.model, r.MAE * 1e3, color=[FAM_COLOR[f] for f in r.family],
             edgecolor="k", lw=0.3)
for y, v in enumerate(r.MAE):
    axes[1].text(v * 1e3 + 0.15, y, f"{v:.4f}", va="center", fontsize=7)
axes[1].set_xlim(0, 19)
axes[1].set_xlabel(r"radius-ratio MAE $(\times 10^{-3})$, lower is better")
axes[1].set_title("(b) characterisation leaderboard")
handles = [plt.Line2D([0], [0], marker="s", ls="", mfc=c, mec="k", ms=6, label=f)
           for f, c in FAM_COLOR.items()]
axes[1].legend(handles=handles, fontsize=6.5, loc="lower right", ncol=2)
fig.savefig(FIG / "fig2_leaderboards.pdf")
plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def pareto_panel(ax, df, ycol, ylabel, title):
    for _, row in df.iterrows():
        ax.scatter(row.additivity, row[ycol],
                   s=150 if row.readable else 70,
                   marker="*" if row.readable else "o",
                   color=FAM_COLOR[row.family], edgecolor="k", lw=0.5, zorder=3)
        ax.annotate(row.model, (row.additivity, row[ycol]), fontsize=6.8,
                    xytext=(5, 4), textcoords="offset points")
    pts = df[["additivity", ycol]].values
    order = pts[np.argsort(-pts[:, 0])]
    front, best = [], -np.inf
    for xx, yy in order:
        if yy >= best:
            front.append((xx, yy)); best = yy
    front = np.array(sorted(front))
    ax.plot(front[:, 0], front[:, 1], "--", color="0.45", lw=1, zorder=1,
            label="Pareto frontier")
    ax.set_xlabel("additive-surrogate fidelity $\\mathcal{A}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower left")


fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
pareto_panel(axes[0], DET, "ROC_AUC", "detection ROC-AUC",
             "(a) detection")
axes[0].set_xlim(0.5, 1.08); axes[0].set_ylim(0.89, 0.995)
pareto_panel(axes[1], REC, "skill", r"skill $=1-\mathrm{MAE}/\Delta R_p$",
             "(b) characterisation")
axes[1].set_xlim(0.24, 1.10); axes[1].set_ylim(0.86, 0.95)
fig.savefig(FIG / "fig3_pareto.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
# The source study screened real folds by *reconstruction* residual, which uses no
# published radius and is therefore label-free.  We replicate that rule exactly:
# rebuild each fold from the recovered Rp/R* with the study's fixed nuisance values
# (b = 0.3, a/R* = 12, t0 = 0) and compare against the observed fold.
kep = np.load(HERE / "data" / "kepler_folded.npz", allow_pickle=True)
real_X = kep["X"].astype(float)
observed = real_X - np.median(real_X, axis=1, keepdims=True)
recon_resid = np.array([
    np.mean(np.abs((transit_flux(t, REAL.recovered.iloc[i], 0.3, 12.0) - 1.0)
                   - observed[i]))
    for i in range(len(REAL))
])
good = recon_resid <= np.quantile(recon_resid, 0.75)

err = np.abs(REAL.recovered - REAL.published)
mae_all = float(err.mean())
mae_sub = float(err[good].mean())

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6),
                         gridspec_kw={"width_ratios": [1.05, 1]})
ax = axes[0]
ax.plot(phase_grid(), real_X[0], lw=1.3, color="#1f77b4", label="observed fold")
ax.plot(phase_grid(),
        transit_flux(t, REAL.recovered.iloc[0], 0.3, 12.0)
        + np.median(real_X[0]) - 1.0,
        lw=1.2, ls="--", color="#9467bd", label="physics-KAN reconstruction")
ax.set_xlabel("orbital phase"); ax.set_ylabel("relative flux")
ax.set_title(f"(a) a real Kepler fold: {REAL.planet.iloc[0]}")
ax.legend(fontsize=7, loc="lower right")

ax = axes[1]
ax.plot([0.02, 0.135], [0.02, 0.135], "k--", lw=1, label="perfect recovery")
ax.scatter(REAL.published[good], REAL.recovered[good], s=52, color="#9467bd",
           edgecolor="k", lw=0.5, zorder=3, label="low reconstruction residual")
ax.scatter(REAL.published[~good], REAL.recovered[~good], s=52, color="0.65",
           marker="X", edgecolor="k", lw=0.5, zorder=3, label="high residual (rejected)")
ax.set_xlabel(r"published $R_p/R_*$")
ax.set_ylabel(r"zero-shot physics-KAN $R_p/R_*$")
ax.set_title(f"(b) full-sample MAE $= {mae_all:.3f}$;\n"
             f"screened-subset MAE $= {mae_sub:.3f}$", fontsize=8.5)
ax.legend(fontsize=6.5, loc="upper left")
ax.set_xlim(0.02, 0.135); ax.set_ylim(0.02, 0.135); ax.set_aspect("equal")
fig.savefig(FIG / "fig4_real_kepler.pdf")
plt.close(fig)

REAL["recon_residual"] = recon_resid
REAL["retained"] = good
REAL.round(5).to_csv(HERE / "results_real_kepler.csv", index=False)

print(f"real-data MAE: full={mae_all:.4f}  screened={mae_sub:.4f}  "
      f"n_retained={int(good.sum())}/{len(REAL)}")
print("rejected:", list(REAL.planet[~good]))
print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
