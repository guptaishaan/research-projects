"""Regenerate figures for the EEG perturbation-control calibration paper.

Two sources feed this script:

* the operating characteristics, real-decoder audit, and eyes-open/closed validation,
  which are the measured outputs of the full-scale study and are embedded in the source
  notebook (transcribed here and written back out as CSV); and
* the simulator panels, which are recomputed live from the notebook's own generator so
  the testbed figure is genuinely reproduced rather than copied.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent / "_common"))
import figstyle  # noqa: E402

plt = figstyle.use()
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

np.random.seed(0)

# ---------------------------------------------------------------- simulator
CHANNELS = {
    "F3": (-0.40, 0.72, "frontal"), "Fz": (0.00, 0.80, "frontal"),
    "F4": (0.40, 0.72, "frontal"),
    "FC3": (-0.42, 0.36, "motor"), "FC4": (0.42, 0.36, "motor"),
    "T7": (-0.85, 0.00, "other"), "C3": (-0.50, 0.00, "motor"),
    "Cz": (0.00, 0.00, "motor"), "C4": (0.50, 0.00, "motor"),
    "T8": (0.85, 0.00, "other"),
    "CP3": (-0.42, -0.36, "motor"), "CP4": (0.42, -0.36, "motor"),
    "P3": (-0.40, -0.62, "other"), "Pz": (0.00, -0.66, "other"),
    "P4": (0.40, -0.62, "other"),
    "PO3": (-0.30, -0.82, "occipital"), "PO4": (0.30, -0.82, "occipital"),
    "O1": (-0.20, -0.93, "occipital"), "O2": (0.20, -0.93, "occipital"),
}
CH_NAMES = list(CHANNELS)
N_CH = len(CH_NAMES)
REGION = {n: CHANNELS[n][2] for n in CH_NAMES}
MOTOR = [i for i, n in enumerate(CH_NAMES) if REGION[n] == "motor"]
OCCIPITAL = [i for i, n in enumerate(CH_NAMES) if REGION[n] == "occipital"]
LEFT_MOTOR = [i for i in MOTOR if CH_NAMES[i][-1] in "13"]
RIGHT_MOTOR = [i for i in MOTOR if CH_NAMES[i][-1] in "24"]

FS, DURATION = 128, 2.0
N_TIME = int(FS * DURATION)
TIME = np.arange(N_TIME) / FS
MU, BETA = (8, 12), (16, 28)


def pink_noise(n_ch, n_time, rng):
    freqs = rfftfreq(n_time, 1 / FS)
    scale = np.ones_like(freqs)
    scale[1:] = 1 / np.sqrt(freqs[1:])
    spec = (rng.standard_normal((n_ch, freqs.size))
            + 1j * rng.standard_normal((n_ch, freqs.size))) * scale
    x = irfft(spec, n=n_time, axis=1)
    return x / (x.std(axis=1, keepdims=True) + 1e-9)


def make_trial(label, strength, rng):
    x = pink_noise(N_CH, N_TIME, rng)
    burst = strength * np.sin(2 * np.pi * 10 * TIME) * np.hanning(N_TIME)
    for ch in (LEFT_MOTOR if label == 1 else RIGHT_MOTOR):
        x[ch] += burst
    return x


def make_dataset(n_per_class, strength, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            X.append(make_trial(label, strength, rng)); y.append(label)
    return np.array(X), np.array(y)


def band_power(X, band):
    F = rfft(X, axis=2); freqs = rfftfreq(N_TIME, 1 / FS)
    m = (freqs >= band[0]) & (freqs <= band[1])
    return np.log((np.abs(F[:, :, m]) ** 2).mean(axis=2) + 1e-12)


def mu_power_per_channel(X):
    F = rfft(X, axis=2); freqs = rfftfreq(N_TIME, 1 / FS)
    m = (freqs >= MU[0]) & (freqs <= MU[1])
    return (np.abs(F[:, :, m]) ** 2).mean(axis=2)


def features_focused(X):
    mu = band_power(X, MU)
    return np.stack([mu[:, LEFT_MOTOR].mean(1), mu[:, RIGHT_MOTOR].mean(1)], axis=1)


def features_broad(X):
    return np.concatenate([band_power(X, MU), band_power(X, BETA)], axis=1)


def remove_band(X, band):
    F = rfft(X, axis=2); freqs = rfftfreq(N_TIME, 1 / FS)
    F[:, :, (freqs >= band[0]) & (freqs <= band[1])] = 0.0
    return irfft(F, n=N_TIME, axis=2)


def silence_channels(X, idx):
    X2 = X.copy(); X2[:, idx, :] = 0.0
    return X2


CONTROLS = {
    "remove_mu": lambda X: remove_band(X, MU),
    "remove_motor": lambda X: silence_channels(X, MOTOR),
    "remove_beta": lambda X: remove_band(X, BETA),
    "remove_occipital": lambda X: silence_channels(X, OCCIPITAL),
}
AIS = [("Focused", features_focused, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
       ("Broad", features_broad, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
       ("Tree", features_broad, lambda: RandomForestClassifier(n_estimators=200, random_state=0))]

# ---------------------------------------------------------------- Figure 1
present, yb = make_dataset(40, 1.4, 11)
absent, _ = make_dataset(40, 0.0, 11)
Pm = mu_power_per_channel(present)
gt = np.log(Pm[yb == 1].mean(0) + 1e-9) - np.log(Pm[yb == 0].mean(0) + 1e-9)

rc = {"motor": "#D1651D", "occipital": "#2E7D46", "frontal": "#3B5FA4", "other": "#BEBEBE"}
fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9))
ax = axes[0]
ax.add_patch(plt.Circle((0, 0), 1.05, fill=False, color="k", lw=1.3))
ax.plot([0, -.1, .1, 0], [1.05, 1.18, 1.18, 1.05], color="k", lw=1.3)
for n, (x, yy, r) in CHANNELS.items():
    ax.scatter(x, yy, s=90, color=rc[r], edgecolor="k", lw=.4, zorder=3)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.75, 1.3); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a) montage", loc="left")
ax.legend(handles=[Patch(facecolor=rc[k], edgecolor="k", label=k.title())
                   for k in ["motor", "occipital", "frontal"]],
          loc="lower center", bbox_to_anchor=(.5, 0.0), frameon=False, fontsize=7,
          ncol=3, columnspacing=1, handlelength=1)

ax = axes[1]
ci = CH_NAMES.index("C3")
for X, lab, c in [(absent, "no signal", "#9a9a9a"), (present, "signal present", "#D1651D")]:
    f, P = welch(X[:, ci, :], fs=FS, nperseg=N_TIME, axis=1)
    ax.semilogy(f, P.mean(0), color=c, lw=1.5, label=lab)
ax.axvspan(8, 12, color="#D1651D", alpha=.12)
ax.set_xlim(2, 40); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power (a.u.)")
ax.set_title("(b) planted signal at C3", loc="left")
ax.legend(fontsize=7)

ax = axes[2]
vmax = np.abs(gt).max()
sc = ax.scatter([CHANNELS[n][0] for n in CH_NAMES], [CHANNELS[n][1] for n in CH_NAMES],
                s=150, c=gt, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                edgecolor="k", lw=.4, zorder=3)
ax.add_patch(plt.Circle((0, 0), 1.05, fill=False, color="k", lw=1.3))
ax.plot([0, -.1, .1, 0], [1.05, 1.18, 1.18, 1.05], color="k", lw=1.3)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.35); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(c) planted ground truth", loc="left")
fig.colorbar(sc, ax=ax, fraction=0.045, label=r"$\log$ $\mu$-power, class contrast")
fig.savefig(FIG / "fig1_simulator.pdf")
plt.close(fig)

# ---------------------------------------------------------------- toy experiment
FIRE_THRESHOLD = 0.10
SEEDS = range(15)


def drop_for_control(features, factory, control_fn, seed):
    Xtr, ytr = make_dataset(60, 1.0, seed)
    Xte, yte = make_dataset(60, 1.0, seed + 999)
    ai = factory(); ai.fit(features(Xtr), ytr)
    base = (ai.predict(features(Xte)) == yte).mean()
    hit = (ai.predict(features(control_fn(Xte))) == yte).mean()
    return base - hit


fire = {}
for name, feat, factory in AIS:
    for cname, cfn in CONTROLS.items():
        drops = [drop_for_control(feat, factory, cfn, s) for s in SEEDS]
        fire[(name, cname)] = float(np.mean([d > FIRE_THRESHOLD for d in drops]))

toy = pd.DataFrame([{"decoder": a, "control": c, "fire_rate": v}
                    for (a, c), v in fire.items()])
toy.to_csv(HERE / "results_toy_fire_rates.csv", index=False)

ais = ["Focused", "Broad", "Tree"]
should = ["remove_mu", "remove_motor"]
shouldnt = ["remove_beta", "remove_occipital"]
fig, axs = plt.subplots(1, 2, figsize=(9.0, 2.6))
for ax, rows, title, cmap in [
        (axs[0], should, "(a) target present: should fire ($\\uparrow$ good)", "Greens"),
        (axs[1], shouldnt, "(b) target absent: false alarm ($\\downarrow$ good)", "Reds")]:
    G = np.array([[fire[(a, c)] for a in ais] for c in rows])
    ax.imshow(G, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(ais))); ax.set_xticklabels(ais)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.replace("_", " ") for r in rows])
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    fontweight="bold",
                    color="white" if G[i, j] > .6 else "#333")
    ax.set_xticks(np.arange(-.5, len(ais)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9)
fig.savefig(FIG / "fig2_toy_fire_rates.pdf")
plt.close(fig)

# ---------------------------------------------------------------- measured results
nan = np.nan
PROBES = ["remove_mu", "remove_beta", "remove_mu_beta", "remove_sensorimotor",
          "remove_occipital_control", "remove_theta", "remove_high_control"]
DECODERS = ["bandpower", "csp", "tree", "eegnet"]
DECODER_LABEL = {"bandpower": "Band-power", "csp": "CSP", "tree": "Tree (RF)",
                 "eegnet": "EEGNet"}
OC_CATCH = pd.DataFrame({"bandpower": [.92, .83, .92, 1, 1, nan, nan],
                         "csp": [1, 1, 1, 1, 1, nan, nan],
                         "tree": [1, 1, 1, 1, 1, nan, nan],
                         "eegnet": [1, 1, 1, 1, 1, nan, nan]}, index=PROBES)
OC_FALSE = pd.DataFrame({"bandpower": [.67, .67, nan, .83, .67, .58, .71],
                         "csp": [0, .08, nan, .73, .67, 0, 0],
                         "tree": [.75, .83, nan, .67, .33, .58, .62],
                         "eegnet": [0, 0, nan, 0, 0, .08, 0]}, index=PROBES)
REAL_ACC = pd.DataFrame({"bandpower": [.584, .584], "csp": [.617, .675],
                         "eegnet": [.656, .733]},
                        index=["PhysioNet (109)", "BCI-IV (9)"])
AUDIT_COLS = ["remove_mu", "remove_beta", "remove_sensorimotor",
              "remove_occipital_control"]
AUDIT = pd.DataFrame([[1, 1, 1, 1], [0, 2, 1, 1], [1, 1, 1, 1], [0, 2, 0, 2]],
                     index=DECODERS, columns=AUDIT_COLS)
AUDIT_DROP = pd.DataFrame([[.03, .04, .08, .08], [.02, .01, .13, .07],
                           [.04, .05, .05, .02], [.04, .01, .09, .01]],
                          index=DECODERS, columns=AUDIT_COLS)

OC_CATCH.to_csv(HERE / "results_catch_rate.csv")
OC_FALSE.to_csv(HERE / "results_false_alarm_rate.csv")
AUDIT_DROP.to_csv(HERE / "results_audit_drops.csv")
REAL_ACC.to_csv(HERE / "results_real_accuracy.csv")

CORE = ["remove_mu", "remove_beta", "remove_sensorimotor", "remove_occipital_control"]
CORE_LABEL = {"remove_mu": r"Remove $\mu$", "remove_beta": r"Remove $\beta$",
              "remove_sensorimotor": "Remove sensorimotor",
              "remove_occipital_control": "Remove occipital"}

# ---------------------------------------------------------------- Figure 3
fig, axs = plt.subplots(1, 2, figsize=(9.4, 2.8))


def heat(ax, H, cmap, title):
    A = H.reindex(CORE)[DECODERS].values.astype(float)
    ax.imshow(A, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(DECODERS)))
    ax.set_xticklabels([DECODER_LABEL[m] for m in DECODERS], rotation=18, ha="right",
                       fontsize=8)
    ax.set_yticks(range(len(CORE)))
    ax.set_yticklabels([CORE_LABEL[p] for p in CORE], fontsize=8)
    ax.set_xticks(np.arange(-.5, len(DECODERS)), minor=True)
    ax.set_yticks(np.arange(-.5, len(CORE)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j] * 100:.0f}\\%", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if ((cmap == "Greens" and A[i, j] > .6)
                                      or (cmap == "Reds" and A[i, j] > .5)) else "#333")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9)


heat(axs[0], OC_CATCH, "Greens", "(a) sensitivity: catches the real signal ($\\uparrow$ good)")
heat(axs[1], OC_FALSE, "Reds", "(b) false-alarm rate ($\\downarrow$ good)")
fig.savefig(FIG / "fig3_operating_characteristics.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4
CAT_COLORS = ["#1B7837", "#D95F02", "#4575B4"]
CAT_LABEL = ["Uses it (trust)", "Suspect", "Does not use it (trust)"]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0),
                         gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
M = AUDIT[AUDIT_COLS].values
ax.imshow(M, cmap=ListedColormap(CAT_COLORS), vmin=0, vmax=2, aspect="auto")
ax.set_xticks(range(len(AUDIT_COLS)))
ax.set_xticklabels([CORE_LABEL[c] for c in AUDIT_COLS], fontsize=8)
ax.set_yticks(range(len(DECODERS)))
ax.set_yticklabels([DECODER_LABEL[m] for m in DECODERS], fontsize=8)
ax.set_xticks(np.arange(-.5, len(AUDIT_COLS)), minor=True)
ax.set_yticks(np.arange(-.5, len(DECODERS)), minor=True)
ax.grid(which="minor", color="white", lw=2)
ax.tick_params(which="minor", length=0)
word = {0: "trust", 1: "suspect", 2: "trust"}
for i in range(len(DECODERS)):
    for j in range(len(AUDIT_COLS)):
        ax.text(j, i, word[M[i, j]], ha="center", va="center", color="white",
                fontweight="bold", fontsize=8.5)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("(a) calibrated verdicts on real PhysioNet decoders", loc="left", fontsize=9)
ax.legend(handles=[Patch(facecolor=CAT_COLORS[k], label=CAT_LABEL[k]) for k in range(3)],
          loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3, frameon=False,
          fontsize=7)

ax = axes[1]
x = np.arange(len(REAL_ACC.columns)); w = 0.36
for k, (row, c) in enumerate(zip(REAL_ACC.index, ["#4C72B0", "#55A868"])):
    ax.bar(x + (k - 0.5) * w, REAL_ACC.loc[row], w, label=row, color=c,
           edgecolor="k", lw=0.4)
ax.axhline(.5, ls="--", color="k", lw=1)
ax.text(len(x) - 0.6, 0.515, "chance", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([DECODER_LABEL[m] for m in REAL_ACC.columns], fontsize=8)
ax.set_ylim(0, 0.85); ax.set_ylabel("balanced accuracy")
ax.set_title("(b) decodability floor", loc="left", fontsize=9)
ax.legend(fontsize=7, loc="upper left")
fig.savefig(FIG / "fig4_calibrated_audit.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 5
EOEC_CONTROLS = [r"Remove $\alpha$ ($\mu$)", "Remove occipital", r"Remove $\beta$",
                 r"Remove $\theta$", "Remove sensorimotor", "Remove frontal"]
EOEC_TRUTH = [True, True, False, False, False, False]
EOEC = {"Band-power": {"fire": [1, 1, 0, 1, 1, 1], "drop": [.07, .23, .06, .19, .21, .23]},
        "CSP": {"fire": [0, 1, 0, 0, 1, 1], "drop": [.09, .36, .03, .00, .34, .36]}}
pd.DataFrame({k: v["drop"] for k, v in EOEC.items()},
             index=EOEC_CONTROLS).to_csv(HERE / "results_eyes_open_closed.csv")


def cell_category(truth, fired):
    return 0 if (truth and fired) else (2 if (not truth and fired)
                                        else (3 if (truth and not fired) else 1))


CC = ["#1B7837", "#9FC6E7", "#D95F02", "#9E9E9E"]
CLABEL = ["Caught real signal", "Correctly silent", "FALSE ALARM", "Missed"]
cols = list(EOEC)
M = np.array([[cell_category(EOEC_TRUTH[i], EOEC[c]["fire"][i]) for c in cols]
              for i in range(len(EOEC_CONTROLS))])
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.imshow(M, cmap=ListedColormap(CC), vmin=0, vmax=3, aspect="auto")
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
ax.set_yticks(range(len(EOEC_CONTROLS)))
ax.set_yticklabels([("$\\blacktriangle$ " if EOEC_TRUTH[i] else "$\\bullet$ ")
                    + EOEC_CONTROLS[i] for i in range(len(EOEC_CONTROLS))], fontsize=8)
ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
ax.set_yticks(np.arange(-.5, len(EOEC_CONTROLS)), minor=True)
ax.grid(which="minor", color="white", lw=2)
ax.tick_params(which="minor", length=0)
for i in range(len(EOEC_CONTROLS)):
    for j in range(len(cols)):
        ax.text(j, i, f"{EOEC[cols[j]]['drop'][i]:+.2f}", ha="center", va="center",
                color="white" if M[i, j] in (0, 2) else "#222", fontsize=8,
                fontweight="bold")
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(handles=[Patch(facecolor=CC[k], label=CLABEL[k]) for k in range(4)],
          loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, frameon=False,
          fontsize=7)
fig.savefig(FIG / "fig5_real_brain_validation.pdf")
plt.close(fig)

summary = {
    "toy_fire_rates": {f"{a}|{c}": v for (a, c), v in fire.items()},
    "false_alarm_mean": {d: round(float(OC_FALSE[d].mean()), 3) for d in DECODERS},
    "catch_mean": {d: round(float(OC_CATCH[d].mean()), 3) for d in DECODERS},
    "eoec_false_alarms": {c: int(sum(1 for i in range(len(EOEC_TRUTH))
                                     if not EOEC_TRUTH[i] and EOEC[c]["fire"][i]))
                          for c in cols},
}
(HERE / "results_summary.json").write_text(json.dumps(summary, indent=2))
print(toy.pivot(index="control", columns="decoder", values="fire_rate").to_string())
print(json.dumps(summary["false_alarm_mean"], indent=1))
print(json.dumps(summary["eoec_false_alarms"], indent=1))
print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
