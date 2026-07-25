# Research Projects

Six self-contained research notebooks, each written to be readable without prior
background in its field, and a formal paper written from each one.

| Project | Question | Headline result |
|---|---|---|
| `Heart Disease Cross Hospital Research.ipynb` | Does a clinical risk model survive a move to another hospital? | 0.910 internal AUC; directed transfers range 0.571–0.888, so a single "generalization gap" hides the structure |
| `bankruptcy_crisis_stress_test/` | Does a bankruptcy model trained in calm years survive a recession? | A random forest improves under the 2008–09 crisis (0.755 → 0.798 AUC) while linear and neural models slip |
| `clip_forgetting_forecasting/` | Can we forecast how much CLIP will forget, early in fine-tuning? | A 5-epoch warm-up forecasts final forgetting at R² = 0.92 — but a zero-parameter persistence rule already reaches 0.85 |
| `exoplanet-model-comparison/` | What is the exchange rate between accuracy and interpretability? | Transparency costs 0.029 AUC on detection and 85% more error on sizing; a label-free physics-informed model lands in between |
| `raddose-phytodosimetry/` | Can gene expression identify radiation exposure in an unseen study? | 84.2% accuracy and 0.883 AUC under leave-one-study-out, after rejecting exact-dose regression on design grounds |
| `eeg-controls-audit/` | Are the perturbation controls used to validate EEG decoders themselves reliable? | False-alarm rates span 1.3% to 68.8% across decoders — a control's trustworthiness depends on what you point it at |

## Papers

`papers/` holds one single-column, IEEE-style manuscript per notebook, with LaTeX
source, a figure-generation script, the compiled PDF, and a source zip. See
[papers/README.md](papers/README.md) for the build instructions and for which papers
recompute their numbers versus transcribe them from an executed notebook.

```bash
cd papers && ./build.sh
```

## Running a notebook

Each project directory carries its own `README.md` and `requirements.txt`. Two projects
need data that is licensed and therefore not committed:

- `bankruptcy_crisis_stress_test/` needs `mirror/american_bankruptcy.csv`.
- `raddose-phytodosimetry/` ships its harmonized matrix and runs as-is.

The rest are self-contained. Every notebook is committed with outputs rendered, so all
of them can be read without being run.

## Conventions shared across the projects

- Preprocessing, feature selection, and hyperparameter choice are fitted strictly inside
  the training boundary, and the notebooks say where that boundary is.
- The independent unit of evaluation is stated explicitly — a hospital, a dataset, a
  study, a subject — and uncertainty is quantified at that level, not at the sample level.
- Every project separates definitions, observations, results, and interpretations, and
  ends with a claim boundary describing what the evidence does not support.
