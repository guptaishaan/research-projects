# Papers

One single-column manuscript per research notebook in this repository, in the
ICAISE 2026 submission format (the ACM Primary Article Template, typeset with
`acmart`'s `manuscript` option). Each directory holds the LaTeX source, a
figure-generation script, the compiled PDF, and a zip of the LaTeX sources.

| Directory | Paper | Source notebook |
|---|---|---|
| `heart-disease-cross-hospital/` | Cross-Hospital Generalization of Heart-Disease Classifiers | `Heart Disease Cross Hospital Research.ipynb` |
| `clip-forgetting-forecasting/` | Early Forecasting of Catastrophic Forgetting in CLIP Fine-Tuning | `clip_forgetting_forecasting/` |
| `bankruptcy-stress-test/` | Does a Bankruptcy Model Trained in Calm Times Survive a Recession? | `bankruptcy_crisis_stress_test/` |
| `exoplanet-model-comparison/` | Accuracy–Interpretability Trade-offs in Exoplanet Transit Modelling | `exoplanet-model-comparison/` |
| `raddose-triage/` | Reframing Plant Radiation Dosimetry as Cross-Study Exposure Triage | `raddose-phytodosimetry/` |
| `eeg-controls-audit/` | Calibrating Perturbation Tests Used to Prove EEG Decoders Read the Brain | `eeg-controls-audit/` |

## Layout of a paper directory

    main.tex            the manuscript
    refs.bib            bibliography
    paperstyle.sty      shared acmart companion (copied from _common/ at build time)
    make_figures.py     regenerates every figure and result table
    figures/*.pdf       vector figures
    results_*.csv|json  the numbers the manuscript quotes
    <name>.pdf          compiled manuscript
    <name>-latex.zip    LaTeX sources, ready to upload

## Building

    ./build.sh                       # all papers
    ./build.sh raddose-triage        # just one

`build.sh` copies the shared style file, runs the figure script, then runs
`pdflatex → bibtex → pdflatex → pdflatex`, and reports page count, undefined
references, and overfull boxes. It finally writes the PDF and the source zip.

Set `PAPER_PYTHON` if the scientific stack lives in a virtualenv:

    PAPER_PYTHON=/path/to/venv/bin/python ./build.sh

Requirements: a TeX Live installation with `acmart`, `cleveref`, and `dsfont`; Python
with `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, and `pyarrow`.

`tectonic` also builds every paper without a local TeX Live, fetching `acmart` and
`ACM-Reference-Format.bst` on first run:

    cd raddose-triage && tectonic -k main.tex

## How the numbers are produced

Three papers recompute their results from data committed to this repository, so every
figure and table is regenerated on each build:

- `raddose-triage` re-runs the full leave-one-study-out benchmark, the study-level
  bootstrap, the gene-stability analysis, and the secondary dose regression.
- `clip-forgetting-forecasting` re-runs the leave-one-dataset-out forecasting
  evaluation, and adds a persistence control absent from the source notebook.
- `eeg-controls-audit` re-runs the simulator and the toy calibration experiment.

Three depend on data that is licensed or too large to redistribute. Their measured
results are transcribed from the executed notebooks and written back out as CSV, and
the figures are rebuilt from those tables:

- `heart-disease-cross-hospital` reads the exported score tables in `outputs/results/`.
- `exoplanet-model-comparison` transcribes the leaderboards and recomputes the transit
  forward model and the label-free residual screen from the embedded Kepler sample.
- `bankruptcy-stress-test` transcribes the stress-test results. Supply the raw panel via
  `BANKRUPTCY_CSV` to enable the per-year analysis.

## Shared assets

    _common/paperstyle.sty   acmart companion: extra packages, \ind, equal-contribution marks
    _common/figstyle.py      matplotlib styling; embeds TrueType, not Type 3
