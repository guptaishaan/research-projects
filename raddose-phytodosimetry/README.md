# RadDose-AT: Predicting Radiation Dose from a Plant's Gene Activity

A single, self-contained Jupyter notebook that walks through a complete small research
project in space biology: predicting the absorbed radiation dose of an *Arabidopsis* plant
from its gene-activity profile ("phytodosimetry"). It covers the raw data, the
preprocessing that makes six independent experiments comparable, training and evaluating
predictive models, a comparison of ten model families by how well they generalize to an
unseen study, and an analysis of which genes carry the dose signal.

The notebook is written to be readable without prior biology or machine-learning
background, with each concept explained as it appears.

## Running it

Everything the notebook needs is in this folder:

- `RadDose_Full_Walkthrough.ipynb` — the notebook.
- `raddose_data/` — the cleaned gene-activity matrix, the sample table, two raw studies
  (used for the preprocessing figures), and the saved analysis results the notebook reads
  (about 12 MB in total).

The notebook uses **relative paths**, so after cloning the repository you can open it from
this folder and run all cells top to bottom. The model-comparison section trains ten
models many times and takes roughly two minutes; everything else is fast.

- **Local environment:** `pip install -r requirements.txt`, then launch Jupyter and run
  the notebook from this folder.
- The committed notebook already contains all rendered outputs, so it can be read as a
  finished report on GitHub without running anything.

## What it covers

1. Background: radiation, plants, gene activity, and what an absorbed dose (Gray) is.
2. The data: 158 RNA-seq samples from six NASA OSDR *Arabidopsis* radiation studies.
3. Preprocessing: log transform, per-study z-scoring, and gene selection, each shown on real data.
4. The honest test: leave-one-study-out evaluation, and why it matters.
5. Training and evaluating a ladder of models.
6. Comparing ten model families by their generalization gap (which models transfer to a new study).
7. A minimal causal gene panel that beats the classic 7-gene literature panel.
8. Checking the signal against independent chemical-DNA-damage data.
9. Conclusions and honest limitations.

## Data provenance

The samples come from the NASA Open Science Data Repository (OSDR) radiation studies
OSD-498, OSD-502, OSD-508, OSD-510, OSD-658, and OSD-782 (gamma rays and heavy ions,
0–100 Gy). Expression values are GeneLab-normalized counts, harmonized per study. The
files under `raddose_data/tables/` are saved outputs of the full analysis pipeline; the
notebook recomputes the core benchmark live and loads these for the more expensive
interpretation stages (the causal gene panel and the external-data validation).
