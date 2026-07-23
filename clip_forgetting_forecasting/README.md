# Forecasting Catastrophic Forgetting in CLIP

A complete, self-contained machine-learning study. When you fine-tune CLIP on a specialized
dataset it improves on that task but **forgets** general knowledge. This project asks whether we
can **forecast how much it will forget, early in training**, from a short warm-up of the training
dynamics.

**Main result.** Static pre-training features alone forecast final forgetting at R² ≈ 0.40
(leave-one-dataset-out). Adding a 5-epoch warm-up of training-dynamics signals raises this to
R² ≈ 0.91 and roughly halves the error. Simple models (ridge, random forest, gradient boosting)
beat an MLP and an LSTM, which overfit on the 48-run dataset.

## Contents

| File | Purpose |
|---|---|
| `CLIP_Forgetting_Forecasting.ipynb` | The full study: preprocessing, training, evaluation, comparison. Start here. |
| `trajectory_dynamics.py` | Generates the experiment (48 fine-tuning runs, logs dynamics each epoch). |
| `forecasting.py` | Builds the prediction table and runs the five models with leave-one-dataset-out. |
| `outputs/dynamics/*.csv` | Cached results the notebook reads (so it runs in seconds). |
| `requirements.txt` | Python dependencies. |

## Running it

```bash
pip install -r requirements.txt
jupyter notebook CLIP_Forgetting_Forecasting.ipynb
```

The notebook is already executed with all figures embedded, so it can also just be read. If you
run it, image datasets download automatically on first use (~6 GB, mostly Food101). A GPU is
recommended but only strictly needed for the short live training demo in Part 5.

## Regenerating from scratch

The cached CSVs can be rebuilt end to end (a few hours on one GPU):

```bash
python trajectory_dynamics.py --epochs 20 --train-n 4000 --eval-n 1500 --lrs 1e-6 5e-6 1e-5
```

This writes `outputs/dynamics/features.csv`, `dynamics_curves.csv` and `baseline.csv`, which the
notebook then loads.

## Honest limitations

- Only 16 datasets, so "new dataset" generalization estimates are noisy — trust the direction, not
  the third decimal.
- Part of the warm-up's advantage is simple curve continuation.
- One architecture, one fine-tuning method, one optimizer, one run per configuration (no seeds).
