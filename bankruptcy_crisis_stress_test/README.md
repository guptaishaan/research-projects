# The 2008 Stress Test: Does a Bankruptcy Model Trained in Calm Times Survive a Recession?

A self-contained notebook that builds a corporate-distress classifier on calm pre-crisis
years and then stress-tests it, frozen, on the 2008–2009 recession.

**The question.** Models that flag firms at risk of failure are fit on history and
deployed into the future. Does a distress model trained on calm years still rank failing
firms during a crisis, and which model family holds up best?

**Main result.** On 78,682 company-years of U.S. public firms (1999–2018), a random
forest trained on 1999–2006 with a company-grouped split reaches 0.755 held-out ROC-AUC,
well ahead of logistic regression (0.646) and a small neural network (0.625). Applied
unchanged to 2008–2009 it does not degrade but *improves* to 0.798, while the linear and
neural models slip. Financial distress is a temporally robust — even sharpened — signal.

**The secondary lesson.** A per-year AUC curve appears to show a cliff at 2007 that looks
like the crisis breaking the model. It is an evaluation artifact: pre-2007 years still
contain the exact firms the model trained on, so those scores are inflated. The
leakage-free comparison shows no break at all.

## Contents

| Path | What it is |
|---|---|
| `bankruptcy_crisis_stress_test.ipynb` | The whole project, end to end, with rendered outputs |
| `requirements.txt` | Pinned dependencies |

The paper written from this notebook lives in `../papers/bankruptcy-stress-test/`.

## Data

The notebook reads `mirror/american_bankruptcy.csv`: one row per company per year, with
18 financial figures (`X1`–`X18`), a `status_label` of `alive`/`failed`, and the fiscal
year. The file is licensed and is not committed here. Place a copy at
`mirror/american_bankruptcy.csv` before running Part 1.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook bankruptcy_crisis_stress_test.ipynb
```

Training all three models takes roughly a minute on a CPU. The notebook is committed with
outputs rendered, so it can also just be read.

## Honest limitations

- Positives are rare (0.82% of company-years), so PR-AUC is low and the recession
  estimates rest on a few dozen failures. Read them as directional.
- "Final reporting year" is a proxy for the legal bankruptcy date.
- Firms recur across the calm and recession windows, so the transfer is across *time*
  rather than to entirely new firms.
- One dataset, one design, one seed, and no confidence intervals.
- Survivorship and reporting biases affect which firms appear at all.
