# ICAISE 2026 submission metadata

Generated from the current compiled manuscripts. Student authors are listed
alphabetically by surname, affiliated to Nexus Labs, and marked for equal contribution;
Srikanth Samy and Ishaan Gupta close each author list.

## Early Forecasting of Catastrophic Forgetting in CLIP Fine-Tuning: How Much Does Watching Training Actually Tell You?

- **Directory:** `papers/clip-forgetting-forecasting/`
- **Cluster:** Computer Vision (`2wvxlhkw`)
- **Keywords:** Catastrophic forgetting, CLIP, fine-tuning dynamics, grouped cross-validation, multimodal learning, transfer learning, zero-shot classification.

**Abstract**

Fine-tuning a vision–language model on a narrow dataset improves the target task while eroding the zero-shot competence that made the model valuable. We ask whether the final magnitude of that erosion can be forecast from a short warm-up of the fine-tuning run. Our experiment comprises 48 fine-tuning runs of CLIP ViT-B/32 (16 specialised datasets × 3 learning rates × 20 epochs), logging five trajectory measurements after every epoch and measuring forgetting as the mean drop in zero-shot accuracy over CIFAR-100, CIFAR-10, and STL-10. Forecasters are evaluated by leave-one-dataset-out cross-validation, so that all three learning-rate runs of a held-out dataset are unseen. Four cheap dataset descriptors plus learning rate reach R² = 0.43; adding a five-epoch warm-up raises this to R² = 0.92 and cuts mean absolute error from 0.070 to 0.025. Simple tabular regressors beat both an MLP (R² = −0.66) and an LSTM (R² = 0.81) in this 48-row, 16-group regime. Critically, we compare raw persistence with a calibrated persistence baseline fitted inside each training fold. Calibrated persistence attains R² = 0.919 and MAE 0.026, statistically indistinguishable at the dataset level from gradient boosting (R² = 0.917, MAE 0.025; MAE-difference 95% CI [−0.0065, 0.0047]). Thus the warm-up contains strong predictive information, but this campaign does not show that drift or loss features improve on a simple calibration of Φ₅. Future work must report both raw and calibrated persistence controls.

## The Control Is the Instrument: Calibrating Perturbation Tests Used as Evidence That EEG Decoders Read Neural Signal

- **Directory:** `papers/eeg-controls-audit/`
- **Cluster:** Biotechnology-A (`ocagun6c`)
- **Keywords:** brain–computer interfaces, EEG, explainability, perturbation tests, simulation, validation

**Abstract**

A standard argument in brain–computer-interface research runs: we removed the μ band, accuracy dropped, therefore the decoder uses sensorimotor rhythm. The argument is valid only if the perturbation has known sensitivity and, critically, a known false-alarm rate for the decoder being tested. Almost no published perturbation control has been characterised that way. We treat controls as measuring instruments and calibrate them against operational ground truth. Using a simulator in which the signal is planted by construction, we measure each control’s catch rate and false-alarm rate across a grid of conditions and four decoder families. The reliability of a control turns out to depend on the decoder, not just on the control. Averaged over probes, EEGNet’s false-alarm rate is 1.3% while band-power’s is 68.8%: on the same tested simulator grid, the identical evidence has sharply different reliability across decoders. These full-grid rates are descriptive because each condition has one generated dataset. In a separate 30-dataset replication of the minimal comparison, conclusions are unchanged for firing thresholds 0.05–0.20. A random forest reading exactly the same features as a band-power linear model cuts spatial false alarms roughly in half while remaining equally fooled by band perturbations, implicating both representation and model class without identifying their interaction. Applying decoder-specific calibration to real PhysioNet drops changes their evidential weight. Finally, we grade the controls on real brains using eyes-open versus eyes-closed, a task whose occipital-alpha ground truth is fixed by physiology: band-power false alarms on three of four controls whose targets are absent and CSP on two, corroborating the specificity warning without establishing motor-imagery ground truth.

## Cross-Hospital Generalization of Heart-Disease Classifiers: A Directed Transfer Analysis on Four UCI Cohorts

- **Directory:** `papers/heart-disease-cross-hospital/`
- **Cluster:** Biotechnology-B (`qvtmt7of`)
- **Keywords:** Clinical machine learning, dataset shift, external validation, heart disease, ROC analysis, reproducibility, transportability.

**Abstract**

A clinical risk model can perform well internally yet transfer poorly to a hospital with a different case mix, measurement practice, or missing-data pattern. We study this failure mode on the four cohorts of the UCI Heart Disease collection (Cleveland, Hungary, Switzerland, Long Beach VA; n = 920 patients). Three model families—L₂-penalized logistic regression, a random forest, and a compact multilayer perceptron—are first evaluated on Cleveland with stratified five-fold cross-validation using all thirteen recorded variables, and are then transported across all twelve directed source→target hospital pairs using only the nine variables with genuine coverage at every site. Imputation and scaling are refitted strictly inside each training boundary, and the Swiss cholesterol field—recorded as zero for every Swiss patient—is excluded from the portable feature set rather than silently imputed from other hospitals. Internal Cleveland discrimination reaches AUC = 0.910 for the random forest. Transfer performance is heterogeneous: directed pairs range from AUC = 0.571 to 0.888. The random forest has macro-average external AUC 0.781 and internal AUC 0.756; this reflects the populations being averaged and does not imply improvement under transfer. Logistic regression has a 0.028 internal-to-external decrease. The MLP changes little but performs worse in absolute terms. We argue that the scalar “generalization gap” routinely reported in the clinical machine-learning literature is an insufficient summary, that source-cohort quality dominates model choice, and that external validity should be reported as a transfer matrix accompanied by absolute target-site performance.

## Does a Bankruptcy Model Trained in Calm Times Survive a Recession? A Temporal Stress Test on U.S. Public Firms

- **Directory:** `papers/bankruptcy-stress-test/`
- **Cluster:** Financial Technology (`bihc5gy3`)
- **Keywords:** bankruptcy prediction, distribution shift, temporal validation, financial ratios, class imbalance

**Abstract**

Corporate-distress models are typically trained and validated on data pooled across time, yet they are deployed forward into economic conditions that may differ sharply from the training period. We ask whether a bankruptcy classifier trained on calm pre-crisis years still functions during a recession. Using 78,682 company-years of U.S. public firms (1999–2018) with 18 financial variables, we frame imminent-failure prediction—a firm’s final reporting year before delisting—and engineer 15 Altman-style financial ratios. Training on 1999–2006 with a company-grouped split to prevent leakage, a random forest attains a held-out ROC-AUC of 0.755, well ahead of logistic regression (0.646) and a multilayer perceptron (0.625); the distress boundary is strongly nonlinear. Stress-testing the frozen calm-era models on the 2008–2009 recession, the random-forest point estimate rises to 0.798 AUC, while the linear and neural estimates slip. Analytic 95% confidence intervals overlap across periods, so the supported conclusion is preservation rather than statistically established improvement. We also show that a naive per-year AUC curve suggests a spurious “cliff” at 2007 that is an evaluation artifact of training-set overlap, underscoring the importance of leakage-free temporal validation.

## Accuracy–Interpretability Trade-offs in Exoplanet Transit Modelling, and a Label-Free Physics-Informed Estimator of Planetary Radii

- **Directory:** `papers/exoplanet-model-comparison/`
- **Cluster:** Aerospace Engineering (`izdqzr3x`)
- **Keywords:** Exoplanets, interpretable machine learning, Kolmogorov–Arnold networks, physics-informed learning, transit photometry, model comparison.

**Abstract**

Transit photometry poses two distinct machine-learning problems: deciding whether a light curve contains a planetary transit, and measuring the planet’s radius ratio Rₚ/R* from it. We compare seven model families on both tasks using simulated light curves with exactly known parameters, and score every model on a second axis besides accuracy—the fidelity with which a purely additive surrogate reproduces its output, an explicit and reproducible operationalisation of one dimension of interpretability. The two axes trade off: on detection, the most accurate model (an MLP, AUC = 0.975) has the second-lowest additive fidelity (A = 0.573), while the fully readable linear model reaches AUC = 0.946 at A = 0.999—a 0.029 AUC premium for near-total transparency. We then introduce a physics-informed Kolmogorov–Arnold network that predicts transit parameters and is trained only to reconstruct the observed flux through a differentiable annular approximation to the Mandel–Agol forward model, never seeing a radius label. It attains MAE = 0.0100 in Rₚ/R*, within 24% of the best fully supervised model, and occupies a distinct region of the Pareto plane under matched simulation. Applied zero-shot to twelve confirmed Kepler planets it recovers radii with full-sample MAE = 0.023, a feasibility result rather than a reliability estimate. We report that full-sample figure as the primary real-data result and show that the label-free reconstruction-residual screen used to identify model mismatch improves it only to 0.018—and rejects one fold whose radius was in fact recovered to within 0.001, demonstrating that reconstruction quality is not a reliable proxy for parameter accuracy.

## When the Target Is Wrong: Reframing Plant Radiation Dosimetry as Cross-Study Exposure Triage

- **Directory:** `papers/raddose-triage/`
- **Cluster:** Experienced Track (`aiw35wob`)
- **Keywords:** Arabidopsis, domain generalization, gene expression, leave-one-study-out validation, radiation biology, study design, triage

**Abstract**

Predicting absorbed radiation dose from gene expression is an attractive framing, but it is only answerable if the underlying experiments actually span the dose axis. Auditing a six-study collection of Arabidopsis thaliana RNA-seq profiles (158 samples, 4,002 genes, NASA OSDR), we find that 138 of 158 samples sit at exactly 0 or 100 Gy and only 20 represent intermediate doses—a design that supports exposed-versus-control triage but not a calibration curve. We therefore change the target and report both outcomes. Under leave-one-study-out validation, in which an entire study is withheld and all adaptive operations including gene selection are fitted inside the training folds, a 50/100-gene stability ensemble reaches 84.2% pooled accuracy, 82.4% balanced accuracy, 0.883 pooled ROCAUC, 0.656 Matthews correlation, 89.0% sensitivity, and 75.9% specificity. The same protocol applied to exact-dose regression yields 33.6 Gy mean absolute error and Spearman ρ = 0.523: the negative result that motivated the reframing. Because the independent unit is a study and not a sample, we bootstrap over the six study-level scores rather than over the 158 samples, giving a macro balanced accuracy of 0.849 with a 95% interval of [0.757, 0.938]—broad, and honestly so. Three canonical DNA-damage-response genes are selected in every outer fold, which makes the classifier biologically plausible without establishing that its markers are specific to radiation rather than to stress in general.
