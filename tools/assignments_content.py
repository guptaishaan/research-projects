# -*- coding: utf-8 -*-
"""Authored content for the student assignments. Consumed by build_assignments.py."""

PROJECTS = {

# ===========================================================================
# 1. HEART DISEASE - cross-hospital generalization
# ===========================================================================
"heart_disease": {
    "project": "Cross-Hospital Generalization of Heart-Disease Classifiers",
    "folder": "heart-disease-cross-hospital",
    "notebook": "Heart Disease Cross Hospital Research.ipynb",

    "a1_intro": (
        "Our project asks a simple but important question: a machine-learning model can look "
        "excellent when tested on patients from the same hospital it was trained on, but does it "
        "still work when it is moved to a different hospital with different patients, machines, and "
        "record-keeping? This is called external validation. Before we write our own paper, we need "
        "to understand where the heart-disease data came from, how the medical field says prediction "
        "models should be tested, and why models so often fail when they travel. The papers below "
        "build that background."
    ),
    "a1_papers": [
        ("Detrano, R. et al. (1989). International application of a new probability algorithm for "
         "the diagnosis of coronary artery disease. American Journal of Cardiology, 64(5).",
         "This is the study that produced the Cleveland/Hungary/Switzerland/VA heart-disease data we use. "
         "It tells us who the patients were and how the diagnosis was defined.",
         [
            "What patient measurements (features) did the authors use to predict heart disease?",
            "The data comes from four different hospitals in different countries. Why might that matter for a model trained on only one of them?",
            "How did the authors decide whether a patient truly had coronary artery disease (the 'ground truth' label)?",
         ]),
        ("Collins, G. S. et al. (2015). Transparent Reporting of a multivariable prediction model "
         "for Individual Prognosis Or Diagnosis (TRIPOD). Annals of Internal Medicine, 162.",
         "TRIPOD is the medical community's checklist for how to report a prediction model honestly. "
         "It defines the vocabulary we will use in our own paper.",
         [
            "In your own words, what is the difference between 'developing' a model and 'validating' a model?",
            "What is the difference between internal validation and external validation?",
            "Name two things TRIPOD says a paper should report that are easy to leave out.",
         ]),
        ("Steyerberg, E. W. & Harrell, F. E. (2016). Prediction models need appropriate internal, "
         "internal-external, and external validation. Journal of Clinical Epidemiology, 69.",
         "This short paper argues that one held-out test is not enough - a model should be checked "
         "across settings. This is exactly the gap our project explores.",
         [
            "Why do the authors say a single internal test can give an overly optimistic picture of a model?",
            "What does 'internal-external' validation mean, and how is it different from ordinary external validation?",
            "How does this argument connect to our idea of training on one hospital and testing on another?",
         ]),
        ("Futoma, J. et al. (2020). The myth of generalisability in clinical research and machine "
         "learning in health care. The Lancet Digital Health, 2, e489-e492.",
         "A very readable opinion piece arguing that expecting one model to work everywhere may be the "
         "wrong goal. It helps us frame our conclusions carefully.",
         [
            "What do the authors mean by the 'myth' of generalisability?",
            "Do the authors think a model that only works in one hospital is useless? Explain their view.",
            "How should this change the way we phrase the conclusion of our own study?",
         ]),
    ],
    "a1_closing": (
        "After reading, write two or three sentences describing the research gap our project fills: "
        "why testing heart-disease models across hospitals (not just within one) is worth doing. "
        "These sentences will seed the Introduction of our paper."
    ),

    "a2_intro": (
        "This assignment is about the data and the method - what actually happens inside the notebook "
        "before any results appear. You will look at how disguised missing values are found, how "
        "preprocessing is kept honest, and how the models are trained and moved between hospitals."
    ),
    "a2_groups": [
        ("Part 1, Step 2 - Looking at the data (the 'sneaky problem')",
         "Cholesterol and blood pressure can never truly be 0, so a 0 here is really a missing value in "
         "disguise. The notebook finds these and converts them to real blanks.",
         [
            "Why is a cholesterol value of 0 a problem, and what does the notebook do about it?",
            "The box plots split each measurement by outcome (disease vs. no disease). For 'thalach' (max heart rate), which group tends to be lower? Why does a feature whose two boxes barely overlap make a good predictor?",
         ]),
        ("Part 1, Step 3 & Step 5 - Preparing data and cross-validation",
         "The notebook fills missing values with the median and rescales features, but it does this "
         "inside each training fold only. It uses 5-fold cross-validation because there are only ~303 patients.",
         [
            "What does it mean to fit preprocessing 'inside the training fold only', and why does doing it on the whole dataset count as cheating (leakage)?",
            "Explain 5-fold cross-validation in one or two sentences. Why is it fairer than a single train/test split when you only have 303 patients?",
         ]),
        ("Part 1, Step 4 - The three models",
         "Three models of increasing flexibility are used: logistic regression (a straight-line-style "
         "classifier), a random forest (400 voting trees), and a small neural network (MLP).",
         [
            "Put the three models in order from simplest to most flexible, and give one sentence on what each one does.",
            "Why might the simplest model (logistic regression) be hard to beat on small, clean medical data?",
         ]),
        ("Part 2 - The transfer test (the heatmaps)",
         "In Part 2 the notebook trains on one hospital and tests on a different one, using only the "
         "nine measurements that all four hospitals actually record. Each heatmap has training hospitals "
         "on the rows and testing hospitals on the columns; the boxed diagonal is same-hospital performance.",
         [
            "Why does Part 2 drop cholesterol and use only nine shared features, when Part 1 used all thirteen?",
            "On a transfer heatmap, what does a cell OFF the diagonal represent, and what would a big drop from the diagonal to the off-diagonal tell you?",
            "Switzerland records cholesterol as 0 for every patient. Why can't we simply 'fill in' that missing feature using the other hospitals?",
         ]),
    ],
    "a2_closing": (
        "Your answers here become the Data and Methods section of our paper. Aim for text a reader could "
        "follow without opening the notebook: what the data is, how it was cleaned, and how the models "
        "were trained and transferred between hospitals."
    ),

    "a3_intro": (
        "Now we look at how well the models actually did - inside Cleveland, and after traveling to a new "
        "hospital. The most important idea in this assignment is that a good internal score does not "
        "guarantee a good score somewhere else."
    ),
    "a3_metrics": [
        ("Accuracy", "the fraction of patients classified correctly. Fair here because Cleveland is roughly balanced (about 46% have disease), but it can be misleading when one class is rare."),
        ("ROC-AUC", "how well the model RANKS sick patients above healthy ones. 0.5 is a coin flip, 1.0 is perfect. This is the study's main score because it does not depend on a single decision threshold."),
        ("F1", "a balance of precision (of those flagged sick, how many really were) and recall (of the truly sick, how many were caught)."),
        ("Generalization gap", "internal AUC minus external AUC. A positive gap means the model did worse when it moved to a new hospital."),
    ],
    "a3_results": [
        "What was the best internal (within-Cleveland) ROC-AUC, roughly, and which model achieved the strongest results inside Cleveland?",
        "The notebook reports that transfer is 'heterogeneous rather than uniformly worse' - some hospital pairs got worse and some got better. Why is reporting the full transfer matrix more honest than reporting one average number?",
        "Which model had the highest average external AUC in this run, and which had the largest internal-minus-external gap? What does that contrast suggest about choosing a model for real deployment?",
        "Write two or three sentences comparing internal and external performance that we could paste directly into the Results section of our paper.",
        "Accuracy, ROC-AUC, and F1 can disagree. Give one situation where a model could have high accuracy but a poor ROC-AUC.",
    ],
    "a3_limitations": (
        "The notebook lists real limitations: only four historical cohorts, small and imbalanced hospitals, "
        "no confidence intervals or calibration, and Part 1 vs. Part 2 answering different questions. "
        "Pick the limitation you think matters most, explain why, and propose one concrete next experiment "
        "that would strengthen the study."
    ),
},

# ===========================================================================
# 2. CLIP FORGETTING FORECASTING
# ===========================================================================
"clip_forgetting": {
    "project": "Early Forecasting of Catastrophic Forgetting in CLIP Fine-Tuning",
    "folder": "clip_forgetting_forecasting",
    "notebook": "clip_forgetting_forecasting/CLIP_Forgetting_Forecasting.ipynb",

    "a1_intro": (
        "Our project studies 'catastrophic forgetting': when you fine-tune a general vision-language "
        "model (CLIP) on a narrow new task, it can get better at that task while quietly losing its "
        "broad, general knowledge. Our specific question is whether we can FORECAST how much a model "
        "will forget by the end of training, just by watching its first few epochs. To write the "
        "Introduction of our paper we need to understand what CLIP is, what forgetting is, and what "
        "people already do about it. These papers cover that background."
    ),
    "a1_papers": [
        ("Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language "
         "Supervision (CLIP). Proceedings of ICML.",
         "The paper that introduced CLIP - the model we fine-tune. Focus on the idea of a shared "
         "image-text space and 'zero-shot' classification.",
         [
            "In plain words, how does CLIP learn to connect images and text?",
            "What does 'zero-shot classification' mean, and why is it the ability we are worried about losing?",
            "Why might a model this general be especially risky to fine-tune on one narrow dataset?",
         ]),
        ("French, R. M. (1999). Catastrophic forgetting in connectionist networks. Trends in "
         "Cognitive Sciences, 3(4).",
         "A short, readable review that explains WHY neural networks forget. It gives us the vocabulary "
         "for the core problem.",
         [
            "In your own words, why does learning something new cause a network to overwrite what it already knew?",
            "What is the 'stability-plasticity' tension the author describes?",
            "How does this classic idea apply to fine-tuning a modern model like CLIP?",
         ]),
        ("Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks (EWC). "
         "Proceedings of the National Academy of Sciences, 114(13).",
         "One famous method for reducing forgetting. We do not implement it, but it shows what a "
         "'solution' to forgetting looks like.",
         [
            "What is the main idea behind Elastic Weight Consolidation for preventing forgetting?",
            "This paper tries to PREVENT forgetting. Our project instead tries to PREDICT it early. Why might predicting be useful even if you are not preventing?",
         ]),
        ("Wortsman, M. et al. (2022). Robust fine-tuning of zero-shot models (WiSE-FT). Proceedings "
         "of CVPR.",
         "Directly about keeping CLIP's zero-shot ability while fine-tuning. This is the closest prior "
         "work to our problem.",
         [
            "What problem with fine-tuning CLIP does this paper try to fix?",
            "How does their approach differ from our goal of forecasting forgetting early in training?",
            "What gap does this leave that our 'early-warning' project could fill?",
         ]),
    ],
    "a1_closing": (
        "Write two or three sentences stating our research gap: prior work either prevents forgetting or "
        "predicts it from a fixed pre-training score, but few ask whether the first few epochs of training "
        "can forecast the final damage. This becomes the last paragraph of our Introduction."
    ),

    "a2_intro": (
        "This assignment is about how the experiment is set up: what CLIP receives as input, how "
        "forgetting is measured, and how the problem is turned into a prediction table. You will look at "
        "the preprocessing pipeline and the leave-one-dataset-out design."
    ),
    "a2_groups": [
        ("Part 2 & Part 3 - The datasets and preprocessing",
         "CLIP is fine-tuned on 16 datasets grouped into three families (natural, digits, medical). "
         "Every image passes through a fixed pipeline: resize, center-crop to 224x224, convert to a "
         "tensor, then normalise.",
         [
            "Why are the 16 datasets grouped into 'natural', 'digits', and 'medical' families?",
            "Several medical datasets are only 28x28 pixels but CLIP needs 224x224. What does the notebook say this upscaling does to the images, and why might it matter later?",
            "What does the 'normalise' step do to the pixel numbers, and why do models train more reliably on centred, similarly-scaled inputs?",
         ]),
        ("Part 4 & Part 5 - Baseline and how forgetting is measured",
         "Before any fine-tuning, CLIP's zero-shot accuracy is recorded on CIFAR-100, CIFAR-10, and "
         "STL-10. Forgetting is then defined as the drop from that baseline after each epoch of "
         "fine-tuning on a new dataset.",
         [
            "Write the definition of 'forgetting' used in this notebook as a short equation or sentence.",
            "Why does the notebook reset CLIP to its original weights before fine-tuning on each new dataset?",
            "Every dataset is given the same number of training images. Why is that fairness step necessary?",
         ]),
        ("Part 7 & Part 8 - Features and the leave-one-dataset-out design",
         "Two kinds of features are built: four static features (computed before training) and dynamic "
         "features measured during the first few epochs (the 'warm-up'). The data has 48 rows = 16 "
         "datasets x 3 learning rates.",
         [
            "Give an example of a static feature and a dynamic feature, and explain the difference in one sentence.",
            "The evaluation holds out an ENTIRE dataset (all of its learning rates) at once. Why would a plain random split of the 48 rows leak information and make the results look too good?",
         ]),
    ],
    "a2_closing": (
        "Turn your answers into the Methods section of our paper: describe the preprocessing pipeline, the "
        "definition of forgetting, the static vs. dynamic features, and the leave-one-dataset-out protocol, "
        "in language a reader can follow without the code."
    ),

    "a3_intro": (
        "Now for the payoff: does watching the first few epochs actually help forecast final forgetting? "
        "This assignment is about the forecasting scores and what they mean."
    ),
    "a3_metrics": [
        ("R-squared (R2)", "the fraction of the variation in forgetting the model explains. 1.0 is perfect, 0 means no better than always guessing the average, and negative is worse than that guess."),
        ("MAE", "mean absolute error - the average size of the forecast's mistake, in the same units as forgetting. Smaller is better."),
        ("Spearman correlation", "how well the forecast RANKS which datasets forget the most, ignoring exact values. Closer to +1 is better."),
        ("Static-only vs. static+warm-up", "the central comparison: forecasting with pre-training features alone vs. adding the first-five-epoch dynamics."),
    ],
    "a3_results": [
        "For the well-behaved models, adding the five-epoch warm-up raised R-squared from roughly 0.3-0.4 to about 0.91. In your own words, what does that jump tell us about the value of watching early training?",
        "MAE, Spearman, and R-squared measure different things. Why might you care about Spearman (ranking) specifically if you just want to know WHICH training runs to stop early?",
        "The neural network and LSTM performed poorly in both settings. The notebook calls this a 'useful lesson'. What is that lesson about model complexity and small datasets?",
        "The study warns that the effective sample size is 16 dataset groups, not 48 rows. Why does that make us cautious about the exact value 0.91?",
        "Write two or three sentences summarising the main result for the Results section of our paper.",
    ],
    "a3_limitations": (
        "The notebook is careful to bound its claim: one CLIP architecture, one optimizer, one run per "
        "configuration, and only 16 dataset groups. It also names a 'persistence baseline' (predict that "
        "final forgetting equals forgetting at epoch 5). Explain why that persistence baseline matters, "
        "and propose one next experiment that would make the forecasting result more convincing."
    ),
},

# ===========================================================================
# 3. CREDIT CONTEXT CURATION
# ===========================================================================
"credit_context": {
    "project": "Context Curation for Credit-Risk Prediction Under Temporal Shift",
    "folder": "credit_context_curation",
    "notebook": "credit_context_curation/credit_context_curation_explained.ipynb",

    "a1_intro": (
        "A new kind of model, the in-context tabular foundation model (like TabPFN), does not train on "
        "your data - it just READS a small table of labeled examples (a 'context') and answers. But it "
        "can only read about 1,000 rows, while we have far more historical loans than that. Our project "
        "asks: out of all past loans, which ~1,000 should we show the model - especially when the economy "
        "drifts over time? These papers explain the models, the standard baseline, and the drift problem."
    ),
    "a1_papers": [
        ("Hollmann, N. et al. (2023). TabPFN: A Transformer That Solves Small Tabular Classification "
         "Problems in a Second. ICLR.",
         "The foundation model at the center of our project. Focus on the idea of learning by reading a "
         "context, not by training.",
         [
            "What is 'in-context learning', and how is it different from ordinary model training?",
            "Why does TabPFN's limit on how many rows it can read create the exact problem our project studies?",
            "What kind of problems is TabPFN designed for - large or small tables? Why does that fit credit data with limited context?",
         ]),
        ("Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of KDD.",
         "XGBoost trained on all the data is our strong baseline - the number the curated contexts must "
         "try to match. Understand it at a high level.",
         [
            "At a high level, how does gradient-boosted trees like XGBoost make predictions?",
            "Why is 'XGBoost trained on the full historical pool' a fair and demanding baseline to compare our small contexts against?",
         ]),
        ("Gama, J. et al. (2014). A survey on concept drift adaptation. ACM Computing Surveys, 46(4).",
         "This survey explains 'concept drift' - when the pattern to be learned changes over time. That "
         "drift is the whole reason context choice is hard.",
         [
            "What is concept drift, in your own words?",
            "Why does splitting loans by TIME (train on the past, test on the future) test drift better than a random split?",
            "Give one reason a pattern learned from 2013-2016 loans might not hold for 2017-2018 loans.",
         ]),
        ("Guo, C. et al. (2017). On Calibration of Modern Neural Networks. Proceedings of ICML.",
         "Our project scores models on calibration (are the predicted probabilities honest?), not just "
         "ranking. This paper explains why calibration is a separate quality.",
         [
            "What does it mean for a model to be 'well calibrated'?",
            "Why can a model rank borrowers well (good AUC) yet still give dishonest probabilities (poor calibration)?",
            "Why would a lender care about calibration, not only about ranking who is riskiest?",
         ]),
    ],
    "a1_closing": (
        "Write two or three sentences describing our gap: foundation models can only read a small context, "
        "so the CHOICE of which historical rows to include - under economic drift - is an open and "
        "practical question. This becomes the core motivation in our Introduction."
    ),

    "a2_intro": (
        "This assignment is about the data, the honest time-split, and the six ways of choosing a context. "
        "You will look at how leakage is avoided and what each selection strategy actually picks."
    ),
    "a2_groups": [
        ("Section 1 & 2 - The data and the temporal split",
         "The data is real Lending Club loans, each labeled 'fully paid' (0) or 'charged off / default' "
         "(1). Loans are split by time: POOL = 2013-2016 (history the model may learn from) and "
         "TEST = 2017-2018 (the future it must predict).",
         [
            "Name three features a lender looks at (e.g., loan amount, interest rate, FICO score) and say what each means.",
            "Why is splitting by time stricter - and more honest - than shuffling all the loans together and splitting randomly?",
            "The notebook shows defaults are rare (class imbalance). Why would a model that always predicts 'paid' look accurate but be useless?",
         ]),
        ("Section 3 - Preparing the data without cheating",
         "Missing values are filled with the median and features are reshaped with a quantile transformer. "
         "Crucially, both are fit on the POOL only and merely applied to the TEST.",
         [
            "Why is it 'cheating' (leakage) to compute the median or the scaler using the 2017-2018 test loans?",
            "State the rule of thumb the notebook gives for avoiding leakage, in your own words.",
         ]),
        ("Section 5 - Six ways to choose the context",
         "Six strategies pick which ~1,000 past loans to show the model: random, most_recent, "
         "class_balanced, economically_similar, high_confidence, and diverse. The PCA 'map' shows each "
         "strategy selecting a different region of the data.",
         [
            "Pick any three strategies and explain, in one sentence each, what bet about 'useful examples' each one is making.",
            "The 'high_confidence' strategy keeps the loans a quick model was LEAST sure about. The notebook warns this can flip the credit-score-vs-default relationship to the wrong sign. Why is selecting an unusual slice of the data dangerous?",
         ]),
    ],
    "a2_closing": (
        "Convert your answers into the Data and Methods section of our paper: the loan data, the temporal "
        "pool/test split, the leakage-safe preprocessing, and the six context-selection strategies."
    ),

    "a3_intro": (
        "Now we compare the strategies and models. The headline is about data efficiency - whether a small, "
        "well-chosen 1,024-row context can rival a model trained on all the data - and about one strategy "
        "that backfires badly."
    ),
    "a3_metrics": [
        ("AUC (ROC-AUC)", "the chance the model scores a real defaulter as riskier than a real payer. 0.5 is guessing, 1.0 is perfect. The headline number because it is not fooled by rare defaults."),
        ("AP (Average Precision)", "focuses on how well the model finds the rare defaulters specifically."),
        ("Brier score", "measures whether predicted probabilities are honest (close to reality). Lower is better."),
        ("ECE (Expected Calibration Error)", "another honesty check: if the model says '20% chance', about 20% of those loans should actually default. Lower is better."),
    ],
    "a3_results": [
        "Across the results, which context strategies tend to land near the top, and which one repeatedly sinks toward or below the 0.5 guessing line?",
        "Same models, same amount of data - only the SELECTION of rows changed - yet scores differ a lot. What does that tell us about the importance of context curation?",
        "The 'high_confidence' trap hurt essentially every model (foundation and ordinary). Why is it important that the failure is shared across model types, not just one model?",
        "TabPFN and TabFM (two independent foundation models) showed similar behavior. Why does agreement between two independently built models make the finding more trustworthy?",
        "Write two or three sentences describing the data-efficiency result (best 1,024-row context vs. full-pool XGBoost) for our Results section, being careful not to overclaim.",
    ],
    "a3_limitations": (
        "The notebook lists honest limits: 'best-of-six' selection can look optimistic unless chosen "
        "independently; 'economically_similar' peeks at aggregate target-period information; only a few "
        "datasets/splits were used; and AUC says nothing about fairness or lending policy. Choose the "
        "limitation you find most important, explain it, and suggest one next step (the notebook hints at "
        "pre-registering one strategy and testing it once on a later, untouched period)."
    ),
},

# ===========================================================================
# 4. EEG CONTROLS AUDIT
# ===========================================================================
"eeg_controls": {
    "project": "Calibrating Perturbation Controls for EEG Decoder Audits",
    "folder": "eeg-controls-audit",
    "notebook": "eeg-controls-audit/EEG_controls_explained_for_students.ipynb",

    "a1_intro": (
        "When scientists claim a brain-reading AI 'really uses' a specific brain rhythm or region, they "
        "often prove it by damaging that part of the input (a 'perturbation control') and watching the "
        "accuracy drop. Our project asks a sharper question: can we trust those controls? We show they can "
        "'cry wolf', and whether they are reliable depends on the decoder being tested. These papers give "
        "us the background on the brain signal, the classic decoders, and why input-damage tests can mislead."
    ),
    "a1_papers": [
        ("Pfurtscheller, G. & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and "
         "desynchronization: basic principles. Clinical Neurophysiology, 110(11).",
         "Explains the mu/beta rhythm changes over motor cortex that motor-imagery decoders read. This is "
         "the 'signal' our whole project revolves around.",
         [
            "What is 'event-related desynchronization', and what happens to the mu rhythm when someone imagines moving a hand?",
            "The signal is a left-vs-right imbalance over motor cortex. Why does imagining the RIGHT hand change activity mostly on the LEFT side of the head?",
            "Why is knowing where the true signal lives important before we test whether a control is trustworthy?",
         ]),
        ("Ramoser, H., Muller-Gerking, J. & Pfurtscheller, G. (2000). Optimal spatial filtering of "
         "single trial EEG during imagined hand movement (CSP). IEEE Trans. Rehab. Eng., 8(4).",
         "Introduces Common Spatial Patterns (CSP), one of the classic decoders our study audits.",
         [
            "In plain words, what is a spatial filter trying to do with the many EEG channels?",
            "Why is a method that combines channels (like CSP) harder to interpret than one that looks at a single channel?",
         ]),
        ("Lawhern, V. J. et al. (2018). EEGNet: A compact convolutional neural network for EEG-based "
         "brain-computer interfaces. Journal of Neural Engineering, 15(5).",
         "EEGNet is the neural-network decoder our study finds to be the 'specific' one - the model whose "
         "controls behave best. Understand it at a high level.",
         [
            "What is EEGNet designed to do, and why is a compact network attractive for EEG?",
            "Our study finds the neural network is more 'specific' than simple band-power models. Why might a model that learns which features to use behave differently under perturbation controls?",
         ]),
        ("Adebayo, J. et al. (2018). Sanity Checks for Saliency Maps. NeurIPS.",
         "From image models, but the lesson is identical to ours: popular explanation methods can look "
         "convincing while failing basic reliability tests.",
         [
            "What surprising failure did the authors find in some popular saliency (explanation) methods?",
            "How is their message - 'test your explanation method before trusting it' - the same as our project's message about perturbation controls?",
            "Why is it dangerous to trust an explanation just because it looks intuitive?",
         ]),
    ],
    "a1_closing": (
        "Write two or three sentences stating our gap: perturbation controls are widely used as evidence, "
        "but their sensitivity and false-alarm behavior are rarely measured for the specific decoder being "
        "audited. Our project treats the control itself as an instrument that must be calibrated. This "
        "becomes our Introduction's thesis."
    ),

    "a2_intro": (
        "This assignment is about the method: the simulator with known ground truth, the three decoders, and "
        "the controls. The key trick is that we BUILD the data so we always know the true answer and can "
        "grade the controls."
    ),
    "a2_groups": [
        ("Sections 5-8 - The simulator and the planted signal",
         "Each fake trial is background 'pink noise' plus a planted 10 Hz mu-rhythm burst added only to the "
         "motor channels on the side opposite the imagined hand. A 'strength' knob controls the signal; "
         "strength = 0 means no signal at all. Figure 2 shows raw channels (the signal is invisible by eye).",
         [
            "Why does the project use a SIMULATOR instead of only real brains? (Hint: what can we know in the simulator that we can never know in a real brain?)",
            "What is the planted ground-truth signal, and which channels carry it?",
            "Why does the study include a 'strength = 0' condition where nothing is planted?",
         ]),
        ("Sections 9-10 - The three decoders and the sanity check",
         "Three decoders are built: a 'Focused' linear model (looks only at motor mu), a 'Broad' linear "
         "model (looks at every channel in mu and beta), and a 'Tree' model (a random forest that sees the "
         "SAME many features as Broad). The sanity check confirms they score ~50% when there is no signal.",
         [
            "The Broad model and the Tree model see the EXACT same features. Why is comparing them the whole point of the study?",
            "In the sanity check, why must every decoder score about 50% when strength = 0? What would it mean if one scored high there?",
         ]),
        ("Sections 11-12 - The controls and how they are graded",
         "Controls damage the input: remove_mu and remove_motor should hurt (signal lives there); "
         "remove_beta and remove_occipital should NOT hurt (no signal there). A control 'fires' if it drops "
         "accuracy by more than 10 points. Figure 8 is the main result.",
         [
            "Define a 'miss' and a 'false alarm' for a control in your own words.",
            "Why are the remove_beta and remove_occipital controls the important test of a control's trustworthiness?",
            "In Figure 8, the Broad model 'lights up red' on the right panel while the Tree model stays clean. What does that red mean, and why is it bad?",
         ]),
    ],
    "a2_closing": (
        "Your answers form the Methods section of our paper: the simulator and planted ground truth, the "
        "three decoder families, the controls, and the fire/miss/false-alarm definitions. Part II of the "
        "notebook scales this same idea to real EEG (PhysioNet), which you can mention as the full study."
    ),

    "a3_intro": (
        "Now the results: which controls are trustworthy, for which decoders, and whether the pattern holds "
        "on real human brains. The core finding is conditional - reliability depends on the decoder."
    ),
    "a3_metrics": [
        ("Decoder accuracy", "how often the decoder guesses left vs. right correctly. It is a PREREQUISITE (the decoder must actually work) - not the thing we are auditing."),
        ("Catch rate (sensitivity)", "when the target signal really IS present, how often the control fires. Higher is better."),
        ("False-alarm rate", "when the target signal is ABSENT, how often the control fires anyway. Lower is better. This is where untrustworthy controls are exposed."),
        ("'Fires' threshold", "a control counts as firing if it drops accuracy by more than 10 points. Question 3 below asks why this threshold is tricky."),
    ],
    "a3_results": [
        "All three decoders caught the real signal (removing mu or silencing motor channels hurt every one). Why is that a necessary first result before we can say anything about the controls?",
        "The Broad (linear) decoder 'cried wolf' - it also lost accuracy when beta or occipital channels were removed, where there is no signal. Why does a linear model that weighs every feature false-alarm, while the Tree model that sees the same features stays calm?",
        "In the real-data section, band-power caught the true occipital-alpha signal but false-alarmed on 3 of 4 unrelated controls. Why does this 'sim-to-real' agreement strengthen the whole project's argument?",
        "State the project's one-sentence punchline (whether you can trust a control depends on which decoder you are testing) in your own words, ready for the Discussion.",
        "Write two or three sentences comparing catch rate vs. false-alarm rate across the decoders for our Results section.",
    ],
    "a3_limitations": (
        "The notebook is unusually honest: the simulator is phenomenological (not a real cortex), each "
        "condition uses one data draw (no error bars yet), only older two-class motor-imagery datasets are "
        "used, and real-brain confirmation so far covers only the alpha task. Pick the limitation you think "
        "is most important and describe the specific next experiment that would address it (the notebook "
        "suggests a 'known-shortcut injection' test - explain why that would be convincing)."
    ),
},

# ===========================================================================
# 5. EXOPLANET MODEL COMPARISON
# ===========================================================================
"exoplanet": {
    "project": "Accuracy-Interpretability Trade-offs in Exoplanet Transit Modeling",
    "folder": "exoplanet-model-comparison",
    "notebook": "exoplanet-model-comparison/Exoplanet_Model_Comparison.ipynb",

    "a1_intro": (
        "When a planet passes in front of its star, the star dims slightly - a 'transit'. Our project "
        "compares many machine-learning models on two jobs: detecting transits and measuring planet size, "
        "and it asks how much ACCURACY you trade away to get an INTERPRETABLE model you can actually read. "
        "It also tests a physics-informed model that learns planet sizes with no size labels. These papers "
        "give us the background on transits, deep learning for detection, the interpretable architecture, "
        "and why interpretability matters."
    ),
    "a1_papers": [
        ("Borucki, W. J. et al. (2010). Kepler Planet-Detection Mission: Introduction and First "
         "Results. Science, 327(5968).",
         "Introduces the Kepler mission and the transit method - the source of the real data at the end of "
         "our notebook.",
         [
            "What is the transit method, and what causes the small repeating dip in a star's brightness?",
            "What does the DEPTH of the dip tell us about the planet? What does the WIDTH tell us?",
            "Why does staring at the same patch of sky for years (as Kepler did) help detect small planets?",
         ]),
        ("Shallue, C. J. & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning. The "
         "Astronomical Journal, 155(2).",
         "A landmark paper using a neural network (CNN) to detect transits in Kepler data - the same "
         "detection task we study.",
         [
            "What machine-learning model did the authors use to detect transits, and how did it do compared to older methods?",
            "Why is telling a real transit apart from ordinary stellar wobble a hard problem?",
            "Their model is powerful but a 'black box'. What does that make hard, and how does it motivate our interpretability question?",
         ]),
        ("Liu, Z. et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.",
         "The interpretable architecture our project highlights. A KAN learns a readable curve for each "
         "input instead of hidden fixed weights.",
         [
            "How is a KAN different from a normal neural network (what does it learn for each input)?",
            "Why does that design make a KAN 'intrinsically interpretable' - i.e., readable after training with no approximation?",
            "Why is a readable model especially valuable in science, where we want to know WHY a prediction was made?",
         ]),
        ("Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions "
         "and use interpretable models instead. Nature Machine Intelligence, 1.",
         "Argues that for important decisions we should prefer models that are interpretable by design "
         "rather than explained after the fact. This frames our accuracy-vs-interpretability trade-off.",
         [
            "What is the difference between an 'interpretable' model and an 'explained' black box, as the author sees it?",
            "Does the author believe you always lose accuracy by choosing an interpretable model? What is her view?",
            "How does this argument connect to our project's Pareto trade-off between accuracy and interpretability?",
         ]),
    ],
    "a1_closing": (
        "Write two or three sentences describing our gap: exoplanet pipelines increasingly use powerful "
        "black-box models, but it is rarely made explicit how much accuracy is traded for a model a "
        "scientist can actually read - and whether a physics-informed model can get both. This becomes our "
        "Introduction's motivation."
    ),

    "a2_intro": (
        "This assignment is about how the data is built and how the models are set up. You will look at the "
        "simulated transits with known answers, the two different preprocessing choices for detection vs. "
        "sizing, and the physics-informed model."
    ),
    "a2_groups": [
        ("Section 5 - Simulating transits with known answers",
         "Because simulated data has exact known answers, we can measure precisely how well each model "
         "recovers the truth. The transit shape comes from a standard physics model (Mandel & Agol). "
         "Positives are real transits; negatives include flat noise AND slow stellar waves.",
         [
            "Why start with SIMULATED transits instead of real Kepler data?",
            "Three physical quantities (Rp/R*, impact parameter b, and a/R*) shape the dip. Pick two and describe how each changes the shape of the dip.",
            "Half the negative examples are gentle stellar 'waves', not flat noise. Why force the detector to reject wavy non-transits, not just flat ones?",
         ]),
        ("Sections 5.5 & 9 - Two different preprocessing choices",
         "For DETECTION, each light curve is standardized (subtract the median and divide by its spread) - "
         "which discards absolute depth. For CHARACTERIZATION (size), we subtract the median but do NOT "
         "rescale, because depth is exactly the signal we now need.",
         [
            "Why is it fine to throw away absolute depth for detection, but harmful to throw it away for measuring planet size?",
            "In one sentence, what does 'standardizing' a light curve do to its numbers, and why does it put examples on comparable footing?",
         ]),
        ("Sections 6.1 & 9.1 - The models and the physics-informed KAN",
         "Seven models are compared (linear, random forest, XGBoost, MLP, CNN, LSTM, KAN). The "
         "physics-informed KAN guesses physical parameters, feeds them through a differentiable transit "
         "model to REBUILD the light curve, and is trained only to match the input - never shown the true "
         "planet sizes.",
         [
            "In plain words, how does the physics-informed KAN learn planet sizes WITHOUT ever seeing a true size label?",
            "Why must the transit-physics model be made 'differentiable' for this to work?",
            "Why is a fair comparison only possible because every model is trained and graded through one common interface?",
         ]),
    ],
    "a2_closing": (
        "Convert your answers into the Data and Methods section of our paper: how transits were simulated, "
        "the two preprocessing choices, the seven models, and the physics-informed reconstruction approach."
    ),

    "a3_intro": (
        "Now the results: which models detect and size planets best, what is traded for interpretability, "
        "and how the physics-informed model does - including on real Kepler planets."
    ),
    "a3_metrics": [
        ("ROC-AUC (detection)", "how well the model ranks a real transit above a non-transit. 0.5 is guessing, 1.0 perfect. The detection headline."),
        ("MAE (characterization)", "the average error in the predicted planet size Rp/R*. Lower is better. The sizing headline."),
        ("Additive-surrogate fidelity (interpretability)", "how closely a simple SUM of per-time-point functions reproduces the model's output. High means the model behaves readably/additively. This is ONE specific, explicit measure of interpretability, not all of it."),
        ("Pareto frontier", "the set of models where no other model beats them on BOTH accuracy and interpretability at once - the honest way to show a trade-off."),
    ],
    "a3_results": [
        "On the detection task, do the most accurate models tend to be the most interpretable ones, or is there a trade-off? Describe what the Pareto plot shows.",
        "The physics-informed KAN used NO size labels. How did it do on the characterization (size) task relative to the black-box models, and why is that result notable?",
        "The notebook says the additive score measures only ONE dimension of interpretability. Why is it important to state that, rather than calling it 'the' interpretability score?",
        "On the real Kepler planets, the model was trained only on simulations and never shown a real light curve. Why is the full 12-planet table the honest result, while the 'best 75%' subset is only exploratory?",
        "Write two or three sentences summarising the accuracy-interpretability trade-off for our Results section.",
    ],
    "a3_limitations": (
        "The notebook names real limits: synthetic train and test data share the same generator; the "
        "physics-informed model has a 'matched-simulation' advantage; the real sample is only 12 planets; "
        "and the filtered subset is post-hoc. Pick the limitation you think most threatens the conclusions, "
        "explain why, and propose one next step (the notebook suggests testing on mismatched noise and on "
        "TESS data)."
    ),
},

# ===========================================================================
# 6. RADDOSE PHYTODOSIMETRY
# ===========================================================================
"raddose": {
    "project": "RadAlert-AT: Cross-Study Radiation-Exposure Triage from Plant Gene Expression",
    "folder": "raddose-phytodosimetry",
    "notebook": "raddose-phytodosimetry/RadDose_Full_Walkthrough.ipynb",

    "a1_intro": (
        "Radiation damages DNA, and plant cells respond by switching certain genes on or off. Our project "
        "asks whether we can read a plant's gene-expression profile and tell if it was exposed to radiation "
        "at all - a screening or 'triage' question - even when the test sample comes from a completely "
        "unseen study. We deliberately switched from predicting exact dose (which the data cannot support) "
        "to predicting exposure yes/no. These papers give us the biology of radiation-responsive genes and "
        "the data-analysis ideas we rely on."
    ),
    "a1_papers": [
        ("Ryu, T. H. et al. (2018). Transcriptome-based biological dosimetry of gamma radiation in "
         "Arabidopsis using DNA damage response genes. Journal of Environmental Radioactivity, 181.",
         "This is the source of the seven-gene DNA-damage-response (DDR-7) panel our notebook uses as a "
         "baseline. It directly motivates our gene-expression approach.",
         [
            "Which biological process do the seven DDR genes report on, and why would radiation change their expression?",
            "The authors relate gene-expression change to radiation DOSE. Our project instead predicts EXPOSURE (yes/no). Why might exposure be an easier and more reliable target than exact dose?",
            "Why is a small, literature-based gene panel a sensible baseline to compare a data-driven model against?",
         ]),
        ("Lee, Y. et al. (2021). Application of Gamma Ray-Responsive Genes for Transcriptome-Based "
         "Phytodosimetry in Rice. Plants, 10(5), 968.",
         "Shows the same idea (gene expression as a radiation readout) in a different plant. Useful for "
         "asking whether radiation-response signals generalize across species and studies.",
         [
            "What is 'phytodosimetry' - using plants to measure radiation?",
            "This paper works in rice, our project in Arabidopsis. Why does showing an effect in more than one species (or study) make a biomarker more trustworthy?",
            "How does this connect to our project's emphasis on testing across completely separate studies?",
         ]),
        ("Sng, B. J. R. et al. (2021). Detection of Genes in Arabidopsis thaliana L. Responding to "
         "DNA Damage from Radiation and Other Stressors in Spaceflight. Genes, 12(6), 938.",
         "Studies radiation-responsive genes alongside OTHER stressors. This is exactly the specificity "
         "worry our project raises: do these genes respond only to radiation?",
         [
            "Why is it a problem if a 'radiation biomarker' gene also responds to drought, heat, or other stress?",
            "Our project says its genes are 'biologically plausible' but not proven causal. How does this paper's focus on multiple stressors support that caution?",
            "What experiment would you need to prove a gene responds specifically to radiation and not to general stress?",
         ]),
        ("Leek, J. T. et al. (2010). Tackling the widespread and critical impact of batch effects in "
         "high-throughput data. Nature Reviews Genetics, 11(10).",
         "Explains 'batch effects' - lab-to-lab technical differences that can masquerade as real biology. "
         "This is why our project standardizes each study and tests across studies.",
         [
            "What is a 'batch effect', and how could it fool a model into 'detecting radiation' that is really just a lab difference?",
            "Why does testing on a completely held-out study (rather than random samples) guard against being fooled by batch effects?",
            "Why is per-study standardization helpful, and what does the notebook note is still imperfect about it?",
         ]),
    ],
    "a1_closing": (
        "Write two or three sentences stating our gap: gene-expression radiation signals exist, but showing "
        "they transfer to an entirely unseen study - without leaking lab-specific quirks - is the harder, "
        "more useful test. This becomes our Introduction's motivation."
    ),

    "a2_intro": (
        "This assignment is about the data, why we changed the prediction target, and how the honest "
        "cross-study test is built. The single most important method here is leave-one-study-out validation."
    ),
    "a2_groups": [
        ("Sections 2-3 - The data and why the target changed",
         "The data is 158 Arabidopsis samples from six NASA studies, each with 4,002 gene features and a "
         "dose label. The original target was exact dose, but 138 of 158 samples sit at just 0 or 100 Gy, "
         "with only 20 samples at intermediate doses.",
         [
            "Why does having 138 of 158 samples at only 0 or 100 Gy make predicting the EXACT dose unreliable?",
            "What is the new prediction target (exposure), and why does it use every study without pretending the data form a smooth dose curve?",
            "The always-exposed baseline already gets 63% accuracy but 0% specificity. What does that tell you about trusting accuracy alone here?",
         ]),
        ("Sections 5-6 - Preprocessing and the honest test",
         "The supplied matrix is already log-transformed and standardized per study, keeping the 4,002 most "
         "variable genes. The evaluation is leave-one-study-out (LOSO): hold out ALL samples from one study, "
         "fit everything on the other five, and repeat.",
         [
            "Why would randomly splitting samples (mixing a study into both train and test) overstate how well the model transfers to a NEW lab?",
            "Explain leave-one-study-out validation in one or two sentences.",
            "The 'leakage checklist' says the F-test gene selector is fitted INSIDE each training fold. Why would selecting genes before cross-validation leak the held-out study and inflate the score?",
         ]),
        ("Section 7 - The model ladder",
         "Six models of increasing sophistication are compared: an always-exposed baseline, a DDR-7 logistic "
         "model, a top-50 logistic model with fold-local gene selection, an RBF-SVM, an ExtraTrees ensemble, "
         "and a stability ensemble that averages ExtraTrees over the top 50 and top 100 genes.",
         [
            "Why start with a deliberately dumb 'always exposed' baseline before the real models?",
            "The stability ensemble averages over two different gene-count choices (50 and 100). Why does that make the result less dependent on one arbitrary decision?",
         ]),
    ],
    "a2_closing": (
        "Turn your answers into the Data and Methods section of our paper: the six-study Arabidopsis data, "
        "the shift from dose regression to exposure classification, the per-study preprocessing, the LOSO "
        "protocol with fold-local feature selection, and the model ladder."
    ),

    "a3_intro": (
        "Now the results: how well the exposure classifier works across unseen studies, why we report "
        "several metrics instead of one, and where the model still fails."
    ),
    "a3_metrics": [
        ("Accuracy", "the fraction of samples classified correctly. Misleading here because exposed samples are the majority (the always-exposed baseline already gets 63%)."),
        ("Balanced accuracy", "the average of how well controls AND exposed samples are handled. 0.50 is chance. A primary metric because it is not fooled by the class imbalance."),
        ("ROC-AUC", "how well the exposure score ranks exposed samples above controls. 0.50 is chance, higher is better."),
        ("Sensitivity vs. Specificity", "sensitivity = fraction of exposed samples caught; specificity = fraction of controls correctly left negative. A screening tool wants high sensitivity but must watch specificity (false alarms)."),
        ("MCC", "Matthews correlation coefficient - a single balanced summary of the whole confusion matrix, honest even when classes are imbalanced."),
    ],
    "a3_results": [
        "The stability ensemble reached about 84% accuracy, 82% balanced accuracy, and 0.88 ROC-AUC while holding out a whole study each time. Why is it important that accuracy and balanced accuracy are close, rather than accuracy being much higher?",
        "Sensitivity was about 89% and specificity about 76%. In plain words, what kind of mistake is the model making more often - missing exposed plants, or false-alarming on controls? Why does that matter for a screening tool?",
        "Why does the notebook report accuracy, balanced accuracy, ROC-AUC, sensitivity, specificity, AND MCC together, instead of just one number? Give a concrete example where two of them disagree.",
        "The intermediate doses (10, 40, 80 Gy) have very few samples. The model still called many of them 'exposed'. Why is that encouraging, but why can't we make strong dose-response claims from it?",
        "Write two or three sentences summarising the exposure-triage result for our Results section, being careful to call it screening, not precise dosimetry.",
    ],
    "a3_limitations": (
        "The notebook is clear that only SIX independent studies exist, so the study count - not the sample "
        "count - governs the uncertainty, and it says an independent seventh study is worth more than a "
        "fancier model. Explain why six studies is the governing limitation, and describe the exact next "
        "experiment the notebook recommends (reserving a genuinely untouched study with multiple doses and "
        "non-radiation stress controls, and freezing the model before opening it)."
    ),
},

}
