# URL-only phishing classifier

Two models scored only from the URL string:

* **Baseline:** hashed character 3–5-grams of the whole URL (2^16 buckets, L2-normalised) + logistic regression.
* **Advanced:** a richer URL representation (host and path n-grams kept apart, word tokens, 22 structure
  features such as host depth, digits in host, IP host, entropy) and the winner of a small auto-ML sweep
  (`src/experiments.py`): logistic regression, LightGBM and XGBoost candidates, plus weighted blends,
  compared in 5-fold cross-validation grouped by registered domain. The selected model is 0.25 × logistic
  regression (C=16) + 0.75 × LightGBM (800 trees, 63 leaves).

Both are stored as ONNX in `models/`; the text-to-feature step runs in Python (`src/preprocess.py`).
`xgboost.onnx` is the best XGBoost candidate from the sweep, saved for comparison only; `predict()` uses
the selected advanced model. The report is [reports/REPORT.md](reports/REPORT.md).

## Layout

```
data/          train.csv, test.csv, external_test.csv (not committed; copy them here)
src/           preprocess, data (cleaning + audit), metrics, model, experiments, train, evaluate
models/        baseline.onnx, selected advanced component(s), and comparison-only xgboost.onnx
reports/       REPORT.md and the JSON files every number in it comes from
notebooks/     01 EDA, 02 preprocessing, 03 training, 04 threshold, 05 evaluation, 06 prediction
tests/         predict() contract tests
predict.py     from predict import predict
predictions.csv  url,probability for every row of data/test.csv (advanced model)
requirements.txt  pinned project dependencies, including xgboost==2.1.1
```

## Setup

Python 3.11 is required.

```bash
uv venv -p 3.11 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
# or: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Reproduce (from the repository root, about 15 minutes on 8 CPU cores)

```bash
cd src
python data.py          # data audit        -> reports/audit.json
python experiments.py   # auto-ML sweep     -> reports/experiments.json, reports/oof_scores.csv
python train.py         # fit + ONNX export -> models/*.onnx, reports/train_summary.json
python evaluate.py      # test + external   -> reports/evaluation.json, predictions.csv
cd .. && python -m pytest -q
```

Only `evaluate.py` reads the two evaluation files. Thresholds come from out-of-fold (OOF) training scores.

## Predict

```bash
python -c "from predict import predict; print(predict('https://www.PayPal-secure.login.tk/webscr?cmd=x'))"
```

The model loads on the first call (not on import), runs on CPU and needs no network.
Invalid input (empty, whitespace or control characters only, `None`, or anything that raises inside
feature extraction) returns `{"label": "clean", "probability": 0.5}`; other non-strings are converted
with `str()`.

`probability` is the model's score for a balanced, train-like population (about half phishing). It is not
calibrated to real traffic (about 1 in 1,000 phishing) and it is overconfident: the mean score on
`test.csv` is 0.07 for legitimate URLs and 0.93 for phishing, but 0.62 for external phishing. Use it to rank
URLs, not as a real-world probability.

## Reading the report

**Models**

- **Logistic regression (LR):** a linear model. Every feature gets a weight, and the weighted sum becomes a
  probability. It is fast and easy to inspect.
- **LightGBM / XGBoost:** gradient-boosted decision trees. Hundreds of small trees are added one after
  another, each correcting the trees before it, so they can learn feature combinations ("short host *and*
  no path").
- **Blend:** a weighted average of the probabilities of several models.
- **Character n-grams:** every run of 3–5 characters in the URL (`pay`, `payp`, …), hashed into a fixed
  number of columns, so no vocabulary has to be stored.

**Validation and metrics**

- **Grouped cross-validation (CV):** `train.csv` is split into 5 folds grouped by registered domain
  (e.g. `paypal.com`). Each model is trained on 4 folds and scored on the 5th. These **out-of-fold (OOF)**
  scores are the only thing used for model selection and thresholds.
- **Recall (TPR):** the share of phishing URLs that are flagged. 
- **FPR:** the share of legitimate URLs that are
  flagged. 
- **Precision:** the share of flagged URLs that are phishing.
- **ROC-AUC:** the chance that a random phishing URL scores higher than a random legitimate one (1.0 is
  perfect, 0.5 is a coin flip). It does not depend on a threshold.
- **Recall @ 1 % FPR:** the phishing caught if we accept flagging 1 in 100 legitimate URLs. This is the
  selection metric, because false positives are what hurt in deployment.
- **Prevalence π:** the share of phishing in real traffic, assumed to be 1 in 1,000.
- **95 % CI:** the range the true value plausibly lies in, given the sample size. AUC intervals resample whole
  domains, and rate intervals are exact binomial (Clopper–Pearson).

**Train/test overlap** (`reports/audit.json`), from strict to loose: exact string; the organisers'
normalisation (case, scheme, leading `www.`, trailing `/`); same host; same registered domain (eTLD+1,
counting private suffixes, so `x.blogspot.com` is its own site). The normalisation misses parameter order,
encodings and, most importantly, **campaigns**: one phishing kit on many URLs or hosts (e.g. 263 phishing rows
on `duilawyeryork.com`). Shared sites and campaigns, not identical URLs, are what make `test.csv` optimistic.

**Why the blend wins** (also in `notebooks/03_train_demo.ipynb`)

- LR adds up many small n-gram clues and ranks the clearest phishing very high. This is why it is strong at
  0.1 % FPR.
- LightGBM learns combinations of the structure features.
- At each model's own 1 %-FPR threshold, LR catches 250 phishing URLs that LightGBM misses, and LightGBM
  catches 355 that LR misses (2,442 are caught by both, 1,015 by neither, out of 4,062). Only 23 of their
  42 false alarms are shared.
- Their OOF scores have a rank correlation of 0.94: high, but not identical, so averaging lets each fix some
  of the other's errors.
- XGBoost correlates 0.98 with LightGBM, so it adds almost nothing and got weight 0.
- The blend is cheap at inference: two ONNX files, with the weights stored in their metadata.

**ONNX:** scikit-learn's n-gram step can't be converted exactly, so features are computed in Python and only
the classifiers are ONNX. They match the Python models to within 2e-7. XGBoost needs zeros passed as
"missing", which `model.Scorer` handles. `models/xgboost.onnx` scores test AUC 0.983 and external 0.879, with
recall 72 % / 42 % at its own 1 %-FPR threshold.
