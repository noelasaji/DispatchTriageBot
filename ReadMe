#  Roadside Assistance Dispatch Classifier

> A conversational triage chatbot that collects structured information from stranded motorists and dispatches the appropriate roadside service, powered by a Random Forest classifier, a deterministic rule engine, and a SQLite-backed agent queue, served via a Gradio interface.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Service Classes](#service-classes)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Evaluation](#evaluation)
- [Running the Tests](#running-the-tests)
- [Database Schema](#database-schema)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

This project implements a two-tier dispatch decision pipeline for roadside assistance call triage:

1. **Rule engine** — a deterministic finite state machine (`determine_next_step`, `get_rule_recommendation`) that maps structured Yes/No responses to a service type.
2. **ML classifier** — a `scikit-learn` Random Forest trained on synthetic call data that acts as a confidence-weighted corroborator of the rule engine.

A **Gradio** front-end exposes two interfaces:
- **Customer Chat** — a guided conversational UI collecting issue type, safety status, and location.
- **Agent / Admin Queue** — a case management view with ML prediction, confidence scores, and CSV export.

All session data, answers, events, outcomes, and agent queue entries are persisted to a local **SQLite** database.

---

## Architecture

```
Customer Chat (Gradio)
        │
        ▼
 State Machine ──► Workflow Q&A ──► Answers Dict
        │
        ├──► Rule Engine ──────────────────────────┐
        │    (get_rule_recommendation)              │
        │                                          ▼
        └──► ML Pipeline ──────────────► Prediction + Confidence
             (RandomForestClassifier)              │
                                                   ▼
                                        SQLite Agent Queue
                                                   │
                                                   ▼
                                        Agent / Admin Tab (Gradio)
```

**Inference flow per completed session:**
1. Rule engine produces a deterministic recommendation.
2. ML pipeline produces a probabilistic prediction with per-class confidence scores.
3. Both are stored in `agent_queue` alongside a full debug JSON payload.
4. Priority is set to `high` if `safe_location == "No"`; otherwise `normal`.

---

## Service Classes

| Class | Condition |
|---|---|
| `tire_change` | Flat tyre; spare available |
| `mobile_tire_support` | Flat tyre; no spare; vehicle drivable |
| `towing` | Flat tyre; no spare; vehicle **not** drivable — or direct tow request |
| `jump_start` | Battery fault; dashboard indicator lights present |
| `diagnostic_or_tow` | Battery fault; no dashboard activity |
| `locksmith` | Lockout; keys confirmed inside vehicle |
| `human_escalation` | Unsafe location; ambiguous lockout; unclassified fault |

---


## Getting Started

### Prerequisites

- Python 3.9+
- Google Colab (recommended) or a local Jupyter environment

### Installation

```bash
pip install gradio pandas scikit-learn matplotlib seaborn joblib
```

Or in Colab (Cell 1 of the notebook):

```python
!pip install -q gradio pandas scikit-learn matplotlib seaborn joblib
```

### Running the Notebook

1. Open `Roadside_Dispatch_Chatbot.ipynb` in Google Colab or JupyterLab.
2. Run all cells top-to-bottom (**Runtime → Run all** in Colab).
3. The final cell launches the Gradio app with a public share link:

```
* Running on public URL: https://xxxxxxxx.gradio.live
```

4. Open the link in your browser. Use the **Customer Chat** tab to simulate a call and the **Agent/Admin** tab to view the dispatch queue.

---

## Project Structure

```
Roadside_Dispatch_Chatbot.ipynb   ← Main notebook
roadside_assistance.db    ← SQLite database (auto-created on first run)
roadside_model.joblib     ← Serialised trained model (auto-created on first run)
confusion_matrix.png      ← Confusion matrix plot (auto-created on first run)
```

> **Note:** `.db` and `.joblib` files are generated at runtime and should be added to `.gitignore` for production repositories.

### Recommended `.gitignore`

```gitignore
*.db
*.joblib
*.png
__pycache__/
.ipynb_checkpoints/
```

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | `RandomForestClassifier` (scikit-learn) |
| Number of trees | 200 (`n_estimators=200`) |
| Splitting criterion | Gini impurity |
| Class weighting | `class_weight="balanced"` |
| Parallelism | `n_jobs=-1` (all available CPU cores) |
| Reproducibility | `random_state=42` |
| Preprocessing | `OneHotEncoder(handle_unknown="ignore")` via `ColumnTransformer` |

### Features

All seven input features are categorical:

| Feature | Values |
|---|---|
| `issue_type` | flat\_tire, battery, lockout, towing, other |
| `safe_location` | Yes, No |
| `vehicle_drivable` | Yes, No |
| `all_tires_ok` | Yes, No |
| `has_spare` | Yes, No |
| `dashboard_lights_on` | Yes, No |
| `locked_keys_inside` | Yes, No |

### Training Data

Training data is generated synthetically by `generate_fake_case()` using realistic class priors:

- 2,000 samples generated per training run
- `safe_location` sampled at 10 % unsafe 
- Labels assigned deterministically from feature values, mirroring the rule engine
- 80/20 stratified train/test split (`stratify=y`)

> ⚠️ **Limitation:** The model is trained on synthetic rule-derived data. It will not generalise to real noisy call data without retraining on operator-labelled transcripts. See [Known Limitations](#known-limitations).

---

## Evaluation

### Reported Metrics

After training, the notebook prints a full `classification_report` and logs:

```
=== Run Log ===
  timestamp:        2025-xx-xxTxx:xx:xx
  n_train:          1600
  n_test:           400
  n_estimators:     200
  class_weight:     balanced
  accuracy:         1.0
  train_time_sec:   x.xxx
  cv_f1_macro_mean: x.xxxx
  cv_f1_macro_std:  x.xxxx
```

### Cross-Validation

5-fold `StratifiedKFold` cross-validation is run on the full dataset with `scoring="f1_macro"`:

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ml_model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
# Reports: F1-macro = μ ± σ
```

> On synthetic data, accuracy and F1-macro will be near 1.0 by construction. These figures will not hold on real call data — treat them as a baseline sanity check only.

### Confusion Matrix

A confusion matrix is generated automatically and saved to `confusion_matrix.png`:

```python
cm = confusion_matrix(y_test, preds, labels=ml_model.classes_)
ConfusionMatrixDisplay(cm, display_labels=ml_model.classes_).plot(...)
```

## Running the Tests

Four unit/integration tests are included directly in the notebook (Cell 19). Run that cell after training to verify all invariants:

```
PASS test_generate_fake_case_keys
PASS test_flat_tire_no_spare_not_drivable_is_towing
PASS test_unsafe_location_is_human_escalation
PASS test_model_predict_returns_known_class

✅ All tests passed!
```

| Test | What it checks |
|---|---|
| `test_generate_fake_case_keys` | All required keys present in every generated sample |
| `test_flat_tire_no_spare_not_drivable_is_towing` | Regression guard for D-1 — verifies the corrected label branch |
| `test_unsafe_location_is_human_escalation` | Safety escalation invariant holds in the trained model |
| `test_model_predict_returns_known_class` | All predictions belong to `ml_model.classes_` |

---

## Database Schema

The SQLite database (`roadside_assistance.db`) is auto-initialised on first run with five tables:

```sql
sessions    (session_id, started_at, current_step, complete)
answers     (id, session_id, field_name, field_value, created_at)
events      (id, session_id, speaker, message, field_name, field_value, created_at)
outcomes    (id, session_id, recommendation, ml_prediction, ml_confidence, summary, created_at)
agent_queue (id, session_id, priority, status, summary, recommendation,
             ml_prediction, ml_confidence, debug_json, created_at)
```

All tables can be exported to CSV from the **Agent/Admin** tab in the Gradio UI.

---

## Known Limitations

- **Synthetic training data only.** The model learns a deterministic function of its own features. It has no exposure to noisy, contradictory, or out-of-distribution real call data. Performance on real traffic is unknown and likely to be lower.
- **No confidence threshold gate.** The model's argmax prediction is used regardless of confidence score. A production system should route to `human_escalation` when `max(P(y|x)) < τ` (recommended τ ≈ 0.65).
- **No model versioning.** `roadside_model.joblib` is overwritten on every run. Use MLflow Model Registry or DVC for production artefact management.
- **SQLite concurrency.** SQLite is not suitable for multi-user concurrent writes. Migrate to PostgreSQL for production deployment.
- **Gradio share links are ephemeral.** Public URLs from `share=True` expire after one week. Use `gradio deploy` to Hugging Face Spaces for persistent hosting.

---

## Roadmap

- [ ] Replace synthetic training data with operator-labelled call transcripts
- [ ] Add confidence threshold gate with configurable `τ`
- [ ] Cost-sensitive classification (asymmetric misclassification costs by dispatch type)
- [ ] MLflow experiment tracking integration
- [ ] PostgreSQL backend for multi-user production deployment
- [ ] REST API wrapper (`FastAPI`) for integration with telephony platforms
- [ ] Conformal prediction sets for calibrated uncertainty estimates

---

## License

Dependencies and their licences:

| Package | Licence |
|---|---|
| scikit-learn | BSD-3-Clause |
| pandas | BSD-3-Clause |
| Gradio | Apache 2.0 |
| matplotlib | PSF |
| joblib | BSD-3-Clause |

---


