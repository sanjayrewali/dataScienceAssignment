# Customer Churn Prediction — Telco
---
Links (fill these in after you push your own code & images)
Code Repository: https://github.com/sanjayrewali/DevOpsKubernetesAssignmentDemo
Screen recording video: https://drive.google.com/file/d/1kjZGo1xRQgGRHhSW5yusybuwhe-ROokI/view?usp=sharing
---

An end-to-end machine learning solution that predicts whether a telecom customer
is likely to **churn**, so the retention team can proactively engage at-risk
customers. The project covers the full workflow:

> Business Problem → Data → Preparation → EDA → Feature Engineering → Model →
> Evaluation → Interpretation → Saved Model → REST API

## Project Structure

```
data-science-assignment-code/
├── data/
│   └── Telco-Customer-Churn.csv      # IBM Telco Customer Churn dataset
├── notebook/
│   └── churn_analysis.ipynb          # Complete analysis & modelling
├── model/
│   └── churn_model.pkl               # Saved pipeline (feature eng + preprocessing + tree)
├── results/                          # Generated plots (EDA, confusion matrix, tree, etc.)
├── churn_utils.py                    # Shared preprocessing/feature-engineering (used by notebook & API)
├── app.py                            # Flask REST API (POST /predict)
├── sample_request.json               # Example request body
├── requirements.txt                  # Python dependencies
└── README.md
```

## Dataset

- **Source:** IBM Telco Customer Churn (7,043 customers, 21 columns)
- **Target:** `Churn` (Yes / No), ~26.5% positive (imbalanced)
- The dataset is included under `data/`. If you need to re-download it:
  ```bash
  curl -sL -o data/Telco-Customer-Churn.csv \
    https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
  ```

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Run the Analysis Notebook

```bash
jupyter notebook notebook/churn_analysis.ipynb
```

Run all cells top to bottom. The notebook performs data preparation, EDA (with
saved plots in `results/`), feature engineering, model development and
comparison, evaluation, interpretation, and finally **saves the trained pipeline**
to `model/churn_model.pkl` and writes `sample_request.json`.

## Run the REST API

```bash
python app.py
# Server starts at http://127.0.0.1:5000
```

### Endpoint: `POST /predict`

Accepts a single customer's attributes as JSON, applies the same preprocessing
used in training, and returns the churn prediction and probability. Invalid input
(missing fields, non-numeric numbers, or malformed JSON) returns HTTP `400` with
an explanatory message.

**Sample request**

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @sample_request.json
```

`sample_request.json`:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 18,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 96.05,
  "TotalCharges": "1740.7"
}
```

**Sample response**

```json
{
  "prediction": "Yes",
  "churn_probability": 0.82
}
```

Additional helper endpoints:

- `GET /` — service info and model metrics
- `GET /health` — health check

## Key Design Decisions

- **No data leakage:** encoding and imputation live inside a scikit-learn
  `Pipeline` that is fit only on the 70% training split; feature engineering is
  stateless. The persisted pipeline accepts **raw** customer JSON end-to-end, so
  the exact same transformations are applied at training and serving time.
- **Reproducibility:** `random_state = 42`, 70:30 stratified split.
- **Model:** Decision Tree Classifier. Two configurations are compared (a shallow
  baseline vs. a `GridSearchCV`-tuned, class-weighted tree); the class-weighted
  tuned model is selected because it maximizes **recall** on the churn class.
- **Precision vs. Recall:** we prioritize **Recall** — missing a real churner
  (lost lifetime value) is far costlier than a false alarm (a cheap retention
  offer).

## Feature Engineering

| Feature | How it is created | Why it helps |
|---|---|---|
| `tenure_group` | Buckets `tenure` into 0-12/13-24/25-48/49-60/61-72 months | Churn risk is highly concentrated in the first year. |
| `num_addon_services` | Count of the 6 add-on services set to "Yes" | Higher engagement lowers churn. |
| `avg_charges_per_month` | `TotalCharges / tenure` | Separates long-term high-value from new high-bill customers. |
| `monthly_to_total_ratio` | `MonthlyCharges / (TotalCharges + 1)` | Proxy for how "new" the spend is. |
