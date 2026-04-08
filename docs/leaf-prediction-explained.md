# Decision Tree Leaf Prediction — End-to-End Explanation

How the codebase goes from raw training data to a salary point estimate and
data-driven range at inference time.

---

## 1. Feature Engineering

Before the tree sees any data, every raw column is encoded into a compact set
of numeric features defined in `src/features/engineering.py`:

| Feature | Type | Range |
|---|---|---|
| `experience_level` | ordinal int | 0 (EN) → 3 (EX) |
| `employment_type` | ordinal int | 0 (FL) → 3 (FT) |
| `remote_ratio` | numeric | 0, 50, 100 |
| `company_size` | ordinal int | 0 (S) → 2 (L) |
| `work_year` | numeric | year integer |
| `job_family` | label int | 0 (other) → 5 (leadership) |
| `location_region` | label int | 0 (rest) → 3 (north america) |
| `is_us_company` | binary int | 0 or 1 |

These encodings are **deterministic** — no sklearn fitting required. The same
mapping is applied identically at training time and inference time, which is
critical for consistency.

---

## 2. Training Pipeline (`src/models/train.py`)

### 2a. Building the sklearn Pipeline

```python
Pipeline([
    ("scaler", RobustScaler()),
    ("model", DecisionTreeRegressor(random_state=...)),
])
```

`RobustScaler` normalises features using the median and IQR (resistant to
outliers). It is a no-op for tree splits but future-proofs the pipeline for
linear baselines and ensures the scaler is saved inside the artifact.

### 2b. Hyperparameter Search

`GridSearchCV` exhaustively tries every combination of:

- `model__max_depth` — controls how deep (complex) the tree can grow
- `model__min_samples_split` — minimum samples needed to split a node
- `model__min_samples_leaf` — minimum samples that must remain in each leaf

It scores each combination by **negative RMSE** across k folds and refits the
winner on the full training set. The result is the `best_estimator_` — a fully
fitted `Pipeline` with the best hyperparameters baked in.

### 2c. How the Tree Is Built Internally

A decision tree recursively partitions the training samples by asking binary
questions about feature values:

```
Is experience_level < 2?
├── YES → Is is_us_company == 1?
│         ├── YES → leaf A  (samples: 84, mean: $95 000)
│         └── NO  → leaf B  (samples: 31, mean: $62 000)
└── NO  → Is job_family < 3?
          ├── YES → leaf C  (samples: 56, mean: $140 000)
          └── NO  → leaf D  (samples: 22, mean: $185 000)
```

At each node, sklearn chooses the feature and threshold that **minimises the
weighted MSE** across the two resulting child groups. It repeats this until the
configured stopping criteria are met (`max_depth`, `min_samples_leaf`, etc.).
Every terminal node is a **leaf**. The tree stores:

- the **mean salary** of training samples in that leaf (the point estimate)
- the **node index** (an integer ID) used for lookup

---

## 3. Computing Leaf Ranges (`compute_leaf_ranges`)

Right after training, `compute_leaf_ranges()` captures the salary spread inside
each leaf from the real training labels:

```python
# 1. Apply every step except the final estimator (scaler only, no prediction)
X_transformed = pipeline[:-1].transform(X_train[FEATURE_COLUMNS])

# 2. Route every training row to its leaf — returns an array of leaf node IDs
leaf_ids = dt.apply(X_transformed)

# 3. For each leaf, compute Q25 and Q75 of the actual salaries there
for leaf_id in np.unique(leaf_ids):
    samples = y_values[leaf_ids == leaf_id]
    ranges[leaf_id] = (np.percentile(samples, 25), np.percentile(samples, 75))
```

`dt.apply()` is a sklearn method that walks the tree branches and returns the
**integer node index** each sample lands in, without making a prediction.

The result is a `dict[int, tuple[float, float]]` — one `(Q25, Q75)` entry per
leaf. This dictionary and the fitted pipeline are saved together into a single
`.joblib` artifact in `models/artifacts/`.

---

## 4. Prediction at Inference Time (`src/models/predict.py`)

When `predict(features)` is called with a new candidate's features:

```python
# Step 1 — Standard sklearn prediction: mean salary of the landing leaf
point_estimate = float(pipeline.predict(row)[0])

# Step 2 — Apply only the preprocessing steps (scaler), not the tree
X_transformed = pipeline[:-1].transform(row)

# Step 3 — Find which leaf this new sample lands in
leaf_id = int(pipeline.named_steps["model"].apply(X_transformed)[0])

# Step 4 — Retrieve the Q25–Q75 pre-computed from training data in that leaf
range_low, range_high = leaf_ranges.get(leaf_id, (point_estimate, point_estimate))
```

The `pipeline[:-1]` slice produces a sub-pipeline with all steps **except** the
final model, giving you the scaled features without triggering a prediction.
`.apply()` then deterministically walks the same tree branches the model uses
and returns the node ID — which is the key into `leaf_ranges`.

The fallback `(point_estimate, point_estimate)` handles the edge case of a
sample reaching a leaf that had no training samples (e.g. in a legacy artifact
saved before leaf ranges were computed).

### What the output looks like

```json
{
  "salary": {
    "mean":     125000,
    "low":      110000,
    "high":     140000,
    "currency": "USD"
  }
}
```

---

## 5. Why the Leaf Range Is Meaningful

The leaf represents a **peer group** — every training sample that answered the
same sequence of yes/no questions about their features. The point estimate is
the **mean** of that group's salaries. The `(Q25, Q75)` tells you how wide the
spread is **within that specific peer group**.

| Narrow range | Wide range |
|---|---|
| Peers earn consistently | Real variance exists in that profile |
| Model is confident in the number | Salary depends on factors not captured |

As the dataset grows, more samples per leaf → tighter, more statistically
stable Q25/Q75 bounds.

---

## 6. Data Flow Summary

```
Raw CSV
  └─► build_features()          → 8 numeric columns + salary_in_usd
        └─► GridSearchCV.fit()  → best DecisionTreeRegressor fitted
              └─► compute_leaf_ranges()  → dict[leaf_id → (Q25, Q75)]
                    └─► joblib.dump()    → models/artifacts/*.joblib

Inference request
  └─► pipeline[:-1].transform() → scaled features
        └─► dt.apply()          → leaf_id
              └─► leaf_ranges[leaf_id]  → (range_low, range_high)
              └─► pipeline.predict()   → point_estimate
```

---

## 7. Key Files

| File | Role |
|---|---|
| [src/features/engineering.py](../src/features/engineering.py) | Feature encoding — `FEATURE_COLUMNS`, ordinal mappings |
| [src/models/train.py](../src/models/train.py) | Pipeline construction, GridSearchCV, `compute_leaf_ranges`, save/load |
| [src/models/predict.py](../src/models/predict.py) | Inference entry point — `predict()`, `PredictionResult` |
| [models/registry/latest.json](../models/registry/latest.json) | Points to the current artifact path |
| [docs/salary-range-design.md](salary-range-design.md) | Design decision: regression + leaf IQR vs. band classification |
