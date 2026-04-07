# Salary Range Design — Decision Record

## The Question

Should the `/predict` endpoint return a **point estimate + data-driven range** (current approach),
or should salary be **classified into pre-defined bands** (alternative)?

---

## Approach A — Regression + Leaf IQR (current)

The tree predicts a number. After training, the Q25–Q75 of actual training
salaries is recorded for each leaf node. At inference time the input is routed
to its leaf and those pre-computed bounds are returned alongside the point
estimate.

**What the response looks like:**
```json
{
  "predicted_salary": 125000,
  "salary_range_low":  110000,
  "salary_range_high": 140000
}
```

**Strengths:**
- Range boundaries are fully data-driven — no manual choices
- The numeric point estimate is preserved for downstream use (LLM narrative, charts)
- Range width is meaningful: a narrow band means peers earn consistently; a wide band signals real variance in that group
- Gets better as data grows — more samples per leaf → tighter, more stable Q25/Q75

**Weaknesses:**
- The tree was optimised for MSE of the point estimate, not for producing meaningful groups for range computation
- With 607 rows and 27 leaves (~18 samples/leaf on average), Q25/Q75 can be statistically fragile — a few unusual salaries in a small leaf shift the bounds noticeably

---

## Approach B — Salary Band Classification (alternative)

Bin `salary_in_usd` into fixed bands (e.g. `<50k / 50–75k / 75–100k / 100–150k / 150–200k / >200k`),
make the band the target, and switch to `DecisionTreeClassifier`.

**What the response would look like:**
```json
{
  "predicted_band": "100k–150k"
}
```

**Strengths:**
- Output is easy to communicate; matches how job postings phrase salary

**Weaknesses:**
- Bin boundaries are arbitrary — any cutoff produces a cliff edge where $74,999 and $75,001 look completely different
- The numeric point estimate is lost — the LLM narrative and charts lose a useful signal
- Band width is fixed by the human-chosen boundaries, not by actual data variance in each group
- "80% classification accuracy" sounds good but hides whether you predicted $75,001 or $99,999 for a true salary of $76,000
- Boundaries would need revisiting as data distribution changes

---

## Comparison

| Criterion | Regression + Leaf IQR | Band Classification |
|-----------|----------------------|---------------------|
| Range boundaries | Data-driven | Manually defined |
| Numeric precision | Kept | Lost |
| Bin boundary cliff-edge | None | Yes |
| Small dataset sensitivity | Visible — ranges widen | Hidden — not eliminated |
| Model optimised for | Minimising MSE of point | Minimising classification error on bins |
| Range width is meaningful | Yes — reflects peer variance | No — always the fixed bin width |
| Improves with more data | Yes — naturally | No — boundaries need manual revisiting |

**Decision: keep Approach A.** The leaf IQR method is honest about data variance and avoids the arbitrary boundary problem. The classification approach trades one issue (noisy ranges) for two others (lost precision, artificial cutoffs) without fixing the root cause.

---

## Root Cause

The noisiness of the current ranges is not a method problem — it is a **data size problem**.
607 rows producing 27 leaves means ~18 samples per leaf on average,
and some leaves contain fewer. Q25 of 8 numbers is approximately the 2nd-lowest value.

---

## Levers to Pull if Ranges Feel Too Wide or Noisy

Pull these in order; stop when ranges are acceptable.

### 1. More data (highest impact)
Larger leaves → more stable percentiles.
The Q25/Q75 of 50 samples is far more reliable than of 18.
Collect more records or augment with a larger public dataset.

### 2. Shallower tree — reduce `max_depth`
Fewer splits → larger leaves → more samples per leaf → more stable Q25/Q75.
Trade-off: ranges become less specific (one range covers a broader profile group).

Add values to `dt_max_depth_options` in `config/settings.py` and retrain:
```python
dt_max_depth_options: list[int | None] = [3, 4, 5, 8, 12, 20, None]
```
GridSearchCV will pick the depth that minimises CV-RMSE, but you can also
cap the search at a shallower depth deliberately.

### 3. Increase `min_samples_leaf`
Forces each leaf to contain at least N training samples before a split is allowed.
This directly controls the minimum leaf size and therefore the minimum number of
points used to compute Q25/Q75.

```python
dt_min_samples_leaf_options: list[int] = [1, 2, 4, 8, 16, 32]
```
Setting a higher minimum (e.g. 16 or 32) guarantees more stable percentiles
at the cost of less granular splits.

### 4. Switch to quantile regression (best method upgrade)
Train two additional `DecisionTreeRegressor` models with
`criterion="friedman_mse"` but optimised against the pinball loss at τ=0.25
and τ=0.75 respectively (via `sklearn`'s `GradientBoostingRegressor` with
`loss="quantile"`, or `lightgbm`).

This produces Q25/Q75 that the *model itself was optimised for*, rather than
computed post-hoc from leaf samples. More principled, more stable, but
requires two extra models to train and serve.

### 5. Widen the percentile band
Use Q10–Q90 instead of Q25–Q75 if the goal is to reduce the chance of the
true salary falling outside the returned range.

Change the percentile calls in `compute_leaf_ranges()` in `src/models/train.py`:
```python
# Current
float(np.percentile(samples, 25)),
float(np.percentile(samples, 75)),

# Wider band
float(np.percentile(samples, 10)),
float(np.percentile(samples, 90)),
```
This does not reduce noise — it trades tighter bounds for better coverage.
