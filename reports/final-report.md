# TrafficSense — Final Project Report

**Project title:** Traffic Congestion Prediction and Route Analysis Using Machine Learning

**Programme:** B.Tech Computer Science and Engineering — Machine Learning Project-Based Learning

**Student / team:** _[fill in]_

**Institution:** _[fill in]_

**Date:** _[fill in]_

---

> **How to read this report.** All numbers marked **[MEASURED]** were produced by
> running the code in this repository on the real UCI dataset and are copied from
> `backend/artifacts/metrics.json` and `model_metadata.json`. Placeholders marked
> **[FILL IN]** are for the student to complete (team details, screenshots,
> institution-specific information). No accuracy figure, metric or reference in
> this report has been invented.

---

## 1. Abstract

Urban traffic congestion degrades travel time, fuel efficiency and air quality.
This project, **TrafficSense**, builds an end-to-end machine learning system that
(i) predicts hourly traffic volume and labels its congestion category, and (ii)
analyses alternative routes on a real road network using congestion-aware edge
costs.

A Random Forest regression pipeline was trained on the UCI *Metro Interstate
Traffic Volume* dataset (48,204 raw hourly records from a Minnesota interstate
sensor, 2012–2018; 40,575 records after cleaning). Temporal and weather features
were engineered, including cyclical encodings of hour, day of week and month. A
chronological 70/15/15 train–validation–test split was used to prevent temporal
leakage, and all imputation and one-hot encoding were fitted inside a
scikit-learn `Pipeline` on training data only.

On the untouched test period, the selected Random Forest model achieved
**MAE = 250.63 vehicles/hour, RMSE = 439.62 vehicles/hour and R² = 0.9507
[MEASURED]**, compared with a training-mean baseline of MAE = 1735.20,
RMSE = 1981.01 and R² = −0.0015 **[MEASURED]**. Mapping predictions into four
congestion categories (Low, Moderate, High, Severe) using thresholds derived from
the training distribution produced a macro F1-score of **0.8751 [MEASURED]**.

The trained model is served by a FastAPI backend exposing health, metrics,
prediction and routing endpoints. Route analysis uses OSMnx to retrieve a
drivable OpenStreetMap network and NetworkX to compute distance-optimal,
free-flow-optimal and congestion-aware paths, with edge-specific congestion
multipliers. A React dashboard presents predictions, evaluation charts and an
interactive map. The system states its limitations explicitly: the model
describes one Minnesota sensor, and the default congestion scenario is a labelled
simulation, not measured traffic.

**Keywords:** traffic volume prediction, congestion classification, Random Forest,
scikit-learn, FastAPI, OSMnx, NetworkX, Dijkstra, route optimisation.

---

## 2. Introduction

Traffic congestion is a first-order problem in modern cities. It increases
journey times, raises fuel consumption and emissions, and reduces the
predictability of travel. Two complementary questions arise for planners and
commuters:

1. **Demand prediction** — how heavy will traffic be under given temporal and
   weather conditions?
2. **Route analysis** — given a network, which routes exist, and how does
   congestion change which route is best?

Machine learning addresses the first question well when historical observations
are available. Graph algorithms address the second. TrafficSense combines both in
a single application with a web interface, and it is deliberately explicit about
where machine learning stops being trustworthy.

This report documents the dataset, the preprocessing and feature engineering, the
model methodology, the system architecture, the routing formulation, the
experimental setup, the measured results, testing, and the limitations.

---

## 3. Problem statement

Given a public historical traffic dataset, build a system that:

1. Predicts hourly traffic volume from time and weather features.
2. Classifies the predicted volume into meaningful congestion categories using
   thresholds derived from data, not assumptions.
3. Evaluates the model honestly against a baseline on a chronological holdout.
4. Serves predictions through a validated API.
5. Performs real graph-based route analysis on a road network.
6. Distinguishes ordinary routing from congestion-aware routing, and never claims
   that a uniform multiplier changes the optimal path.
7. Presents the results, and the limitations, in an interactive dashboard.

---

## 4. Objectives and scope

### 4.1 Objectives

1. Reproducible preparation of the UCI Metro Interstate Traffic Volume dataset.
2. Feature engineering for cyclical time and weather.
3. Training and comparison of a baseline and a Random Forest regressor.
4. Chronological validation to avoid temporal leakage.
5. Derivation of congestion thresholds from training data only.
6. Honest reporting of MAE, RMSE, R², precision, recall, macro F1 and a confusion
   matrix.
7. A validated FastAPI service with graceful error handling.
8. Graph-based route analysis with OSMnx and NetworkX.
9. Edge-specific congestion-aware route costs, or an explicitly labelled
   demonstration scenario.
10. A responsive React dashboard with real data only.
11. Automated backend and frontend tests.
12. Honest documentation of limitations.

### 4.2 In scope

- Hourly traffic volume regression and congestion category classification.
- One-city demonstration of road-network routing with a configurable study area.
- A simulated congestion scenario for demonstrating congestion-aware routing.

### 4.3 Out of scope

- Live traffic ingestion.
- Per-road traffic prediction across an arbitrary city using this model.
- Real-time shortest-path queries at production scale.
- Geocoding and address search (coordinates only).

---

## 5. Literature review

Traffic prediction has been studied with statistical, classical machine learning
and deep learning methods. The references below were verified to resolve to real
publications; DOIs are given so a reader can check each one.

**Tree ensembles.** Breiman introduced Random Forests, an ensemble of decorrelated
decision trees using bootstrap aggregation and random feature subsampling
(Breiman, 2001). Random Forests handle non-linear interactions, mixed feature
types and outliers well, and provide feature importance, which makes them a
strong practical baseline for tabular traffic data. Liu (2017) applied Random
Forests specifically to road traffic congestion prediction. Chen and Guestrin
(2016) introduced XGBoost, a regularised gradient-boosting system that often
outperforms bagging on tabular data.

**Statistical and state-space approaches.** Early work modelled traffic flow as a
time series or state-space process. Flow rate and time-mean-speed prediction on
urban freeway networks using state-space models is a representative example
(Transportation Research Part C, 2014). These methods are interpretable but
depend on strong linearity or stationarity assumptions.

**Leakage and validation.** Kaufman, Rosset and Perlich (2012) give a systematic
treatment of leakage in data mining, showing how temporal contamination inflates
apparent accuracy. This motivates the chronological split used here.

**Geospatial network analysis.** Boeing (2017) introduced OSMnx, a Python library
for acquiring, constructing, analysing and visualising street networks from
OpenStreetMap. Dijkstra's algorithm (Dijkstra, 1959) remains the standard
single-source shortest-path method for non-negative edge weights, and is the
basis of the routing here.

**Datasets.** The UCI *Metro Interstate Traffic Volume* dataset (Hogue, 2019) is
widely used as a benchmark. Recent studies apply machine learning to it,
including linear regression and ensemble comparisons (Chandel, 2020; Rahman,
2025) and LSTM/Bi-LSTM sequence models (Pranolo, 2025). The recurring caveat in
this literature — that the dataset describes a single interstate sensor and does
not generalise to arbitrary roads — is a limitation TrafficSense adopts
explicitly.

**Research gap addressed here.** Much student work reports a single accuracy
number without a baseline, without a chronological split and without stating
geographic limits. TrafficSense instead reports a baseline comparison, a
chronological holdout, classification metrics, and explicit provenance for every
congestion number it displays.

---

## 6. Dataset description

| Property | Value |
| --- | --- |
| Name | Metro Interstate Traffic Volume |
| Repository | UCI Machine Learning Repository, dataset id 492 |
| URL | https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume |
| DOI | 10.24432/C5X60B |
| Licence | CC BY 4.0 |
| Observations (raw) | 48,204 hourly records |
| Observations (clean) | **40,575 [MEASURED]** |
| Period | 2012-10-02 to 2018-09-30 |
| Location | Interstate 94, Minneapolis–St. Paul, Minnesota, USA |
| Sensors | One |
| Target | `traffic_volume` (vehicles per hour) |

**Geographic limitation.** The dataset reflects the traffic pattern at a single
interstate sensor. It cannot support reliable live traffic prediction for
Hyderabad, or for every road in any city. TrafficSense states this in the API
response, in the dashboard, and here.

### 6.1 Column descriptions

| Column | Description | Unit |
| --- | --- | --- |
| `holiday` | US national holiday name, or `None` | category |
| `temp` | Air temperature | kelvin |
| `rain_1h` | Rainfall in the preceding hour | millimetres |
| `snow_1h` | Snowfall in the preceding hour | millimetres |
| `clouds_all` | Cloud cover | percent (0–100) |
| `weather_main` | Short weather label | category |
| `weather_description` | Long weather description (dropped: duplicates `weather_main`) | category |
| `date_time` | Observation timestamp | local time |
| `traffic_volume` | Hourly traffic volume (label) | vehicles per hour |

---

## 7. Data preprocessing

Implemented in `backend/app/preprocessing.py`. The pipeline is deterministic and
order-independent.

1. **Column normalisation.** Headers are lower-cased and mapped to snake case.
2. **Validation.** The expected columns are checked; a missing column raises an
   actionable error naming the file and the column.
3. **Timestamp parsing.** ISO-8601 parsing with a per-element fallback. Rows with
   unparseable timestamps are dropped.
4. **Numeric coercion.** Numeric fields are coerced; the label must be present.
5. **Plausibility.** Negative weather values and negative volume are treated as
   invalid and removed.
6. **Duplicate timestamps.** Records sharing a timestamp are removed, keeping the
   first. In this dataset the repeated timestamps carry identical traffic
   measurements, so removing them does not distort the label distribution.
7. **Chronological sort.** Rows are sorted by timestamp ascending.

**Measured effect:** 48,204 raw rows → **40,575 clean rows [MEASURED]** (17 exact
duplicate rows and 7,629 duplicate timestamps removed; the 7,629 duplicates share
identical `traffic_volume` values with their first occurrence).

**No synthetic data.** If the dataset is missing, the pipeline raises a clear
setup error and documents the download command. It never substitutes generated
data and presents it as real.

---

## 8. Feature engineering

| Feature | Definition | Unit / range |
| --- | --- | --- |
| `temp` | Air temperature | kelvin |
| `rain_1h` | Rainfall in the preceding hour | mm |
| `snow_1h` | Snowfall in the preceding hour | mm |
| `clouds_all` | Cloud cover | percent |
| `weather_main` | Weather category | category |
| `holiday` | Holiday name or `None` | category |
| `hour` | Hour of day | 0–23 |
| `day_of_week` | Monday = 0 … Sunday = 6 | 0–6 |
| `month` | Calendar month | 1–12 |
| `is_weekend` | 1 on Saturday/Sunday | 0/1 |
| `hour_sin`, `hour_cos` | Cyclical hour encoding | −1..1 |
| `dow_sin`, `dow_cos` | Cyclical day-of-week encoding | −1..1 |
| `month_sin`, `month_cos` | Cyclical month encoding | −1..1 |

Cyclical encoding is used because hour 23 and hour 0 are adjacent in time but far
apart as integers. The sine/cosine pair places them next to each other on the unit
circle.

**Why `weather_description` is dropped.** It is a near-duplicate of
`weather_main` and would inflate the feature space without adding information.

**Leakage prevention.** Only `traffic_volume` is the label. No lagged or rolling
target values are used as predictors, so the model cannot see the value it must
predict.

---

## 9. Model methodology

Implemented in `backend/app/train.py` and `backend/app/evaluation.py`.

### 9.1 Candidates

1. **Baseline — training mean.** Predicts the mean `traffic_volume` of the
   training split for every row. This is the honest reference: a model that
   cannot beat it is not useful.
2. **Random Forest Regressor.** `n_estimators=220`, `min_samples_leaf=2`, `random_state=42`, with
   median imputation for numeric features and most-frequent imputation plus
   one-hot encoding for categorical features, all inside the pipeline.
3. **XGBoost (optional).** Used only if the dependency is installed and reliable;
   otherwise skipped with a note. It was **not** available in this environment, so
   the comparison is between the baseline and Random Forest **[MEASURED]**.

### 9.2 Preprocessing inside the pipeline

All imputation and encoding are part of the scikit-learn `Pipeline`, so they are
fitted only on the training data of each split. This is the standard defence
against leakage.

### 9.3 Chronological split

| Split | Fraction | Rows | Period |
| --- | --- | --- | --- |
| Train | 70% | 28,403 | 2012-10-02 09:00 → 2017-05-10 05:00 |
| Validation | 15% | 6,086 | 2017-05-10 06:00 → 2018-01-19 15:00 |
| Test | 15% | 6,086 | 2018-01-19 16:00 → 2018-09-30 23:00 |

**[MEASURED]** — from `model_metadata.json`. No shuffling is applied. The model is
selected on validation, refitted on train + validation, and evaluated once on the
test split.

### 9.4 Congestion categories

Thresholds are the **25th, 60th and 85th percentiles of the training volume
distribution** (computed in `derive_congestion_thresholds()`), and they are saved
with the model:

| Category | Rule | Value **[MEASURED]** |
| --- | --- | --- |
| Low | volume ≤ `low_max` | ≤ 1,211 veh/h |
| Moderate | `low_max` < volume ≤ `moderate_max` | 1,211 – 4,256 veh/h |
| High | `moderate_max` < volume ≤ `high_max` | 4,256 – 5,502 veh/h |
| Severe | volume > `high_max` | > 5,502 veh/h |

The same thresholds are used for inference and for deriving the classification
metrics, so the reported classification performance is consistent with what the
API returns.

---

## 10. System architecture

```
React dashboard (Vite)
   │  fetch /api/health, /api/metrics, /api/predict, /api/routes
   ▼
FastAPI backend (Uvicorn)
   ├── model bundle   ← traffic_model.joblib + model_metadata.json + metrics.json
   ├── GraphCache     ← cached OSMnx graph (per study area)
   ├── plan_routes()  ← NetworkX shortest paths with edge-specific costs
   └── Pydantic       ← request/response validation
   ▼
Data sources
   ├── UCI Metro Interstate Traffic Volume CSV (training)
   └── OpenStreetMap Overpass API (road network, cached)
```

**Backend modules**

| Module | Responsibility |
| --- | --- |
| `app/preprocessing.py` | Loading, cleaning, feature engineering, chronological split |
| `app/evaluation.py` | Metrics, congestion categories, confusion matrix, chart series |
| `app/train.py` | Training orchestration, model selection, artifact writing |
| `app/model_store.py` | Artifact paths and model bundle loading |
| `app/routing.py` | Graph cache, congestion scenarios, path computation |
| `app/schemas.py` | Pydantic models |
| `app/main.py` | FastAPI app, endpoints, CORS, error handling |

---

## 11. Routing algorithm and cost formulation

Implemented in `backend/app/routing.py`.

### 11.1 Graph construction

A drivable network is retrieved with `osmnx.graph_from_point(..., network_type=
"drive")` for a configurable study area. Each edge is annotated with:

- `length_m` — segment length in metres;
- `speed_kph` — from the OSM `maxspeed` tag, falling back to a per-highway-class
  default when the tag is missing;
- `travel_time_s` — `length_m / (speed_kph / 3.6)`.

The graph is cached to `artifacts/osm_cache/graph_<hash>.pkl` keyed by the study
area, so the Overpass API is not queried on every request.

### 11.2 Cost formulation

For a path `P` consisting of edges `e`:

```
free_flow_time(P)     = Σ_e  travel_time_s(e)
congestion_cost(P)    = Σ_e  travel_time_s(e) × multiplier(e)
distance(P)           = Σ_e  length_m(e)
```

Three weights are used with NetworkX shortest-path routines:

| Weight | Objective |
| --- | --- |
| `length_m` | Shortest distance |
| `travel_time_s` | Fastest in free-flow conditions |
| `cost_s` | Fastest under the congestion scenario |

### 11.3 Congestion multipliers

A `CongestionScenario` supplies `multiplier(u, v, k, data)` per edge.

**`free_flow`** — multiplier 1.0 everywhere. Uniform.

**`simulated_peak`** — multiplier

```
1 + severity × (class_weight − 1) × jitter
```

where `class_weight` depends on the OSM highway class and `jitter ∈ [0.6, 1.4]`
is a deterministic hash of the edge identity `(u, v, k)`. Because the multiplier
varies per edge, it **can** change which path is optimal. This is a
**simulation**, not measured traffic; the API reports
`provenance: "simulated"`.

**`custom`** — caller-supplied `highway_multipliers` (per road class) and
`edge_multipliers` (explicit per-edge overrides). This is the integration point
for real road-specific traffic data.

> **Important correctness point.** A single global multiplier scales every route
> equally and therefore cannot change the ranking of paths by travel time. The
> API reports `is_uniform_multiplier`, and the system never claims that the
> `free_flow` scenario changes the fastest route.

### 11.4 Alternatives

Distinct simple paths are enumerated with `nx.shortest_simple_paths` under the
congestion-aware cost, up to `max_alternatives + 3` candidates. Duplicate paths
are removed by node sequence. Candidates are ranked with the congestion-aware
optimum first, then by congestion-aware time.

### 11.5 Edge cases handled

- **Disconnected origin/destination** → HTTP **404** with an explanatory message.
- **Origin and destination snap to the same node** → HTTP **422**.
- **Graph unavailable / Overpass unreachable** → HTTP **503** with a message
  naming the study area variables.
- **Offline mode** (`TRAFFICSENSE_OFFLINE=1`) → no download; cached graph only.
- **Missing edge attributes** → per-class speed fallbacks.

---

## 12. Implementation details

### 12.1 Backend endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Service information |
| GET | `/api/health` | Status and model availability |
| GET | `/api/metrics` | Saved evaluation metrics and metadata |
| POST | `/api/predict` | Traffic volume prediction and congestion category |
| POST | `/api/routes` | Route analysis with alternatives |

Error handling: **503** when the model or graph is unavailable, **404** when no
route exists, **422** for invalid input or an unknown scenario. CORS origins come
from `TRAFFICSENSE_CORS_ORIGINS` and default to local development origins, never
a wildcard.

### 12.2 Frontend components

| Component | Responsibility |
| --- | --- |
| `Header.jsx` | Branding, subtitle, backend/model status |
| `PredictionForm.jsx` | Validated inputs with units, loading state, reset |
| `PredictionCard.jsx` | Predicted volume, category, thresholds, disclaimer |
| `MetricsPanel.jsx` | Metrics, baseline comparison, confusion matrix |
| `TrafficCharts.jsx` | Actual-vs-predicted, by-hour, by-weekday, feature importance |
| `RoutePlanner.jsx` | Coordinate inputs, map-click selection, route comparison |
| `RouteMap.jsx` | Leaflet map with route geometries and OSM attribution |

Charts are built only from the backend's saved metrics. No chart value is
invented. Styling uses plain CSS in `src/style.css`; **Tailwind CSS is not
used**.

---

## 13. Experimental setup

| Item | Value |
| --- | --- |
| Python | 3.13.15 |
| scikit-learn | 1.4+ |
| pandas / NumPy | 2.2+ / 1.26+ |
| Random Forest | `n_estimators=220`, `min_samples_leaf=2`, `random_state=42` |
| Split | Chronological 70 / 15 / 15 |
| Selection metric | Validation MAE |
| Test evaluation | Single evaluation on the untouched test split |
| Reproducibility | Fixed `random_state=42`; deterministic split |
| Hardware | 4 vCPU, 15 GiB RAM, Linux container |

**Reproduce with**

```bash
cd backend
python -m app.train
```

Training completed in roughly half a minute on the hardware above; the Random
Forest fit itself took **5.84 s [MEASURED]** on the training split.

---

## 14. Actual model results

All figures below are copied from `backend/artifacts/metrics.json`. They are
**measured**, not estimated.

### 14.1 Test-set regression metrics

| Model | MAE (veh/h) | RMSE (veh/h) | R² |
| --- | --- | --- | --- |
| Training-mean baseline | 1735.20 | 1981.01 | −0.0015 |
| **Random Forest (selected)** | **250.63** | **439.62** | **0.9507** |

The Random Forest reduces MAE by **85.6%** relative to the baseline. The baseline
R² of approximately zero is expected: predicting the training mean on a test
period with a different mean yields no explanatory power.

### 14.2 Validation-set metrics (used for selection)

| Model | MAE | RMSE | R² | Fit time (s) |
| --- | --- | --- | --- | --- |
| Baseline (mean) | 1722.10 | 1968.34 | −0.0024 | 0.08 |
| Random Forest | 283.07 | 486.63 | 0.9387 | 5.84 |

### 14.3 Congestion classification (test set, training-derived thresholds)

| Metric | Value |
| --- | --- |
| Macro precision | 0.8790 |
| Macro recall | 0.8718 |
| **Macro F1** | **0.8751** |

**Per class [MEASURED]**

| Category | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Low | 0.9771 | 0.9363 | 0.9563 | 1,461 |
| Moderate | 0.8773 | 0.8867 | 0.8820 | 2,065 |
| High | 0.7990 | 0.8347 | 0.8165 | 1,652 |
| Severe | 0.8625 | 0.8293 | 0.8456 | 908 |

**Confusion matrix [MEASURED]**

| Actual \ Predicted | Low | Moderate | High | Severe |
| --- | --- | --- | --- | --- |
| **Low** | 1368 | 87 | 3 | 3 |
| **Moderate** | 32 | 1831 | 190 | 12 |
| **High** | 0 | 168 | 1379 | 105 |
| **Severe** | 0 | 1 | 154 | 753 |

Errors concentrate between adjacent categories, which is the expected pattern
for an ordinal label. The "High" class is the hardest, consistent with its
narrower volume band.

### 14.4 Feature importance (top features) [MEASURED]

| Feature | Importance |
| --- | --- |
| `hour_cos` | 0.6573 |
| `hour` | 0.1379 |
| `hour_sin` | 0.0544 |
| `day_of_week` | 0.0375 |
| `is_weekend` | 0.0323 |
| `dow_sin` | 0.0274 |

Hour of day dominates, which matches the strong diurnal pattern of urban traffic.
Weather features contribute far less at this single sensor.

### 14.5 Mean volume by weekday [MEASURED]

| Day | Mean volume (veh/h) | Records |
| --- | --- | --- |
| Monday | 3,313.56 | 5,799 |
| Tuesday | 3,534.72 | 5,703 |
| Wednesday | 3,608.28 | 5,803 |
| Thursday | 3,653.91 | 5,797 |
| Friday | 3,674.78 | 5,877 |
| Saturday | 2,822.56 | 5,784 |
| Sunday | 2,426.25 | 5,812 |

Weekday volumes exceed weekend volumes, as expected for a commuting corridor.

### 14.6 Routing results [MEASURED]

Example request in the demo study area (Gachibowli, Hyderabad), scenario
`simulated_peak`, severity 1.0:

| Rank | Route | Distance (km) | Free-flow (min) | Congestion-aware (min) | Mean multiplier |
| --- | --- | --- | --- | --- | --- |
| 1 | shortest distance | 1.346 | 1.62 | 2.07 | 1.267 |

Graph size: 38 nodes, 63 edges. Because the study area is small, the three
objective functions frequently coincide; the system returns multiple alternatives
where distinct paths exist and a clear 404 where none exists.

> The road-network figures depend on the study area, which is configurable, and on
> the OpenStreetMap data at the time of the request. Reproduce them with the
> command in section 13 and the request in `README.md` §12.

---

## 15. Graphs and screenshots to be added

Figure 1 is generated by training and is included in this repository.

| Figure | Status | Location |
| --- | --- | --- |
| Fig. 1 — Actual vs predicted traffic volume (test period) | **Generated** | `reports/figures/actual_vs_predicted.png` |
| Fig. 2 — Predicted vs actual scatter | Generated by the dashboard | Screenshot **[FILL IN]** |
| Fig. 3 — Mean traffic volume by hour of day | Generated by the dashboard | Screenshot **[FILL IN]** |
| Fig. 4 — Mean traffic volume by weekday | Generated by the dashboard | Screenshot **[FILL IN]** |
| Fig. 5 — Feature importance bar chart | Generated by the dashboard | Screenshot **[FILL IN]** |
| Fig. 6 — Confusion matrix heatmap | Generated by the dashboard | Screenshot **[FILL IN]** |
| Fig. 7 — Dashboard: prediction form and result | — | Screenshot **[FILL IN]** |
| Fig. 8 — Dashboard: route planner with alternative routes | — | Screenshot **[FILL IN]** |
| Fig. 9 — System architecture diagram | Section 10 | Redraw neatly **[FILL IN]** |

To capture dashboard screenshots: start the backend and frontend (README §10–11),
open `http://localhost:5173`, and capture the prediction section, the analytics
section, and the route planner with at least two routes displayed.

---

## 16. Testing and discussion

### 16.1 Backend tests

`cd backend && python -m pytest -q`

**Result: 73 passed, 1 warning [MEASURED]**

| File | Tests | Coverage |
| --- | --- | --- |
| `tests/test_preprocessing.py` | 23 | Column validation, cleaning, timestamp handling, feature engineering, chronological split |
| `tests/test_api.py` | 24 | Health, metrics, prediction structure, input validation, missing-model behaviour, threshold boundaries |
| `tests/test_routing.py` | 18 | Nearest node, distance and cost maths, shortest path, alternatives, disconnected graphs, caching, download failure |
| `tests/test_routes_api.py` | 9 | Routing endpoint, multiple alternatives, 404, 503, custom scenario, offline mode |

**Unit tests never download road data.** They inject a small synthetic graph or a
temporary cache and pass `allow_download=False`. The remaining warning is a
Starlette deprecation notice about `httpx` in `TestClient`; it does not affect
correctness.

### 16.2 Frontend tests

`cd frontend && npm test`

**Result: 12 passed [MEASURED]**

| File | Tests | Coverage |
| --- | --- | --- |
| `src/test/Prediction.test.jsx` | 7 | Form submission, validation errors, loading state, API error display, reset, prediction card rendering, disclaimer |
| `src/test/RoutePlanner.test.jsx` | 5 | Empty state, successful comparison rendering, backend failure, identical-points validation, coordinate range validation |

### 16.3 Lint and production build

| Check | Result |
| --- | --- |
| `npm run lint` | 0 errors, 0 warnings **[MEASURED]** |
| `npm run build` | Success; `dist/` produced **[MEASURED]** |

### 16.4 Integration verification

| Check | Result |
| --- | --- |
| `GET /api/health` | `model_available: true`, `random_forest`, `dataset_rows: 40575` **[MEASURED]** |
| `POST /api/predict` (08:00, June, clear) | 5,775.9 veh/h, category **Severe** **[MEASURED]** |
| `POST /api/routes` | HTTP 200, route geometry returned **[MEASURED]** |
| Dashboard renders real metrics and confusion matrix | Verified in a browser **[MEASURED]** |
| Dashboard route planner renders the comparison table | Verified in a browser **[MEASURED]** |

### 16.5 Discussion

- The Random Forest clearly outperforms the baseline, confirming that time and
  weather features carry real signal.
- The hour of day dominates importance, which is consistent with commuting
  behaviour.
- Classification errors are concentrated between neighbouring categories, which
  is the expected behaviour for a monotone label with fixed cut points.
- The congestion-aware route can differ from the shortest-distance route only
  because multipliers are edge-specific. This was a deliberate design decision to
  avoid the common error of applying one global multiplier.
- The study area is small, so many origin/destination pairs have a single
  feasible path. The system reports the alternatives that exist rather than
  inventing them.

---

## 17. Limitations

1. **Single sensor.** The model reflects one Minnesota interstate sensor. It is
   not valid for other cities or roads.
2. **Historical, not live.** Predictions are pattern-based estimates, not live
   measurements. The API and UI state this.
3. **Simulated congestion.** The default routing scenario uses simulated,
   edge-specific multipliers. It is labelled `provenance: "simulated"`.
4. **Model not applied per road.** The volume model is not used to set per-road
   congestion, because it was never trained on those roads.
5. **Speed-limit dependence.** Travel times rely on OSM speed limits, which may
   be missing or outdated.
6. **Node snapping.** Origins and destinations snap to the nearest road node.
7. **Study-area coverage.** Points outside the configured area may snap far away
   or have no connecting path.
8. **External dependency.** Road-network retrieval depends on the public Overpass
   API, which can be slow or rate-limited; caching and offline mode mitigate this.
9. **Small demo graph.** The default study area yields a small graph, so
   alternative routes are not always available.
10. **XGBoost not compared.** The optional XGBoost candidate was unavailable in
    this environment, so only the baseline and Random Forest are compared.

---

## 18. Conclusion and future scope

TrafficSense delivers a complete, working pipeline: a real public dataset is
cleaned reproducibly; temporal and weather features are engineered; a Random
Forest model is trained and evaluated on a chronological holdout with honest
metrics (test MAE 250.63, RMSE 439.62, R² 0.9507, macro F1 0.8751 against a
baseline of MAE 1735.20); the model is served through a validated FastAPI
service; and real graph-based route analysis with edge-specific congestion costs
is presented in an interactive React dashboard.

The project's distinguishing feature is its honesty. It reports a baseline, it
uses a chronological split, it derives congestion thresholds from training data,
it labels simulated congestion as simulated, and it refuses to present a uniform
multiplier as if it changed the optimal route.

**Future scope**

1. Replace simulated multipliers with observed, road-specific traffic data.
2. Train per-corridor models once multi-location data is available.
3. Add travel-time prediction with confidence intervals.
4. Implement time-dependent (departure-time-aware) routing.
5. Add geocoding and address search to the route planner.
6. Stream live traffic and compare observed against predicted.
7. Add model monitoring for drift and periodic retraining.
8. Scale routing with contraction hierarchies or a precomputed routing service.

---

## 19. References

1. Hogue, J. (2019). *Metro Interstate Traffic Volume* [Dataset]. UCI Machine
   Learning Repository. https://doi.org/10.24432/C5X60B
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
   https://doi.org/10.1023/A:1010933404324
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
   *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
   Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785
4. Kaufman, S., Rosset, S., & Perlich, C. (2012). Leakage in Data Mining:
   Formulation, Detection, and Avoidance. *ACM Transactions on Knowledge
   Discovery from Data*, 6(4), 1–21. https://doi.org/10.1145/2382577.2382579
5. Liu, Y. (2017). Prediction of Road Traffic Congestion Based on Random Forest.
   *2017 10th International Symposium on Computational Intelligence and Design
   (ISCID)*. https://doi.org/10.1109/iscid.2017.216
6. Boeing, G. (2017). OSMnx: New Methods for Acquiring, Constructing, Analyzing,
   and Visualizing Complex Street Networks. *Computers, Environment and Urban
   Systems*, 65, 126–139. https://doi.org/10.1016/j.compenvurbsys.2017.05.004
7. Dijkstra, E. W. (1959). A Note on Two Problems in Connexion with Graphs.
   *Numerische Mathematik*, 1(1), 269–271. https://doi.org/10.1007/BF01386390
8. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
   *Journal of Machine Learning Research*, 12, 2825–2830.
   https://jmlr.org/papers/v12/pedregosa11a.html
9. McKinney, W. (2010). Data Structures for Statistical Computing in Python.
   *Proceedings of the 9th Python in Science Conference*, 56–61.
   https://doi.org/10.25080/majora-92bf1922-00a
10. Harris, C. R., et al. (2020). Array Programming with NumPy. *Nature*,
    585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
11. Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in
    Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
12. Ramírez, A., et al. (2014). Flow Rate and Time Mean Speed Predictions for the
    Urban Freeway Network Using State Space Models. *Transportation Research Part
    C: Emerging Technologies*, 44, 20–32.
    https://doi.org/10.1016/j.trc.2014.02.014
13. Chandel, S. (2020). An Accurate Estimation of Interstate Traffic of Metro City
    Using Linear Regression Model of Machine Learning. *SSRN Electronic Journal*.
    https://doi.org/10.2139/ssrn.3598310
14. Rahman, M. (2025). A Machine Learning Paradigms for Metro Interstate Traffic
    Volume Prediction. *2025 International Conference on Electrical, Computer and
    Communication Engineering (ECCE)*.
    https://doi.org/10.1109/ecce64574.2025.11013836
15. Pranolo, A. (2025). Urban Traffic Volume Prediction using LSTM and Bi-LSTM:
    Performance Evaluation on the Metro Interstate Traffic Volume Dataset. *ILKOM
    Jurnal Ilmiah*, 17(3), 227–240.
    https://doi.org/10.33096/ilkom.v17i3.3001.227-240
16. OpenStreetMap contributors. *OpenStreetMap* [Data]. Open Database Licence
    (ODbL). https://www.openstreetmap.org/copyright
17. FastAPI documentation. https://fastapi.tiangolo.com
18. NetworkX documentation. https://networkx.org
