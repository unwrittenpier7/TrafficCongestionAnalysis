# TrafficSense

**Traffic Congestion Prediction and Route Analysis Using Machine Learning**

A B.Tech CSE Machine Learning Project-Based Learning (PBL) application. TrafficSense
trains a machine learning model on a real public traffic dataset, predicts hourly
traffic volume and congestion category, and analyses alternative routes on a real
OpenStreetMap road network. Results are presented in a React dashboard.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Problem statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Architecture](#4-architecture)
5. [Technology stack](#5-technology-stack)
6. [Dataset source and attribution](#6-dataset-source-and-attribution)
7. [Dataset column descriptions](#7-dataset-column-descriptions)
8. [Setup instructions (Windows PowerShell)](#8-setup-instructions-windows-powershell)
9. [Model training](#9-model-training)
10. [Running the backend](#10-running-the-backend)
11. [Running the frontend](#11-running-the-frontend)
12. [API usage examples](#12-api-usage-examples)
13. [Testing](#13-testing)
14. [Configuration reference](#14-configuration-reference)
15. [Model artifacts and what to commit](#15-model-artifacts-and-what-to-commit)
16. [Deployment](#16-deployment)
17. [Limitations](#17-limitations)
18. [Future scope](#18-future-scope)
19. [Troubleshooting](#19-troubleshooting)
20. [Academic deliverables](#20-academic-deliverables)

---

## 1. Project overview

TrafficSense is a full-stack application with three parts:

| Part | What it does |
| --- | --- |
| **ML backend** | Cleans the UCI Metro Interstate Traffic Volume dataset, engineers time features, trains a Random Forest regressor, evaluates it on a chronological holdout, and saves the fitted pipeline plus real metrics. |
| **FastAPI service** | Serves the trained model (`/api/predict`), the saved evaluation results (`/api/metrics`), and road-network route analysis (`/api/routes`). |
| **React dashboard** | Prediction form, prediction result with congestion category, model analytics charts built from real evaluation output, and an interactive OpenStreetMap route planner. |

Everything runs locally. No paid API and no API key is required for the core
demonstration.

---

## 2. Problem statement

Urban traffic congestion increases travel time, fuel consumption and air
pollution. Operators and commuters need to answer two questions:

1. **How heavy will traffic be?** Given the hour, the day and weather conditions,
   can we estimate the hourly traffic volume and label the congestion level?
2. **Which route should I take?** Given an origin and destination, what are the
   alternative routes, how long is each, and how does congestion change the
   ranking of routes?

TrafficSense addresses both questions with a machine learning model for volume
prediction and a graph-based route optimiser for route analysis.

---

## 3. Objectives

1. Build a reproducible data pipeline for a real public traffic dataset.
2. Engineer cyclical time features and use the dataset's own weather and holiday
   features.
3. Train and compare a baseline and a Random Forest regressor without temporal
   leakage.
4. Derive congestion category thresholds from training data only.
5. Report honest evaluation metrics: MAE, RMSE, R², and classification
   precision, recall, macro F1 and a confusion matrix.
6. Expose the model through a validated FastAPI service.
7. Perform real graph-based route analysis with OSMnx and NetworkX.
8. Distinguish ordinary routing from congestion-aware routing using
   edge-specific costs.
9. Present results in a responsive React dashboard that never fabricates data.
10. Document limitations honestly and provide automated tests.

---

## 4. Architecture

```
                    ┌───────────────────────────────────────┐
                    │            React dashboard            │
                    │  Vite + React + Recharts + Leaflet    │
                    │                                       │
                    │  PredictionForm ─┐   RoutePlanner ─┐   │
                    │  PredictionCard  │   RouteMap      │   │
                    │  MetricsPanel    │   Charts        │   │
                    └─────────┬────────┴────────┬────────┘   │
                              │  fetch /api/*   │            │
                              ▼                 ▼            │
                    ┌───────────────────────────────────────┐
                    │            FastAPI backend            │
                    │                                       │
                    │  GET  /             service info      │
                    │  GET  /api/health   model status      │
                    │  GET  /api/metrics  saved results     │
                    │  POST /api/predict  ML prediction     │
                    │  POST /api/routes   route analysis    │
                    └──────┬──────────────────────┬─────────┘
                           │                      │
              ┌────────────▼──────────┐  ┌────────▼─────────────────┐
              │  scikit-learn model   │  │  OSMnx + NetworkX graph  │
              │  traffic_model.joblib │  │  cached road network     │
              │  metrics.json         │  │  edge-specific costs     │
              │  model_metadata.json  │  │  Dijkstra / k-shortest   │
              └────────────▲──────────┘  └────────▲─────────────────┘
                           │                      │
              ┌────────────┴──────────┐  ┌────────┴─────────────────┐
              │ UCI Metro Interstate  │  │  OpenStreetMap (Overpass │
              │ Traffic Volume CSV    │  │  API) drivable network   │
              └───────────────────────┘  └──────────────────────────┘
```

**Backend request flow for a prediction**

1. Pydantic validates and range-checks the request body.
2. `_feature_row()` builds a one-row DataFrame using exactly the training feature
   order.
3. The saved scikit-learn `Pipeline` applies the fitted imputers and encoder and
   predicts the volume.
4. The prediction is mapped to a congestion category using the thresholds saved
   with the model.

**Backend request flow for routes**

1. Pydantic validates coordinates and the congestion scenario.
2. `GraphCache` returns a cached road graph, downloading it only if absent.
3. Origin and destination snap to the nearest road-network nodes.
4. `plan_routes()` computes the distance-optimal, free-flow-optimal and
   congestion-aware-optimal paths, plus additional distinct alternatives.
5. Each path is summarised with distance, free-flow time, congestion-aware time,
   multipliers and geometry.

---

## 5. Technology stack

**Machine learning backend**

| Package | Purpose |
| --- | --- |
| Python 3.11+ (tested on 3.13.15) | Runtime |
| FastAPI + Uvicorn | Web API and ASGI server |
| Pandas + NumPy | Data loading and feature engineering |
| scikit-learn | Pipelines, imputation, encoding, Random Forest |
| Joblib | Model persistence |
| Pydantic | Request/response validation |
| NetworkX | Graph algorithms (Dijkstra, k-shortest simple paths) |
| OSMnx | Road-network retrieval from OpenStreetMap |
| Matplotlib | Actual-versus-predicted figure |
| Pytest + HTTPX | Automated tests |

**Frontend**

| Package | Purpose |
| --- | --- |
| React 18 + Vite | UI framework and build tool |
| Recharts | Analytics charts |
| Leaflet + React Leaflet | Interactive OpenStreetMap map |
| Plain CSS (`src/style.css`) | Styling. **No Tailwind CSS is used.** |
| Vitest + Testing Library | Component tests |

---

## 6. Dataset source and attribution

| Field | Value |
| --- | --- |
| **Dataset** | Metro Interstate Traffic Volume |
| **Source** | UCI Machine Learning Repository |
| **URL** | <https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume> |
| **Direct CSV** | <https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip> |
| **Licence** | CC BY 4.0 |
| **Coverage** | Hourly observations, 2012-10-02 to 2018-09-30 |
| **Location** | Interstate 94, Minneapolis–St. Paul, Minnesota, USA |

> **Geographic limitation.** The dataset comes from **one sensor on one
> interstate highway in Minnesota**. It does not describe Hyderabad, or any other
> city, or every road in a city. TrafficSense predictions are estimates of the
> volume pattern at that sensor's location. The application states this
> limitation in the API response, in the dashboard, and in the report.

**Citation**

> Hogue, J. (2019). *Metro Interstate Traffic Volume* [Dataset]. UCI Machine
> Learning Repository. https://doi.org/10.24432/C5X60B

**Download the data**

```bash
# Option A: manual download (recommended; works without the ucimlrepo package)
# Download the ZIP from the URL above, extract it, and place the CSV at:
#   backend/data/Metro_Interstate_Traffic_Volume.csv

# Option B: automatic download via ucimlrepo
cd backend
python -c "from ucimlrepo import fetch_ucirepo; d = fetch_ucirepo(id=492); d.data.original.to_csv('data/Metro_Interstate_Traffic_Volume.csv', index=False)"
```

The loader accepts a manually placed CSV, a `.zip` archive, or the
`ucimlrepo` package. If no data is found, the pipeline fails with a clear setup
message. **It never generates synthetic data and presents it as real traffic.**

---

## 7. Dataset column descriptions

Raw columns (as shipped by UCI, before normalisation):

| Column | Description | Unit |
| --- | --- | --- |
| `holiday` | US national holiday name, or `None` | category |
| `temp` | Air temperature at the sensor | kelvin |
| `rain_1h` | Rainfall in the preceding hour | millimetres |
| `snow_1h` | Snowfall in the preceding hour | millimetres |
| `clouds_all` | Cloud cover | percent (0–100) |
| `weather_main` | Short weather label from the source feed | category |
| `weather_description` | Longer weather description | category (not used as a predictor) |
| `date_time` | Timestamp of the observation | local time |
| `traffic_volume` | Hourly traffic volume — **the label** | vehicles per hour |

**Engineered features** (defined in `backend/app/preprocessing.py` and saved in
`model_metadata.json`):

| Feature | Description | Unit / range |
| --- | --- | --- |
| `hour` | Hour of day from `date_time` | 0–23 |
| `day_of_week` | Day of week, Monday = 0 | 0–6 |
| `month` | Calendar month | 1–12 |
| `is_weekend` | 1 on Saturday or Sunday | 0/1 |
| `hour_sin`, `hour_cos` | Cyclical encoding of the hour | −1..1 |
| `dow_sin`, `dow_cos` | Cyclical encoding of the day of week | −1..1 |
| `month_sin`, `month_cos` | Cyclical encoding of the month | −1..1 |

**Leakage avoidance.** Only `traffic_volume` is the label. `weather_description`
is dropped because it duplicates `weather_main`. No future traffic measurement
and no rolling or lagged target value is used as a predictor, so the model cannot
peek at the value it must predict.

**Cleaning rules** (`clean_dataset`):

1. Column names are normalised to lower snake case.
2. Timestamps are parsed; unparseable rows are dropped.
3. Numeric columns are coerced; the label must be present.
4. Negative weather values and negative volume are treated as invalid.
5. Rows with a duplicate timestamp are removed, keeping the first record. The
   source file contains several thousand repeated timestamps; those rows carry
   identical traffic measurements, so dropping them does not bias the label.

On the real dataset: **48,204 raw rows → 40,575 clean rows.**

---

## 8. Setup instructions (Windows PowerShell)

These commands are for Windows. The equivalent Linux/macOS commands are noted
after each step.

### 8.1 Prerequisites

- Python 3.11 or newer
- Node.js 18 or newer
- Git

### 8.2 Clone and enter the project

```powershell
git clone <your-repository-url>
cd traffic-sense
```

### 8.3 Create and activate a virtual environment

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run this once:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Linux/macOS equivalent:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

### 8.4 Install backend dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 8.5 Install frontend dependencies

```powershell
cd ..\frontend
npm install
```

### 8.6 Provide the dataset

Download the CSV and place it at `backend/data/Metro_Interstate_Traffic_Volume.csv`
(see [section 6](#6-dataset-source-and-attribution)).

---

## 9. Model training

From the `backend` directory, with the virtual environment active:

```powershell
python -m app.train
```

Useful options:

```powershell
python -m app.train --help
python -m app.train --no-download          # use only a local CSV
python -m app.train --no-xgboost           # skip the optional XGBoost candidate
python -m app.train --artifacts-dir ..\artifacts
```

The script:

1. Loads and cleans the dataset.
2. Splits it chronologically: 70% train, 15% validation, 15% test. No shuffling.
3. Builds a scikit-learn `Pipeline` per candidate model. All imputation and
   one-hot encoding are fitted **inside** the pipeline on training data only.
4. Evaluates each candidate on the validation split and selects the best by MAE.
5. Refits the selected model on train + validation.
6. Evaluates once on the untouched test split.
7. Derives congestion thresholds from the **training** distribution and
   classifies test predictions with them.
8. Writes the artifacts listed below.

**Artifacts written to `backend/artifacts/`**

| File | Contents |
| --- | --- |
| `traffic_model.joblib` | Fitted scikit-learn pipeline (compressed) |
| `model_metadata.json` | Feature definitions, units, thresholds, split periods, dataset info |
| `metrics.json` | Real evaluation metrics, confusion matrix, chart series |
| `actual_vs_predicted.png` | Actual-versus-predicted figure |

---

## 10. Running the backend

From the `backend` directory, with the virtual environment active:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

- API documentation (Swagger UI): <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/api/health>

A production-style start without `--reload`:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> Use **one worker** unless you place the model artifacts on shared storage. Each
> worker loads its own copy of the model into memory.

---

## 11. Running the frontend

In a second terminal, from the `frontend` directory:

```powershell
npm run dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api` to
`http://127.0.0.1:8000`, so no CORS configuration is needed in development.

Production build and preview:

```powershell
npm run build
npm run preview
```

---

## 12. API usage examples

### GET `/` — service information

```bash
curl http://127.0.0.1:8000/
```

### GET `/api/health`

```bash
curl http://127.0.0.1:8000/api/health
```

```json
{
  "status": "ok",
  "model_available": true,
  "model_name": "random_forest",
  "model_created_utc": "2026-10-27T13:19:32+00:00",
  "dataset_rows": 40575,
  "detail": "Trained model loaded and ready for predictions."
}
```

### GET `/api/metrics`

Returns the real saved metrics, metadata, confusion matrix and chart series.

```bash
curl http://127.0.0.1:8000/api/metrics
```

### POST `/api/predict`

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 8,
    "day_of_week": 1,
    "month": 6,
    "temperature": 22,
    "temperature_unit": "celsius",
    "rainfall_mm": 0,
    "snowfall_mm": 0,
    "cloud_cover": 30,
    "weather_category": "clear",
    "holiday_category": "None"
  }'
```

Response (abridged):

```json
{
  "predicted_traffic_volume": 5775.9,
  "congestion_category": "Severe",
  "congestion_thresholds": {
    "low_max": 1211.0,
    "moderate_max": 4256.0,
    "high_max": 5502.0
  },
  "model_name": "random_forest",
  "disclaimer": "This is a historical-data-based estimate from a model trained on one Minnesota interstate sensor (2012-2018). It is NOT a live traffic measurement..."
}
```

### POST `/api/routes`

```bash
curl -X POST http://127.0.0.1:8000/api/routes \
  -H "Content-Type: application/json" \
  -d '{
    "origin":      {"latitude": 17.4388, "longitude": 78.3504},
    "destination": {"latitude": 17.4442, "longitude": 78.3537},
    "scenario": "simulated_peak",
    "severity": 1.0,
    "max_alternatives": 3
  }'
```

Response (abridged):

```json
{
  "scenario": {
    "name": "simulated_peak",
    "provenance": "simulated",
    "is_uniform_multiplier": false,
    "note": "Multipliers differ per road segment, so they can change the optimal path."
  },
  "routes": [
    {
      "rank": 1,
      "label": "shortest_distance",
      "distance_km": 1.346,
      "free_flow_time_minutes": 1.62,
      "congestion_aware_time_minutes": 2.07,
      "mean_congestion_multiplier": 1.267,
      "geometry": [[17.438826, 78.350458], [17.444233, 78.353719]]
    }
  ],
  "graph": {"nodes": 38, "edges": 63},
  "attribution": "Road data (c) OpenStreetMap contributors, ODbL."
}
```

**Supported `scenario` values**

| Value | Meaning | Edge-specific? |
| --- | --- | --- |
| `free_flow` | No congestion; multiplier 1.0 everywhere | No (uniform) |
| `simulated_peak` | Simulated peak congestion from OSM road class + edge identity | **Yes** |
| `custom` | Caller-supplied `highway_multipliers` and `edge_multipliers` | **Yes** |

> A uniform multiplier cannot change the ranking of paths by travel time — it
> scales every route equally. TrafficSense therefore never claims that
> `free_flow` changes the fastest route. Only `simulated_peak` and `custom`
> apply **per-segment** multipliers, and the API reports
> `is_uniform_multiplier` so the client can see which case it is.

---

## 13. Testing

### Backend

```powershell
cd backend
python -m pytest -q
```

73 tests cover preprocessing, feature engineering, chronological splitting,
training on a small fixture dataset, artifact loading, prediction response
structure, input validation, health and metrics endpoints, missing-model
behaviour, congestion threshold boundaries, route distance and cost maths,
shortest-path and alternative-route behaviour on a small synthetic graph,
disconnected routes, invalid coordinates, and road-network download failures.

**Unit tests never download road data.** They inject a small synthetic graph or a
temporary cache, and use `allow_download=False`.

### Frontend

```powershell
cd frontend
npm test        # Vitest component tests
npm run lint    # ESLint
npm run build   # production build
```

---

## 14. Configuration reference

Copy `.env.example` to `.env` and edit it, or set plain environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRAFFICSENSE_CORS_ORIGINS` | local dev origins | Comma-separated allowed frontend origins. **Never use `*` in production.** |
| `TRAFFICSENSE_ARTIFACTS_DIR` | `backend/artifacts` | Where the model, metrics and graph cache live. Point this at a persistent volume on ephemeral hosts. |
| `TRAFFICSENSE_STUDY_AREA_NAME` | `Gachibowli, Hyderabad (demo study area)` | Label shown for the study area. |
| `TRAFFICSENSE_STUDY_POINT` | `17.44,78.35` | `latitude,longitude` centre of the road-network study area. |
| `TRAFFICSENSE_STUDY_DIST` | `600` | Radius in metres around the study point. Keep it small. |
| `TRAFFICSENSE_NETWORK_TYPE` | `drive` | OSMnx network type (`drive`, `drive_service`, `walk`, …). |
| `TRAFFICSENSE_OFFLINE` | unset | When `1`, road-network downloads are disabled and only a cached graph is used. |
| `VITE_API_BASE_URL` | empty (same origin) | Backend base URL baked into the frontend build. |
| `VITE_PROXY_TARGET` | `http://127.0.0.1:8000` | Vite dev-server proxy target. |

---

## 15. Model artifacts and what to commit

`.gitignore` deliberately excludes generated artifacts and the dataset:

- `backend/data/*.csv` — third-party download, ~3 MB. Re-download it.
- `backend/artifacts/*.joblib` — tens of MB. Regenerate with `python -m app.train`.
- `backend/artifacts/osm_cache/` — regenerated on demand from OpenStreetMap.
- `backend/artifacts/*.png` — regenerated by training.
- `frontend/node_modules/`, `frontend/dist/` — regenerated by `npm install` / `npm run build`.

**Do commit** the source, `metrics.json` and `model_metadata.json` if you want a
record of a specific run in your submission, and everything under `reports/`.

**Never commit** `.env` files, credentials or keys.

---

## 16. Deployment

### Backend (any Python host: Render, Railway, Fly.io, Azure App Service, a VM)

1. Build command: `pip install -r backend/requirements.txt`
2. Start command:
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   run from the `backend` directory.
3. Set environment variables:
   - `TRAFFICSENSE_CORS_ORIGINS=https://your-frontend.example.com`
   - `TRAFFICSENSE_ARTIFACTS_DIR=/var/lib/trafficsense/artifacts`
4. **Model persistence.** The model file is generated, not committed. On a host
   with an ephemeral filesystem you must either:
   - run training as part of the build/release step, or
   - mount a persistent volume at `TRAFFICSENSE_ARTIFACTS_DIR`, or
   - host the artifact in object storage and download it at startup.

   Without one of these, the service starts and reports
   `model_available: false`; `/api/predict` returns **503** with a clear message
   instead of a fabricated number.

### Frontend (any static host: Netlify, Vercel, GitHub Pages, S3 + CloudFront)

1. Build command: `npm ci && npm run build` (run in `frontend/`)
2. Publish directory: `frontend/dist`
3. Set `VITE_API_BASE_URL=https://your-backend.example.com` **at build time**.
4. Add the frontend origin to `TRAFFICSENSE_CORS_ORIGINS` on the backend.

---

## 17. Limitations

1. **One sensor, one highway.** The model learns the traffic pattern of a single
   Minnesota interstate sensor (2012–2018). It is not a general model for
   Hyderabad or any other city.
2. **Historical, not live.** Predictions describe the historical pattern for a
   given hour and weather condition. They are not a live traffic measurement.
3. **Simulated congestion.** Per-segment congestion multipliers in the default
   route scenario are **simulated** for demonstration. They are not measured
   traffic counts. The API labels this as `provenance: "simulated"` and the UI
   shows a warning banner.
4. **Model not used for per-road congestion.** The volume model is not applied
   road-by-road; doing so would be unjustified because it was never trained on
   those roads.
5. **Speed limits.** Travel times use OpenStreetMap speed limits, which may be
   missing or out of date. Missing values fall back to per-class defaults.
6. **Snapping.** Origin and destination snap to the nearest road node, so a route
   starts at the closest road rather than the exact point.
7. **Graph coverage.** Only the configured study area is loaded. Points outside
   it may snap to a distant node or have no connecting path.
8. **Overpass API availability.** Road-network download depends on the public
   Overpass API, which can be slow or rate-limited. TrafficSense caches the graph
   and returns a clear 503 when the network is unavailable.

---

## 18. Future scope

1. Integrate a real, road-specific traffic data source (for example an official
   city feed or a probe-vehicle dataset) to replace the simulated congestion
   multipliers with observed ones.
2. Train per-corridor or per-city models once multi-location data is available.
3. Add travel-time prediction with confidence intervals.
4. Support time-dependent routing (departure-time-aware costs).
5. Add geocoding and address search to the route planner.
6. Stream live traffic into the dashboard and show observed versus predicted.
7. Deploy the model behind a feature store and monitor for drift.

---

## 19. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `/api/predict` returns 503 | Model artifact missing | Run `python -m app.train` in `backend/` |
| `/api/routes` returns 503, "Could not download the road network" | Overpass API unreachable or rate-limited | Retry later, reduce `TRAFFICSENSE_STUDY_DIST`, or run with `TRAFFICSENSE_OFFLINE=1` against a pre-cached graph |
| `/api/routes` returns 404, "No road route connects" | Origin and destination are in different graph components or outside the study area | Choose points inside the study area and closer together |
| Frontend shows "Backend unreachable" | Backend not running or wrong URL | Start uvicorn and check `VITE_API_BASE_URL` / `VITE_PROXY_TARGET` |
| `Activate.ps1` cannot be loaded | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Training fails with "dataset not found" | CSV not placed in `backend/data/` | Follow section 6 |
| Training is slow | Random Forest on 40k rows | Expected; ~10–30 s. Reduce `n_estimators` in `app/train.py` if needed |

---

## 20. Academic deliverables

| File | Contents |
| --- | --- |
| `reports/final-report.md` | Full project report (abstract → references) |
| `reports/viva-questions.md` | 25+ viva questions with answers |
| `reports/figures/` | Generated figures |

---

## Licence and attribution

- **Traffic dataset:** Metro Interstate Traffic Volume, UCI Machine Learning
  Repository, CC BY 4.0.
- **Road data:** © OpenStreetMap contributors, available under the Open Database
  Licence (ODbL). Map tiles are served by the OpenStreetMap tile servers under
  their [tile usage policy](https://operations.osmfoundation.org/policies/tiles/).
