# TrafficSense — Viva Questions and Answers

A study guide for the project viva. Answers are concise but technically accurate,
and they refer to what this project actually does.

---

## Part A — Regression and classification

**Q1. What is the difference between regression and classification? Which does
this project use for the main task?**

Regression predicts a continuous numeric value; classification predicts a
discrete label. TrafficSense's main task is **regression**: it predicts hourly
`traffic_volume` in vehicles per hour. It additionally performs
**classification** as a derived task, mapping the predicted (and actual) volume
into one of four congestion categories. The classification is derived from the
same numeric prediction using fixed thresholds, so the two outputs are always
consistent with each other.

**Q2. Why is the congestion category derived from thresholds rather than
predicted by a separate classifier?**

Two reasons. First, a single model produces both outputs, which keeps the system
simple and guarantees consistency between the number and the label. Second, the
categories are an ordinal transformation of the volume, so training a second
model would only add an extra source of error. The thresholds themselves come
from the training distribution, so the categories reflect the data rather than a
guess.

**Q3. What are the four congestion categories and how are the thresholds
derived?**

Low (≤ 1,211 veh/h), Moderate (1,211–4,256), High (4,256–5,502) and Severe
(> 5,502). The thresholds are the **25th, 60th and 85th percentiles of the
training volume distribution**, computed from the **training split only**.
Deriving them from the full dataset would leak test information into the model's
decision boundary.

**Q4. Why must thresholds be derived from training data only?**

If thresholds were computed on the whole dataset, they would encode information
from the test period. The reported classification metrics would then be
optimistic. Deriving them from training data mirrors how the model is fitted:
everything that shapes the model's behaviour must come from training data.

---

## Part B — Random Forest

**Q5. Explain a Random Forest in one paragraph.**

A Random Forest is an ensemble of decision trees. Each tree is trained on a
bootstrap sample of the training rows (sampling with replacement) and, at every
split, considers only a random subset of features. Averaging many such
decorrelated trees reduces variance dramatically compared with a single deep
tree, which is why a forest generalises much better. Because the trees are
trained independently, the algorithm parallelises well.

**Q6. Why did you choose Random Forest for this problem?**

Three reasons. Traffic volume depends on non-linear interactions — the effect of
the hour is different on a weekend than on a weekday — which trees capture
naturally. The features are tabular and mixed (numeric plus categorical), which is
exactly the setting where tree ensembles excel. Finally, Random Forest reports
feature importance, which supports interpretation and the report.

**Q7. How does a Random Forest avoid overfitting compared with a single decision
tree?**

Two sources of randomness: bagging (each tree sees a different bootstrap sample)
and feature subsampling (each split sees a random subset of features). Together
they decorrelate the trees, so their errors partially cancel when averaged.
A single unpruned tree can memorise the training data and has high variance; the
forest has much lower variance.

**Q8. What is feature importance and what did it show here?**

Feature importance measures how much each feature reduces impurity across all
trees, normalised to sum to one. In TrafficSense the dominant features are
`hour_cos` (0.657), `hour` (0.138) and `hour_sin` (0.054). This confirms the
strong diurnal cycle of traffic and shows that weather contributes much less at
this single sensor.

**Q9. What are the main hyperparameters of the Random Forest you used?**

`n_estimators=220` (number of trees), `min_samples_leaf=2` (regularisation) and `random_state=42` (reproducibility).
`n_estimators` is the main accuracy/runtime trade-off: more trees give a more
stable prediction at a roughly linear increase in training time.

---

## Part C — Data leakage and validation

**Q10. What is data leakage, and give an example you avoided.**

Leakage is when information that would not be available at prediction time
influences training, producing optimistic results. Two examples avoided here:
(i) imputation and one-hot encoding are fitted **inside** the scikit-learn
pipeline on training folds only, never on the whole dataset; and (ii) no lagged or
rolling `traffic_volume` value is used as a predictor, so the model cannot see
the value it is asked to predict.

**Q11. Why is a random train/test split wrong for time-series-like traffic
data?**

A random split lets the model train on Tuesday 10:00 and test on Tuesday 11:00.
Neighbouring hours are highly correlated, so the model effectively sees the
answer, and the test score is inflated. Time must be respected.

**Q12. Describe your splitting strategy.**

A chronological 70/15/15 split with no shuffling. The first 70% of records by
timestamp form the training set (2012-10-02 to 2017-05-10, 28,403 rows), the next
15% form the validation set (6,086 rows), and the final 15% form the untouched
test set (2018-01-19 to 2018-09-30, 6,086 rows). The model is selected on
validation, refitted on train + validation, and evaluated once on test.

**Q13. Why did you keep a separate validation set instead of tuning on the test
set?**

If the test set were used to choose the model, it would no longer be an unbiased
estimate of generalisation. Using a separate validation set for selection keeps
the test set as a single, honest measurement.

**Q14. Why is the baseline R² approximately zero?**

The baseline predicts the **training** mean for every test row. The test period
has a different mean volume, so the baseline has no explanatory power on it and
R² comes out at about zero (−0.0015). That is exactly what a baseline should
look like, and it shows why the Random Forest's 0.9507 is meaningful.

---

## Part D — Metrics

**Q15. What is MAE and what does your value mean in plain language?**

Mean Absolute Error is the average absolute difference between predicted and
actual volume. TrafficSense achieved **MAE = 250.63 vehicles/hour** on the test
period, meaning predictions are on average about 251 vehicles/hour away from the
actual hourly count.

**Q16. What is RMSE and how does it differ from MAE?**

Root Mean Squared Error is the square root of the average squared error. Because
errors are squared before averaging, RMSE penalises large errors more than MAE
does. RMSE is always ≥ MAE; here RMSE = 439.62 versus MAE = 250.63, which
indicates the presence of some large individual errors.

**Q17. What is R² and how do you interpret 0.9507?**

R² is the proportion of the variance in the target explained by the model,
relative to always predicting the mean. A value of 0.9507 means the model explains
about 95% of the variance in test-period traffic volume. R² can be negative when
a model is worse than the mean predictor, as seen with the baseline.

**Q18. Why report macro F1 rather than accuracy for the congestion categories?**

The categories are imbalanced (Severe has 908 test cases versus Moderate's 2,065).
Plain accuracy would be dominated by the larger classes. Macro F1 averages the
per-class F1 scores with equal weight, so a model that neglects a minority class
is penalised. TrafficSense reports macro F1 = 0.8751.

**Q19. How do you read the confusion matrix?**

Rows are actual categories, columns are predicted. The diagonal holds correct
predictions. In our matrix, 1,368 of 1,461 actual "Low" rows were classified
correctly, while errors cluster between adjacent categories — for example 168
actual "High" rows were predicted "Moderate". Off-diagonal mass appears almost
only next to the diagonal, which is the expected pattern for an ordinal label.

**Q20. Why not just report the highest accuracy you observed?**

Because accuracy without a baseline, without a proper split and without stating
the data's geographic scope is meaningless. Reporting only a peak number is a
common flaw in student projects. TrafficSense reports a baseline comparison, a
chronological holdout, multiple metrics and the dataset's limits.

---

## Part E — Feature engineering

**Q21. Why encode the hour cyclically with sine and cosine?**

As integers, hour 23 and hour 0 look maximally distant even though they are one
hour apart. Mapping the hour onto the unit circle with `sin(2πh/24)` and
`cos(2πh/24)` places them adjacently, so the model can learn the wrap-around
continuity of the daily cycle. The same reasoning applies to day of week and
month.

**Q22. Why keep both `hour` and `hour_sin`/`hour_cos`?**

The cyclical pair encodes continuity; the raw integer gives the tree a
convenient monotone split point. The feature importance results show the model
uses both: `hour_cos` at 0.657 and `hour` at 0.138.

**Q23. Why did you drop `weather_description`?**

It is a finer-grained duplicate of `weather_main` (for example "light rain"
versus "rain"). Keeping both inflates the feature space and splits the same
signal across two highly correlated categorical variables, which adds noise to
tree splits without adding information.

**Q24. Why median imputation for numeric and most-frequent for categorical?**

The median is robust to outliers and preserves the central tendency of a numeric
feature, which is safer than the mean when a sensor occasionally records extreme
values. Categorical features cannot be averaged, so the most frequent category is
the natural fallback. Both imputers are fitted inside the pipeline, so the fill
values come from training data only.

---

## Part F — Backend, FastAPI and React

**Q25. Why use a scikit-learn Pipeline rather than preprocessing manually?**

A `Pipeline` binds preprocessing and the estimator into one object. It guarantees
that the same transformations are applied at training and inference time, in the
same order, and that imputers and encoders are fitted only on training data.
Saving one pipeline with Joblib also means the API cannot accidentally use a
different feature order than training.

**Q26. How does the API guarantee that inference uses the same feature order as
training?**

`_feature_row()` in `app/main.py` builds a one-row DataFrame using the feature
list saved in `model_metadata.json`, which is the same list the pipeline was
trained on. The pipeline then applies the fitted transformations. Column order is
therefore fixed by the saved metadata, not by request order.

**Q27. How do you validate inputs in FastAPI?**

With Pydantic models. For example `hour` is constrained to 0–23, `day_of_week` to
0–6, `month` to 1–12, `cloud_cover` to 0–100, and coordinates to valid latitude
and longitude ranges. Invalid input is rejected with HTTP 422 and a message
explaining the constraint, before any model code runs.

**Q28. What happens if the model artifact is missing?**

The health endpoint reports `model_available: false`, and `/api/predict` returns
HTTP 503 with a message telling the user to train the model. The API never
fabricates a prediction.

**Q29. Why is CORS configured from an environment variable instead of a
wildcard?**

A wildcard (`*`) would let any website call the API from a user's browser, which
is unsafe in production. TrafficSense reads allowed origins from
`TRAFFICSENSE_CORS_ORIGINS` and defaults to local development origins only.

**Q30. Why does the frontend never display a prediction when the request fails?**

Because a stale or invented success is worse than an error. On failure the
component sets an error message and explicitly clears the prediction state, so
the result card stays in its empty state. The user sees the real problem instead
of a misleading number.

---

## Part G — Graph theory, routing and algorithms

**Q31. How is the road network represented as a graph?**

As a directed multigraph (`networkx.MultiDiGraph`). Nodes are road junctions or
shape points with latitude/longitude attributes. Edges are road segments carrying
attributes: `length_m`, `speed_kph` and `travel_time_s`. It is directed because
one-way streets exist, and a multigraph because two nodes can be joined by more
than one distinct segment.

**Q32. Why a multigraph rather than a simple graph?**

Between the same pair of nodes there can be several physically distinct road
segments (a service road beside a highway, or dual carriageways). A simple graph
would force you to discard all but one edge and lose real information. The
multigraph keeps every segment, and the routing code selects the best edge
between two nodes when computing cost.

**Q33. Explain Dijkstra's algorithm and its complexity.**

Dijkstra's algorithm finds shortest paths from a source to all nodes when edge
weights are non-negative. It repeatedly extracts the node with the smallest
tentative distance (using a priority queue) and relaxes its outgoing edges. With
a binary heap the complexity is O((V + E) log V). NetworkX uses Dijkstra when
`weight` is supplied and the graph has non-negative weights.

**Q34. Why can't you use Dijkstra with negative edge weights?**

Dijkstra assumes that once a node is extracted with the smallest tentative
distance, that distance is final. A negative edge could later provide a shorter
path to an already-finalised node, breaking the assumption. Bellman-Ford handles
negative weights instead. Road lengths and travel times are non-negative, so
Dijkstra is appropriate here.

**Q35. What is your route cost function?**

For a path `P` of edges `e`:

```
cost(P) = Σ_e  travel_time_s(e) × multiplier(e)
```

with `travel_time_s(e) = length_m(e) / (speed_kph(e) / 3.6)`. Distance is
`Σ length_m(e)` and free-flow time is `Σ travel_time_s(e)`. Each is used as a
NetworkX edge weight to produce a different optimal path.

**Q36. Why is a single global congestion multiplier useless for route
optimisation?**

Because it multiplies every route by the same constant. If route A is faster than
route B before the multiplier, it remains faster after it. The ranking cannot
change, so a uniform multiplier can never alter the optimal path. This is why
TrafficSense uses **edge-specific** multipliers and reports
`is_uniform_multiplier` so the client can see which case applies.

**Q37. How does the simulated congestion scenario assign per-edge multipliers?**

The multiplier is `1 + severity × (class_weight − 1) × jitter`, where
`class_weight` depends on the OSM highway class and `jitter ∈ [0.6, 1.4]` is
derived from a hash of the edge identity `(u, v, k)`. It is deterministic, stable
across runs, and — critically — different for different edges, so it can change
which path is optimal. It is explicitly labelled a simulation.

**Q38. How do you find alternative routes?**

With `nx.shortest_simple_paths`, which enumerates distinct simple paths in
increasing order of cost. TrafficSense requests a few more candidates than the
caller asked for, removes duplicates by node sequence, and returns up to
`max_alternatives`. It also always includes the distance-optimal, free-flow-optimal
and congestion-aware-optimal paths when they differ.

**Q39. What happens when origin and destination are disconnected?**

`nx.shortest_path` raises `NetworkXNoPath`. TrafficSense catches it and returns
HTTP 404 with a message explaining that no road route connects the points and
suggesting points closer together or inside the study area. It never crashes.

**Q40. How do you handle a road segment with no speed limit in the data?**

`_edge_speed_kph()` reads the OSM `maxspeed` tag when present and otherwise falls
back to a per-highway-class default (for example motorway 100 km/h, residential
30 km/h). Missing attributes therefore degrade gracefully rather than raising.

**Q41. Why snap coordinates to the nearest road node?**

Coordinates supplied by a user generally do not lie exactly on a road. Routing
needs graph nodes as start and end points. The system finds the nearest node by a
linear scan (avoiding a scipy KD-tree dependency) and reports the snapped node in
`origin_snap` / `destination_snap` so the user can see the adjustment.

**Q42. Why cache the road graph?**

Downloading a road network from Overpass is slow and puts load on a public,
shared service. `GraphCache` loads the graph once per study area, stores it as a
pickle keyed by a hash of the area parameters, and reuses it across requests. A
corrupt cache file is detected and rebuilt.

**Q43. What is OpenStreetMap, and what are the usage obligations?**

OpenStreetMap is a collaboratively built, openly licensed map of the world.
Road data is available under the Open Database Licence (ODbL), which requires
attribution to "OpenStreetMap contributors". Map tiles served by the OSM tile
servers are subject to the OSMF tile usage policy, which requires a valid
attribution and forbids heavy automated use. TrafficSense displays the required
attribution on the map and caches road data rather than re-downloading it.

---

## Part H — Model limitations, honesty and deployment

**Q44. Can this model predict traffic in Hyderabad?**

No. The model was trained on a single Minnesota interstate sensor. The traffic
patterns it learned reflect that corridor's commuting behaviour and weather. It
has no knowledge of Hyderabad's roads, population or driving patterns. The
application states this limitation in the API response, in the dashboard and in
the report. The route planner's study area is configurable, but the **model** is
not a city-agnostic traffic model.

**Q45. Is the congestion data in the route planner real?**

No. The default scenario is a **simulated** peak congestion scenario, labelled
`provenance: "simulated"` in the API and shown as a warning in the UI. The API
also supports a `custom` scenario where a caller can supply real per-road or
per-class multipliers — that is the integration point for genuine traffic data.

**Q46. Why not use the volume model to set congestion on every road?**

Because the model was never trained on those roads. Applying it to arbitrary road
segments would be an unsupported extrapolation and would give the false
impression that the output is road-specific measurement. TrafficSense refuses to
do this and says so explicitly in the route response's limitations list.

**Q47. Why do travel times sometimes look short?**

Because they use OpenStreetMap speed limits, which may be optimistic or missing.
When the tag is missing, a per-class default is used. The results are estimates,
and the API labels them as such rather than as measurements.

**Q48. How would you deploy this, and what is the artifact-persistence trap?**

Backend on any Python host (Render, Railway, Fly.io, a VM) running
`uvicorn app.main:app`; frontend as static files on Netlify, Vercel or S3 with
`VITE_API_BASE_URL` set at build time. The trap is that the trained model is a
generated file. On a host with an ephemeral filesystem it disappears on redeploy,
and the API then reports `model_available: false` and returns 503. The fix is to
train during the build/release step, mount a persistent volume at
`TRAFFICSENSE_ARTIFACTS_DIR`, or fetch the artifact from object storage at
startup.

**Q49. Why is a baseline model necessary in a report?**

Without a baseline you cannot tell whether the model learned anything. The
training-mean baseline gives the honest reference point: MAE 1,735 versus the
Random Forest's 251 shows the model captures real structure. A report that gives
only a model's accuracy, with no baseline, is uninterpretable.

**Q50. What would you do differently with more time?**

Replace the simulated congestion multipliers with observed road-specific traffic
counts; train per-corridor models on multi-location data; add departure-time-aware
(time-dependent) routing; add confidence intervals to predictions; and add
geocoding so users can type addresses instead of coordinates.
