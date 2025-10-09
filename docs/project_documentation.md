Airfield Weather Dashboard – Project Documentation
1. Introduction

The Airfield Weather Dashboard is an AI-driven prototype designed for nowcasting, prediction, and visualization of severe weather events around airfields.
It simulates real-time integration of weather sensors, radar feeds, and AI/ML models, making weather data actionable for aviation safety.

The system provides:

Live monitoring of storm activity on an interactive map.

AI-based predictions of thunderstorm and gale wind probabilities.

Active alerting system with severity levels.

Data sources overview for transparency.

Historical trend visualization to analyze risk over time.

2. Motivation

Aviation safety depends heavily on accurate, timely weather information.

Current issue: Many airports lack unified dashboards that combine AI predictions, live radar/sensor data, and automated alerts.

Our solution: Provide a single-pane dashboard where operators can view forecasts, risk levels, and alerts in real time — even when data comes from multiple heterogeneous sources.

3. System Architecture
3.1 High-level Flow

Data Sources (simulated):

Weather stations (temperature, humidity, pressure, wind).

Radar feed (synthetic storm cells with intensity + movement vectors).

Satellite-like “radar intensity” field.

AI prediction engine.

AI/ML Layer

Ensemble Models trained on synthetic weather data:

Random Forest (tabular) → aggregates mean/max/min/std features.

LSTM (time-series) → captures temporal patterns.

CNN (radar-like images) → interprets spatial intensity maps.

Meta-learner (Logistic Regression) → combines outputs into final risk.

Models predict probabilities of:

Thunderstorm occurrence.

Gale-force winds.

Backend (Django + DRF)

REST APIs:

/api/frontend/dashboard/ → live dashboard payload (alerts, predictions, weather, storms).

/api/predictions/history/ → historical risk data for charts.

/api/airfields/ → airfield metadata.

Management commands (one-click pipeline):

train_ensemble → trains ensemble models on synthetic data.

predict → runs AI predictions for each airfield.

pipeline → executes full workflow (training → prediction → alerts).

sanitize_predictions → ensures JSON-serializable DB records.

Database schema:

Airfield → airfield metadata (name, ICAO, lat/lon, elevation).

WeatherStation → simulated stations linked to airfields.

WeatherObservation → time-series sensor data.

Prediction → AI outputs (with confidence + details).

Alert → real-time weather hazard notifications.

Frontend (HTML + Tailwind + Leaflet + Chart.js)

Interactive storm activity map with live storm cells (animated movement vectors).

Alerts panel showing active warnings (severity levels: RED, ORANGE, GREEN).

Prediction cards with probability progress bars + confidence + horizon.

Data sources widget showing availability/health of inputs.

Weather summary (temperature, humidity, pressure, wind, radar intensity).

Risk Trend chart (6-hour probability evolution of thunderstorm/gale risks):

Zoomable + pannable (chartjs-plugin-zoom).

Tooltips show probabilities and supporting technical details.

4. Technical Details
4.1 AI Models

Random Forest Classifiers (Sklearn)

Input: Tabular aggregates of temperature, humidity, pressure, wind, gust.

Output: Binary probabilities for thunderstorm/gale.

LSTM (Keras/TensorFlow)

Input: Weather sequences of length 30.

Learns temporal dependencies (e.g., pressure drops → storms).

CNN (Keras/TensorFlow)

Input: 32×32 radar-like intensity images generated from humidity + gust.

Learns spatial weather patterns (cloud/storm density).

Meta-learner (Logistic Regression)

Input: Predictions from RF, LSTM, CNN.

Output: Final calibrated risk probability.

Training Data: Synthetic but realistic (8,000+ samples).

Accuracy (AUC scores):

RF Gale: ~0.95

RF Thunder: ~0.51 (baseline).

LSTM Thunder: ~0.40–0.50 (needs more data/epochs).

CNN Thunder: ~0.46–0.50.

Meta-Learner: ~0.51–0.55.

⚠️ Note: Current numbers reflect synthetic dataset limitations, but pipeline is fully functional.

4.2 API Endpoints

/api/frontend/dashboard/?airfield=ICAO
Returns live dashboard data:

{
  "airfield": {"name":"Test Field","icao":"TEST","lat":23.02,"lon":72.57},
  "alerts": [...],
  "predictions": [...],
  "weatherData": {...},
  "stormCells": {...},
  "dataSources": [...]
}


/api/predictions/history/?airfield=ICAO&hours=6
Returns last 6h of predictions for chart rendering.

4.3 Frontend Features

Storm Activity Map

Leaflet-based.

Thunderstorm (purple), Gale (orange), RainShower (blue).

Storms animated according to speed/direction vectors.

Popups show type, intensity, movement.

Risk Trend Chart

Multi-line chart: Thunderstorm % (red), Gale % (orange).

Interactive zoom/pan.

Tooltips show full technical context (temp, humidity, pressure, gust, radar, confidence).

Alerting System

Severity levels color-coded (RED, ORANGE, GREEN).

Sorted by creation time.

Linked to prediction engine + thresholds.

5. Demonstration Workflow

Start Django server:

python manage.py runserver


Run pipeline (train + predict + refresh alerts):

python manage.py pipeline --samples 8000 --seq-len 30 --epochs 10 --batch 64


Open Dashboard:
Navigate to http://127.0.0.1:8000/dashboard/

Switch airfields via dropdown.

Watch storm activity animation.

Explore risk chart with zoom/pan.

Review alerts + predictions + weather summary.

6. Key Innovations

Unified AI/ML + visualization pipeline inside Django.

Interactive storm overlays (movement + intensity simulation).

AI ensemble approach (RF + LSTM + CNN + meta-learner).

Data sanitization pipeline ensures reliable DB storage.

One-command orchestration: pipeline automates full cycle.

Highly interactive dashboard (map, charts, real-time refresh).

7. Limitations & Future Work

Current dataset is synthetic; integration with real sensors/radar would improve accuracy.

LSTM/CNN accuracy limited due to small synthetic dataset. Future work → larger, more realistic datasets.

Extend alert types beyond thunderstorm/gale (e.g., hail, visibility, turbulence).

Deploy pipeline with real-time streaming data (Kafka/MQTT).

Add export features (CSV/PNG) for reporting.

8. Conclusion

This prototype demonstrates a real, functioning AI/ML system capable of simulating real-time weather nowcasting at airfields.
It integrates data ingestion, AI ensemble modeling, database persistence, and frontend visualization into a single cohesive platform, showcasing both technical engineering depth and applied AI innovation for aviation safety.