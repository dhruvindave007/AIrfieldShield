from datetime import timedelta

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APIClient

from alerts.models import Alert
from core.models import Prediction, WeatherObservation
from ingestion.services import parse_iem_csv, store_observations

pytestmark = pytest.mark.django_db


@pytest.fixture
def client():
    return APIClient()


def test_dashboard_requires_known_airfield(client, airfield):
    assert client.get("/api/frontend/dashboard/").status_code == 400
    assert client.get("/api/frontend/dashboard/?airfield=XXXX").status_code == 404


def test_dashboard_returns_real_observation_not_random(client, airfield, add_obs):
    add_obs(minutes_ago=10, temperature_c=31.5, pressure_hpa=1002.0)
    first = client.get("/api/frontend/dashboard/?airfield=vecc").json()
    second = client.get("/api/frontend/dashboard/?airfield=VECC").json()
    assert first["weatherData"]["temperature"] == 31.5
    assert first["weatherData"] == second["weatherData"]        # deterministic
    assert first["stormCells"] is None                          # nothing simulated
    status = {s["name"].split(" (")[0]: s["status"] for s in first["dataSources"]}
    assert status["METAR observations"] == "OK"
    assert status["Nowcast model"] == "DOWN"
    assert status["Weather radar"] == "NOT CONNECTED"


def test_dashboard_hides_expired_and_acknowledged_alerts(client, airfield):
    now = timezone.now()
    Alert.objects.create(airfield=airfield, alert_type="Thunderstorm", level="RED", message="a",
                         valid_until=now + timedelta(hours=1))
    Alert.objects.create(airfield=airfield, alert_type="Thunderstorm", level="RED", message="old",
                         valid_until=now - timedelta(hours=1))
    Alert.objects.create(airfield=airfield, alert_type="StrongWind", level="RED", message="ack",
                         valid_until=now + timedelta(hours=1), acknowledged=True)
    data = client.get("/api/frontend/dashboard/?airfield=VECC").json()
    assert [a["message"] for a in data["alerts"]] == ["a"]


def test_prediction_history(client, airfield):
    Prediction.objects.create(airfield=airfield, horizon_minutes=60, thunderstorm_prob=0.2, gale_wind_prob=0.1)
    Prediction.objects.create(airfield=airfield, horizon_minutes=180, thunderstorm_prob=0.4, gale_wind_prob=0.1)
    r = client.get("/api/predictions/history/?airfield=VECC&hours=6")
    assert r.status_code == 200
    assert [h["thunderstorm_prob"] for h in r.json()["history"]] == [0.2]
    r = client.get("/api/predictions/history/?airfield=VECC&horizon=180")
    assert [h["thunderstorm_prob"] for h in r.json()["history"]] == [0.4]
    # garbage parameters fall back to defaults instead of raising a 500
    assert client.get("/api/predictions/history/?airfield=VECC&hours=abc").status_code == 200
    assert client.get("/api/predictions/history/").status_code == 400


def test_latest_alerts_and_predictions_endpoints(client, airfield):
    Alert.objects.create(airfield=airfield, alert_type="Thunderstorm", level="ORANGE", message="m",
                         valid_until=timezone.now() + timedelta(hours=1))
    Prediction.objects.create(airfield=airfield, thunderstorm_prob=0.3, gale_wind_prob=0.0)
    r = client.get("/api/alerts/latest/?limit=notanumber")
    assert r.status_code == 200 and len(r.json()) == 1
    r = client.get("/api/alerts/latest-predictions/?airfield=VECC")
    assert r.status_code == 200 and len(r.json()["predictions"]) == 1


def test_acknowledge_requires_authentication(client, airfield):
    alert = Alert.objects.create(airfield=airfield, alert_type="Thunderstorm", level="RED", message="m")
    assert client.post(f"/api/alerts/ack/{alert.pk}/").status_code in (401, 403)
    alert.refresh_from_db()
    assert alert.acknowledged is False

    user = get_user_model().objects.create_user("operator", password="x")
    client.force_authenticate(user)
    assert client.post(f"/api/alerts/ack/{alert.pk}/").status_code == 200
    alert.refresh_from_db()
    assert alert.acknowledged and alert.acknowledged_by == "operator"


def test_anonymous_cannot_write(client, airfield):
    assert client.post("/api/airfields/", {"name": "X", "icao": "XXXX"}).status_code in (401, 403)
    assert client.post("/api/predictions/", {}).status_code in (401, 403, 405)
    assert client.get("/api/airfields/").status_code == 200


IEM_CSV = """station,valid,tmpf,dwpf,relh,drct,sknt,gust,alti,vsby,metar
VECC,2024-05-06 14:30,86.00,77.00,74.50,150.00,8.00,M,29.65,2.49,VECC 061430Z 15008KT 4000 HZ SCT018 FEW025CB BKN100 30/25 Q1004 TEMPO 30030G40KT 0800 TSRA
VECC,2024-05-06 15:00,69.80,69.80,100.00,150.00,30.00,40.00,29.65,0.50,VECC 061500Z 15030G40KT 0800 +TSRA BKN015 FEW025CB OVC080 21/21 Q1004 NOSIG
"""


def test_ingestion_parses_and_upserts_idempotently(airfield):
    obs = parse_iem_csv(IEM_CSV)
    assert store_observations(airfield, obs) == 2
    assert store_observations(airfield, obs) == 0          # second run creates nothing
    assert WeatherObservation.objects.count() == 2
    first, second = WeatherObservation.objects.order_by("timestamp")
    assert first.thunderstorm is False and first.trend_thunderstorm is True
    assert first.cumulonimbus is True and first.wind_gust_ms is None
    assert second.thunderstorm is True and second.precipitation is True
    assert second.temperature_c == pytest.approx(21.0)
    assert second.wind_gust_ms == pytest.approx(40 * 0.514444)
    assert second.pressure_hpa == pytest.approx(29.65 * 33.8639)
    assert second.raw_payload["metar"].startswith("VECC 061500Z")
    assert second.timestamp.tzinfo is not None


def test_ingestion_rejects_error_pages():
    with pytest.raises(ValueError):
        parse_iem_csv("<html>Service unavailable</html>")
