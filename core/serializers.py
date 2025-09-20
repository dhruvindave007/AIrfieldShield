# core/serializers.py
from rest_framework import serializers
from .models import Airfield, WeatherObservation, Prediction

class AirfieldSerializer(serializers.ModelSerializer):
    class Meta:
        model = Airfield
        fields = ["id", "name", "icao", "latitude", "longitude"]

class WeatherObservationSerializer(serializers.ModelSerializer):
    station_id = serializers.CharField(source="station.station_id", read_only=True)
    class Meta:
        model = WeatherObservation
        fields = [
            "id", "station_id", "timestamp",
            "temperature_c", "humidity", "pressure_hpa",
            "wind_speed_ms", "wind_gust_ms", "wind_dir_deg", "raw_payload"
        ]

class PredictionSerializer(serializers.ModelSerializer):
    airfield = AirfieldSerializer(read_only=True)
    class Meta:
        model = Prediction
        fields = [
            "id", "airfield",
            "thunderstorm_prob", "gale_wind_prob",
            "confidence", "details", "created_at"
        ]
