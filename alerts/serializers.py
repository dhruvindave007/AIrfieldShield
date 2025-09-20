# alerts/serializers.py
from rest_framework import serializers
from .models import Alert
from core.serializers import AirfieldSerializer
from core.models import Prediction

class AlertSerializer(serializers.ModelSerializer):
    airfield = AirfieldSerializer(read_only=True)

    class Meta:
        model = Alert
        fields = (
            "id",
            "airfield",
            "alert_type",
            "message",
            "level",
            "created_at",
            "valid_until",
            "meta",
            "acknowledged",
            "acknowledged_by",
            "acknowledged_at",
        )
        read_only_fields = ("created_at", "acknowledged_at")

# Brief serializer re-used from core for Predictions
class PredictionBriefSerializer(serializers.ModelSerializer):
    airfield = AirfieldSerializer(read_only=True)
    class Meta:
        model = Prediction
        fields = ("id", "airfield", "created_at", "horizon_minutes", "thunderstorm_prob", "gale_wind_prob", "confidence")
