from django.db import models

class Airfield(models.Model):
    name = models.CharField(max_length=200)
    icao = models.CharField(max_length=100, unique=True, null=True, blank=True)

    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    elevation_m = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.icao})" if self.icao else self.name


class WeatherStation(models.Model):
    airfield = models.ForeignKey(Airfield, on_delete=models.CASCADE, related_name="stations")
    station_id = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()

    def __str__(self):
        return f"{self.station_id} @ {self.airfield}"


class WeatherObservation(models.Model):
    station = models.ForeignKey(WeatherStation, on_delete=models.CASCADE, related_name="observations")
    timestamp = models.DateTimeField(db_index=True)
    temperature_c = models.FloatField(null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True)
    pressure_hpa = models.FloatField(null=True, blank=True)
    wind_speed_ms = models.FloatField(null=True, blank=True)
    wind_gust_ms = models.FloatField(null=True, blank=True)
    wind_dir_deg = models.FloatField(null=True, blank=True)
    raw_payload = models.JSONField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["timestamp"]),
        ]

    def __str__(self):
        return f"{self.station} @ {self.timestamp}"


class Prediction(models.Model):
    airfield = models.ForeignKey(Airfield, on_delete=models.CASCADE, related_name="predictions")
    created_at = models.DateTimeField(auto_now_add=True)
    horizon_minutes = models.IntegerField(default=60)
    thunderstorm_prob = models.FloatField(null=True, blank=True)
    gale_wind_prob = models.FloatField(null=True, blank=True)
    details = models.JSONField(null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Pred {self.airfield} +{self.horizon_minutes}m @ {self.created_at}"
