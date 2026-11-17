from django.contrib import admin

from .models import Airfield, Prediction, WeatherObservation, WeatherStation

admin.site.register(Airfield)
admin.site.register(WeatherStation)


@admin.register(WeatherObservation)
class WeatherObservationAdmin(admin.ModelAdmin):
    list_display = ("station", "timestamp", "temperature_c", "pressure_hpa", "wind_speed_ms", "thunderstorm")
    list_filter = ("station", "thunderstorm")
    date_hierarchy = "timestamp"


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("airfield", "created_at", "horizon_minutes", "thunderstorm_prob", "gale_wind_prob", "model_version")
    list_filter = ("airfield", "horizon_minutes")
