from django.db import models
from core.models import Airfield

class Alert(models.Model):
    airfield = models.ForeignKey(Airfield, on_delete=models.CASCADE, related_name="alerts")
    alert_type = models.CharField(max_length=100)  # e.g., 'Thunderstorm', 'GaleWind'
    message = models.TextField()
    level = models.CharField(max_length=20)  # e.g., GREEN/YELLOW/RED
    created_at = models.DateTimeField(auto_now_add=True)
    valid_until = models.DateTimeField(null=True, blank=True)
    meta = models.JSONField(null=True, blank=True)

    # --- new acknowledgement fields ---
    acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.CharField(max_length=200, null=True, blank=True)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    # ------------------------------------

    def __str__(self):
        return f"{self.level} {self.alert_type} @ {self.airfield}"
