from django.db import models

class AIModel(models.Model):
    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=100)  # e.g., "LSTM", "CNN", "RandomForest"
    version = models.CharField(max_length=50, default="v1")
    trained_at = models.DateTimeField(null=True, blank=True)
    metrics = models.JSONField(null=True, blank=True)
    path = models.CharField(max_length=500, null=True, blank=True)  # storage path of serialized model

    def __str__(self):
        return f"{self.name} ({self.version})"
