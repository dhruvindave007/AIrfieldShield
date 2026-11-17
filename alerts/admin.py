from django.contrib import admin

from .models import Alert


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ("airfield", "alert_type", "level", "created_at", "valid_until", "acknowledged")
    list_filter = ("level", "alert_type", "acknowledged")
