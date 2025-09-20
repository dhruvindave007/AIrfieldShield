# airfieldshield/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("core.urls")),        # Core APIs
    path("api/", include("alerts.urls")),      # Alerts APIs
    path("", include("dashboard.urls")),       # 👈 Add this for your dashboard UI
]
