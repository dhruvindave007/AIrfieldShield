# dashboard/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="dashboard_home"),   # root path -> dashboard
]
