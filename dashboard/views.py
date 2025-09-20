# dashboard/views.py
from django.shortcuts import render

def home(request):
    return render(request, "dashboard/home.html")