# alerts/views.py
from django.utils import timezone
from django.db.models import Q
from rest_framework.generics import ListAPIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from django.views.decorators.csrf import csrf_exempt

from .models import Alert
from core.models import Prediction, Airfield
from .serializers import AlertSerializer
from core.serializers import PredictionSerializer

class LatestAlertsAPIView(ListAPIView):
    serializer_class = AlertSerializer

    def get_queryset(self):
        now = timezone.now()
        limit = int(self.request.query_params.get("limit", 20))
        include_ack = self.request.query_params.get("include_ack", "false").lower() == "true"
        airfield_q = self.request.query_params.get("airfield")

        qs = Alert.objects.filter(Q(valid_until__isnull=True) | Q(valid_until__gte=now)).order_by("-created_at")
        if not include_ack:
            qs = qs.filter(acknowledged=False)

        if airfield_q:
            try:
                aid = int(airfield_q)
                qs = qs.filter(airfield__id=aid)
            except Exception:
                qs = qs.filter(Q(airfield__icao__iexact=airfield_q) | Q(airfield__name__icontains=airfield_q))
        return qs[:limit]


class LatestPredictionsPerAirfield(APIView):
    def get(self, request):
        airfield_q = request.query_params.get("airfield")
        if airfield_q:
            try:
                aid = int(airfield_q)
                airfields = Airfield.objects.filter(id=aid)
            except Exception:
                airfields = Airfield.objects.filter(Q(icao__iexact=airfield_q) | Q(name__icontains=airfield_q))
        else:
            airfields = Airfield.objects.all()

        results = []
        for a in airfields:
            latest = a.predictions.order_by("-created_at").first()
            if latest:
                results.append(latest)
        serializer = PredictionBriefSerializer(results, many=True, context={"request": request})
        return Response({"predictions": serializer.data}, status=status.HTTP_200_OK)


@csrf_exempt
@api_view(["POST"])
@permission_classes([permissions.AllowAny])
def acknowledge_alert(request, pk):
    """
    POST /api/alerts/<pk>/acknowledge/
    Body (optional): {"acknowledged_by": "operator name"}
    """
    alert = get_object_or_404(Alert, pk=pk)
    if alert.acknowledged:
        serializer = AlertSerializer(alert, context={"request": request})
        return Response({"detail": "Already acknowledged", "alert": serializer.data}, status=status.HTTP_200_OK)

    ack_by = request.data.get("acknowledged_by")
    alert.acknowledged = True
    if ack_by:
        alert.acknowledged_by = str(ack_by)[:200]
    alert.acknowledged_at = timezone.now()
    alert.save(update_fields=["acknowledged", "acknowledged_by", "acknowledged_at"])
    serializer = AlertSerializer(alert, context={"request": request})
    return Response({"detail": "Acknowledged", "alert": serializer.data}, status=status.HTTP_200_OK)
