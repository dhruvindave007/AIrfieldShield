from django.db.models import Q
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.generics import ListAPIView
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import Airfield
from core.query import filter_airfields, int_param

from .models import Alert
from .serializers import AlertSerializer, PredictionBriefSerializer


class LatestAlertsAPIView(ListAPIView):
    """GET /api/alerts/latest/?airfield=<id|icao|name>&limit=20&include_ack=false"""
    serializer_class = AlertSerializer
    pagination_class = None

    def get_queryset(self):
        now = timezone.now()
        params = self.request.query_params
        limit = int_param(params.get("limit"), default=20, lo=1, hi=200)
        qs = Alert.objects.select_related("airfield").filter(
            Q(valid_until__isnull=True) | Q(valid_until__gte=now)).order_by("-created_at")
        if params.get("include_ack", "false").lower() != "true":
            qs = qs.filter(acknowledged=False)
        if params.get("airfield"):
            qs = qs.filter(airfield__in=filter_airfields(Airfield.objects.all(), params["airfield"]))
        return qs[:limit]


class LatestPredictionsPerAirfield(APIView):
    """GET /api/alerts/latest-predictions/?airfield=<id|icao|name>"""

    def get(self, request):
        airfields = filter_airfields(Airfield.objects.all(), request.query_params.get("airfield"))
        results = []
        for airfield in airfields:
            latest = airfield.predictions.order_by("-created_at").first()
            if latest:
                results.append(latest)
        data = PredictionBriefSerializer(results, many=True).data
        return Response({"predictions": data})


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def acknowledge_alert(request, pk):
    """POST /api/alerts/ack/<pk>/  (authenticated operators only)."""
    alert = get_object_or_404(Alert, pk=pk)
    if alert.acknowledged:
        return Response({"detail": "Already acknowledged", "alert": AlertSerializer(alert).data})
    alert.acknowledged = True
    alert.acknowledged_by = request.user.get_username()[:200]
    alert.acknowledged_at = timezone.now()
    alert.save(update_fields=["acknowledged", "acknowledged_by", "acknowledged_at"])
    return Response({"detail": "Acknowledged", "alert": AlertSerializer(alert).data},
                    status=status.HTTP_200_OK)
