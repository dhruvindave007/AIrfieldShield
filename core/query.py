"""Small helpers for validating query parameters."""
from django.db.models import Q


def int_param(value, default: int, lo: int, hi: int) -> int:
    """Parse an integer query parameter, falling back to ``default`` on garbage
    and clamping to [lo, hi] (never raises, never allows unbounded queries)."""
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def filter_airfields(qs, query):
    """Filter airfields by numeric id, exact ICAO code or (partial) name."""
    if not query:
        return qs
    if str(query).isdigit():
        return qs.filter(id=int(query))
    return qs.filter(Q(icao__iexact=query) | Q(name__icontains=query))


def find_airfield(qs, query):
    """Single best match: id, then exact ICAO, then exact name, then partial name."""
    if not query:
        return None
    if str(query).isdigit():
        return qs.filter(id=int(query)).first()
    return (qs.filter(icao__iexact=query).first()
            or qs.filter(name__iexact=query).first()
            or qs.filter(name__icontains=query).order_by("id").first())
