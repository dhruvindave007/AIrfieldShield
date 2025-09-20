# utils/json_sanitize.py
import json
from datetime import datetime, date
from decimal import Decimal

try:
    import numpy as _np
except Exception:
    _np = None


def sanitize_for_json(obj):
    """
    Recursively convert Python objects that are not JSON serializable into JSON-safe types.
    This is used before saving any 'details' into a JSONField so we don't run into
    "Object of type datetime is not JSON serializable".
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:
            return str(obj)
    if _np is not None:
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            try:
                return obj.tolist()
            except Exception:
                return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    # fallback
    try:
        return str(obj)
    except Exception:
        return None


def dumps_safe(obj):
    """
    Return a JSON string (compact) for obj after sanitizing.
    """
    return json.dumps(sanitize_for_json(obj))
