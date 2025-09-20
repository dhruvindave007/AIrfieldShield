#!/usr/bin/env python3
"""
Management command: sanitize_predictions

Usage:
  # Dry run (do not modify DB)
  python manage.py sanitize_predictions --dry-run

  # Sanitize all predictions (destructive)
  python manage.py sanitize_predictions

  # Sanitize first 100 predictions
  python manage.py sanitize_predictions --limit 100

  # Show verbose output for each modified id
  python manage.py sanitize_predictions --verbose
"""

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from ai_models.models import AIModel  # noqa: F401 (optional)
from core.models import Prediction

import json
import logging
from datetime import datetime, date
from decimal import Decimal

logger = logging.getLogger(__name__)


def _import_numpy():
    try:
        import numpy as _np  # type: ignore
        return _np
    except Exception:
        return None


def needs_sanitize(obj):
    """
    Quick recursive check for python objects that will break json.dumps:
    - datetime/date, Decimal, numpy types/arrays, set/tuple
    """
    _np = _import_numpy()
    if obj is None:
        return False
    if isinstance(obj, (str, int, float, bool)):
        return False
    if isinstance(obj, (datetime, date, Decimal)):
        return True
    if _np is not None and isinstance(obj, (_np.integer, _np.floating, _np.ndarray)):
        return True
    if isinstance(obj, dict):
        for k, v in obj.items():
            if needs_sanitize(v):
                return True
        return False
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            if needs_sanitize(v):
                return True
        return False
    # unknown types -> consider unsafe
    return True


def sanitize_for_json(obj):
    """
    Convert non-JSON-serializable types recursively into JSON-safe types.
    Mirrors the sanitizer used by prediction code.
    """
    _np = _import_numpy()
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        # Use ISO format (keeps timezone if present)
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, Decimal):
        try:
            return float(obj)
        except Exception:
            return str(obj)
    if _np is not None:
        # numpy scalars -> native
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
    # Fallback: try str()
    try:
        return str(obj)
    except Exception:
        return None


class Command(BaseCommand):
    help = "Sanitize Prediction.details JSON so it contains only JSON-serializable types"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Only report which Prediction rows would be modified, do not write changes.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Limit number of Prediction rows processed (for testing).",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print each changed Prediction id and a small diff-like summary.",
        )

    def handle(self, *args, **options):
        dry_run = options.get("dry_run", False)
        limit = options.get("limit")
        verbose = options.get("verbose", False)

        qs = Prediction.objects.all().order_by("id")
        total = qs.count()
        self.stdout.write(self.style.NOTICE(f"Total Prediction rows in DB: {total}"))
        if limit:
            qs = qs[:limit]
            self.stdout.write(self.style.NOTICE(f"Processing first {limit} rows (limit set)."))

        changed = 0
        scanned = 0

        for pred in qs:
            scanned += 1
            details = pred.details
            # if no details or already JSON-safe, skip quickly
            if not details:
                continue

            if not needs_sanitize(details):
                continue

            new_details = sanitize_for_json(details)

            # quick check: ensure new_details is JSON-serializable
            try:
                json.dumps(new_details)
            except TypeError as e:
                self.stdout.write(self.style.ERROR(f"Prediction id={pred.id}: sanitized details still not JSON-serializable: {e}"))
                # fallback: stringify entire details
                new_details = {"_original_details_str": str(details)}
                try:
                    json.dumps(new_details)
                except Exception:
                    self.stdout.write(self.style.ERROR(f"Prediction id={pred.id}: even fallback serialization failed; skipping."))
                    continue

            if dry_run:
                changed += 1
                if verbose:
                    self.stdout.write(f"[DRY] would sanitize Prediction id={pred.id}")
                else:
                    # show a small line
                    self.stdout.write(f"[DRY] would sanitize id={pred.id}")
                continue

            # Save sanitized details
            try:
                with transaction.atomic():
                    pred.details = new_details
                    # Only update the details column to be efficient
                    pred.save(update_fields=["details"])
                changed += 1
                if verbose:
                    self.stdout.write(self.style.SUCCESS(f"Sanitized Prediction id={pred.id}"))
                else:
                    self.stdout.write(self.style.SUCCESS(f"Sanitized id={pred.id}"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Failed to save sanitized details for Prediction id={pred.id}: {e}"))
                logger.exception("Failed to save sanitized details for Prediction id=%s", pred.id)

        self.stdout.write(self.style.SUCCESS(f"Scanned {scanned} rows. Sanitized {changed} rows. Dry-run={dry_run}"))
