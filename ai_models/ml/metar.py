"""Parsing of raw METAR text into the few flags the model needs.

Only the *observation body* of a report is used for observed weather. The trend
section (TEMPO/BECMG/NOSIG) is a short forecast appended by the duty forecaster;
it is parsed separately and exposed as its own signal.
"""
from __future__ import annotations

import re

_WX_CODES = (
    "MI|PR|BC|DR|BL|SH|TS|FZ|DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS"
)
_WX_TOKEN = re.compile(rf"^(?:\+|-|VC)?(?:{_WX_CODES})+$")
_CB_TOKEN = re.compile(r"^(?:FEW|SCT|BKN|OVC|///)(?:\d{3}|///)(?:CB|TCU)$")
_TREND_SPLIT = re.compile(r"\s(TEMPO|BECMG|NOSIG|RMK)\b")
_PRECIP = ("RA", "DZ", "SH", "GR", "GS", "SN")


def split_metar(raw: str) -> tuple[str, str]:
    """Return (observation body, trend/remarks remainder)."""
    if not isinstance(raw, str):
        return "", ""
    m = _TREND_SPLIT.search(raw)
    if not m:
        return raw, ""
    return raw[: m.start()], raw[m.start():].strip()


def _wx_tokens(text: str) -> list[str]:
    return [t for t in text.split() if _WX_TOKEN.match(t)]


def parse_metar_flags(raw: str) -> dict:
    """Extract observed thunderstorm / CB / precipitation flags and trend-TS flag.

    ts_obs    thunderstorm at or in the vicinity of the aerodrome (TS, TSRA, VCTS, ...)
    cb_obs    cumulonimbus or towering cumulus reported in a cloud group
    precip_obs any precipitation group
    trend_ts  the appended TEMPO/BECMG trend forecast mentions thunderstorm
    """
    body, rest = split_metar(raw)
    wx = _wx_tokens(body)
    trend = rest.split(" RMK")[0] if rest else ""
    return {
        "ts_obs": any("TS" in t for t in wx),
        "cb_obs": any(_CB_TOKEN.match(t) for t in body.split()),
        "precip_obs": any(any(p in t for p in _PRECIP) for t in wx),
        "trend_ts": any("TS" in t for t in _wx_tokens(trend)),
    }
