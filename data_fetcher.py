"""
data_fetcher.py – Fetch & parse disaster event data from GDACS APIs.

Sources:
  1. CAP XML   → sent time, population, vulnerability, alert level, severity, country
  2. Media API → EMM news headlines (title, pubdate, source, link)
  3. GeoJSON   → event summary (magnitude, alert, country) — fallback to CAP
"""

import re
import requests
from lxml import etree
from datetime import datetime

# ── URL Templates ──────────────────────────────────────────────────────────────

CAP_URL = "https://www.gdacs.org/contentdata/resources/TC/{eid}/cap_{eid}.xml"
MEDIA_URL = "https://www.gdacs.org/gdacsapi/api/emm/getemmnewsbykey?eventtype=TC&eventid={eid}"
GEOJSON_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventdata?eventtype=TC&eventid={eid}"

HEADERS = {"User-Agent": "DSM-A1-Pipeline/1.0"}
TIMEOUT = 30  # seconds


# ── 1. CAP XML ─────────────────────────────────────────────────────────────────

def _parse_population_number(text: str) -> float:
    """Extract numeric population from strings like '0.197 million' or '1.529 million'."""
    if not text:
        return 0.0
    m = re.search(r"([\d.]+)\s*million", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Try plain number
    nums = re.findall(r"[\d.]+", text)
    if nums:
        return float(nums[0])
    return 0.0


def fetch_cap_xml(event_id: int) -> dict:
    """
    Fetch and parse the CAP XML for a given TC event.

    Returns dict with keys:
        sent, population_text, population_millions, vulnerability,
        alertlevel, severity, country, eventname, fromdate, todate
    """
    url = CAP_URL.format(eid=event_id)
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()

    root = etree.fromstring(resp.content)
    ns = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}

    result = {}

    # Top-level <sent>
    sent_el = root.find("cap:sent", ns)
    result["sent"] = sent_el.text.strip() if sent_el is not None else None

    # Parameters inside <info>
    params = {}
    for param in root.findall(".//cap:info/cap:parameter", ns):
        name_el = param.find("cap:valueName", ns)
        val_el = param.find("cap:value", ns)
        if name_el is not None and val_el is not None:
            params[name_el.text.strip()] = (val_el.text or "").strip()

    result["eventname"] = params.get("eventname", "")
    result["alertlevel"] = params.get("alertlevel", "")
    result["severity_text"] = params.get("severity", "")
    result["country"] = params.get("country", "")
    result["vulnerability"] = params.get("vulnerability", "")
    result["population_text"] = params.get("population", "")
    result["population_millions"] = _parse_population_number(result["population_text"])
    result["fromdate"] = params.get("fromdate", "")
    result["todate"] = params.get("todate", "")
    result["sourceid"] = params.get("sourceid", "")

    # Extract max wind speed from severity text
    ws = re.search(r"(\d+)\s*km/h", result["severity_text"])
    result["max_wind_kmh"] = int(ws.group(1)) if ws else None

    return result


# ── 2. Media API (EMM News) ───────────────────────────────────────────────────

def fetch_media(event_id: int) -> list[dict]:
    """
    Fetch EMM news articles for a TC event.

    Returns list of dicts with keys: emmid, pubdate, source, link, title, description
    """
    url = MEDIA_URL.format(eid=event_id)
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()

    articles = resp.json()
    # Normalise pubdate strings to datetime objects
    for art in articles:
        try:
            art["pubdate_dt"] = datetime.fromisoformat(art["pubdate"])
        except (ValueError, KeyError):
            art["pubdate_dt"] = None

    return articles


# ── 3. GeoJSON Event Data ─────────────────────────────────────────────────────

def fetch_geojson(event_id: int, cap_fallback: dict | None = None) -> dict:
    """
    Fetch the GeoJSON event summary.

    The GDACS API returns a single GeoJSON Feature (not FeatureCollection).
    Falls back to CAP-derived data if the endpoint times out or errors.
    """
    url = GEOJSON_URL.format(eid=event_id)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        # API returns a single Feature with properties at top level
        props = {}
        if data.get("type") == "Feature":
            props = data.get("properties", {})
        elif "features" in data and len(data["features"]) > 0:
            # FeatureCollection fallback
            props = data["features"][0].get("properties", {})

        if props:
            return {
                "eventname": props.get("eventname", ""),
                "alertlevel": props.get("alertlevel", ""),
                "alertscore": props.get("alertscore"),
                "country": props.get("country", ""),
                "fromdate": props.get("fromdate", ""),
                "todate": props.get("todate", ""),
                "description": props.get("description", ""),
                "htmldescription": props.get("htmldescription", ""),
                "glide": props.get("glide", ""),
                "iscurrent": props.get("iscurrent"),
                "coordinates": data.get("geometry", {}).get("coordinates", []),
                "source": "geojson",
            }
        return {"source": "geojson_empty"}

    except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
        # Fallback to CAP data
        if cap_fallback:
            return {
                "alertlevel": cap_fallback.get("alertlevel"),
                "country": cap_fallback.get("country"),
                "source": "cap_fallback",
            }
        return {"source": "error", "error": str(e)}


# ── Convenience ────────────────────────────────────────────────────────────────

def fetch_all(event_id: int) -> dict:
    """Fetch all three data sources for a single event."""
    cap = fetch_cap_xml(event_id)
    media = fetch_media(event_id)
    geojson = fetch_geojson(event_id, cap_fallback=cap)

    return {
        "event_id": event_id,
        "cap": cap,
        "media": media,
        "media_count": len(media),
        "geojson": geojson,
    }
