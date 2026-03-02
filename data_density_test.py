"""
data_density_test.py – Check data density across candidate GDACS events.

Validates:
  1. Media API  : headline count (>50 for NLP viability)
  2. CAP XML    : population, vulnerability, alertlevel, sent timestamp
  3. GeoJSON    : event metadata availability
  4. Text depth : average description length for NLP quality
"""

import re
import requests
from lxml import etree

HEADERS = {"User-Agent": "DSM-A1-Pipeline/1.0"}
TIMEOUT = 25


def check_data_density(event_id, event_type="TC"):
    """
    Comprehensive density check for a single GDACS event.
    Returns dict with availability flags for every metric the pipeline needs.
    """
    results = {
        "id": event_id,
        # Media
        "headlines": 0,
        "avg_desc_len": 0,
        "english_pct": 0,
        # CAP XML
        "cap_ok": False,
        "has_sent": False,
        "has_population": False,
        "population_millions": 0,
        "has_vulnerability": False,
        "vulnerability": "",
        "has_alertlevel": False,
        "alertlevel": "",
        "eventname": "",
        "max_wind_kmh": 0,
        # GeoJSON
        "geojson_ok": False,
    }

    # ── 1. Media API ──────────────────────────────────────────────────────
    media_url = (
        f"https://www.gdacs.org/gdacsapi/api/emm/getemmnewsbykey"
        f"?eventtype={event_type}&eventid={event_id}"
    )
    try:
        resp = requests.get(media_url, headers=HEADERS, timeout=TIMEOUT)
        articles = resp.json() if resp.status_code == 200 else []
        if isinstance(articles, list):
            results["headlines"] = len(articles)

            # Measure text depth: avg description length
            desc_lens = [len(a.get("description", "")) for a in articles if a.get("description")]
            results["avg_desc_len"] = round(sum(desc_lens) / max(len(desc_lens), 1))

            # Estimate English content (rough heuristic: ASCII-heavy titles)
            eng = sum(1 for a in articles if a.get("title", "").isascii())
            results["english_pct"] = round(eng / max(len(articles), 1) * 100)
    except Exception as e:
        print(f"  [WARN] Media API error for {event_id}: {e}")

    # ── 2. CAP XML ────────────────────────────────────────────────────────
    cap_url = (
        f"https://www.gdacs.org/contentdata/resources/{event_type}"
        f"/{event_id}/cap_{event_id}.xml"
    )
    try:
        resp = requests.get(cap_url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code == 200:
            root = etree.fromstring(resp.content)
            ns = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}
            results["cap_ok"] = True

            # <sent> for Response Delta
            sent = root.find("cap:sent", ns)
            results["has_sent"] = sent is not None and bool(sent.text.strip())

            # Parameters
            params = {}
            for p in root.findall(".//cap:info/cap:parameter", ns):
                n = p.find("cap:valueName", ns)
                v = p.find("cap:value", ns)
                if n is not None and v is not None:
                    params[n.text.strip()] = (v.text or "").strip()

            results["eventname"] = params.get("eventname", "")
            results["alertlevel"] = params.get("alertlevel", "")
            results["has_alertlevel"] = bool(results["alertlevel"])
            results["vulnerability"] = params.get("vulnerability", "")
            results["has_vulnerability"] = bool(results["vulnerability"])

            pop_text = params.get("population", "")
            results["has_population"] = bool(pop_text)
            m = re.search(r"([\d.]+)\s*million", pop_text, re.IGNORECASE)
            results["population_millions"] = float(m.group(1)) if m else 0

            ws = re.search(r"(\d+)\s*km/h", params.get("severity", ""))
            results["max_wind_kmh"] = int(ws.group(1)) if ws else 0
    except Exception as e:
        print(f"  [WARN] CAP XML error for {event_id}: {e}")

    # ── 3. GeoJSON ────────────────────────────────────────────────────────
    geo_url = (
        f"https://www.gdacs.org/gdacsapi/api/events/geteventdata"
        f"?eventtype={event_type}&eventid={event_id}"
    )
    try:
        resp = requests.get(geo_url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            results["geojson_ok"] = True
    except Exception:
        pass  # GeoJSON often times out, non-critical

    return results


def print_report(results):
    """Pretty-print a density report for one event."""
    r = results
    name = r["eventname"] or f"ID-{r['id']}"

    # Determine overall density grade
    checks = [
        r["headlines"] > 50,        # Enough for NLP
        r["has_sent"],              # For Response Delta
        r["has_population"],        # For Forgotten Crisis Index
        r["has_vulnerability"],     # For Vulnerability Benchmark
        r["has_alertlevel"],        # For Vulnerability Benchmark
        r["avg_desc_len"] > 100,    # Descriptions long enough for NLP
    ]
    passed = sum(checks)
    grade = "HIGH DENSITY" if passed >= 5 else ("MEDIUM" if passed >= 3 else "LOW DATA")
    icon = "[+++]" if passed >= 5 else ("[++]" if passed >= 3 else "[!]")

    print(f"\n  {icon} {name} (ID: {r['id']}) -- {grade} ({passed}/6 checks)")
    print(f"      Headlines     : {r['headlines']:>4}  (English ~{r['english_pct']}%)")
    print(f"      Avg Desc Len  : {r['avg_desc_len']:>4} chars")
    print(f"      CAP XML       : {'OK' if r['cap_ok'] else 'MISSING'}")
    print(f"        sent        : {'Yes' if r['has_sent'] else 'No'}")
    print(f"        population  : {r['population_millions']}M" if r['has_population'] else "        population  : No")
    print(f"        vulnerability: {r['vulnerability']}" if r['has_vulnerability'] else "        vulnerability: No")
    print(f"        alertlevel  : {r['alertlevel']}" if r['has_alertlevel'] else "        alertlevel  : No")
    print(f"        wind speed  : {r['max_wind_kmh']} km/h")
    print(f"      GeoJSON       : {'OK' if r['geojson_ok'] else 'timeout/missing'}")


# ── Run Density Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Candidates
    historical_ids = [1000498, 1000552, 1000667]  # Mangkhut, Idai, Amphan
    recent_ids     = [1001256, 1001233, 1001254]  # Gezani, Kalmaegi, Fytia

    print("=" * 60)
    print("  GDACS Event Data Density Test")
    print("=" * 60)

    print("\n--- Historical Events (2018-2021) ---")
    for eid in historical_ids:
        res = check_data_density(eid)
        print_report(res)

    print("\n--- Recent Events (Last 12 Months) ---")
    for eid in recent_ids:
        res = check_data_density(eid)
        print_report(res)

    print("\n" + "=" * 60)
    print("  Metric coverage key:")
    print("    sent         -> Response Delta (dT)")
    print("    population   -> Forgotten Crisis Index")
    print("    vulnerability -> Vulnerability Benchmark")
    print("    alertlevel   -> Vulnerability Benchmark")
    print("    headlines>50 -> Sentiment Volatility + NER")
    print("    desc_len>100 -> NLP text quality")
    print("=" * 60)