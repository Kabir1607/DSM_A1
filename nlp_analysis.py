"""
nlp_analysis.py -- Focused NLP & Comparative Analysis for Event A vs Event B.

Event A (Historical): Cyclone Amphan (2020) — India/Bangladesh
Event B (Recent):     Cyclone Gezani (2026) — Madagascar

Covers all assignment requirements:
  Task 2.1: Response Delta
  Task 2.2: NER (organisations, deaths, losses, relief funds)
  Task 2.3: Sentiment Volatility (alarmist vs analytical tone comparison)
  Task 3.1: Forgotten Crisis Index
  Task 3.2: Vulnerability Benchmark (coping capacity vs alert vs magnitude)

Outputs:
  - nlp_results.json   (machine-readable, full results)
  - nlp_report.txt     (human-readable intelligence brief)

Usage:
    python -u nlp_analysis.py
"""

import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime

os.environ["PYTHONUNBUFFERED"] = "1"

from textblob import TextBlob

# -- Configuration -------------------------------------------------------------

CACHE_FILE = "pipeline_cache.json"
EVENT_A_KEY = "Historical_Amphan"
EVENT_B_KEY = "Recent_Gezani"

OUTPUT_JSON = "nlp_results.json"
OUTPUT_REPORT = "nlp_report.txt"


# -- NER Patterns --------------------------------------------------------------

_ORGS_LONG = [
    "Red Cross", "Red Crescent", "Oxfam", "Save the Children",
    "Doctors Without Borders", "World Vision", "Action Contre la Faim",
    "European Union", "United Nations", "European Commission",
    "Indian Army", "Indian Navy", "National Disaster Response Force",
]
_ORGS_SHORT = [
    "UNICEF", "UNDP", "UNHCR", "WHO", "WFP", "OCHA", "IFRC", "ICRC",
    "MSF", "CARE", "NDRF", "NDMA", "FEMA", "BNGRC", "USAID", "EU", "UN",
    "IMD", "JTWC",
]
_GOV_ORGS = [
    "Government", "Ministry", "Prime Minister", "President",
    "Chief Minister", "National Government",
]

_DEATH_RE = re.compile(
    r"(?:"
    r"(?:kill(?:ed|ing|s)?|died|dead|death[s]?|fatalit(?:y|ies)|perish(?:ed)?)"
    r"\s+(?:at\s+least\s+)?(\d[\d,]*)"
    r"|"
    r"(\d[\d,]*)\s+(?:people\s+)?(?:kill(?:ed|s)?|dead|died|fatalit(?:y|ies)|perish(?:ed)?)"
    r"|"
    r"death\s+toll\s+(?:of\s+|reach(?:ed|es)?\s+|rose\s+to\s+|stands?\s+at\s+)?(\d[\d,]*)"
    r")",
    re.IGNORECASE,
)

_DISPLACED_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:million\s+)?(?:people\s+)?(?:displac|evacuat|homeless|shelter|relocat)",
    re.IGNORECASE,
)

_DAMAGE_RE = re.compile(
    r"(\$[\d,.]+\s*(?:billion|million|bn|m)?)\s*(?:in\s+)?(?:damage|loss|destruction|cost)"
    r"|(?:damage|loss|cost)(?:s|ed)?\s*(?:of\s+|estimated\s+at\s+|worth\s+)?(\$[\d,.]+\s*(?:billion|million|bn|m)?)"
    r"|(\d[\d,]*)\s+(?:homes?|houses?|buildings?|structures?)\s+(?:destroy|damag|flatten)",
    re.IGNORECASE,
)

_MONEY_RE = re.compile(
    r"(?:"
    r"\$\s?[\d,]+(?:\.\d+)?\s*(?:million|billion|bn|m|k|trillion)?"
    r"|EUR\s?[\d,]+(?:\.\d+)?\s*(?:million|billion)?"
    r"|USD\s?[\d,]+(?:\.\d+)?\s*(?:million|billion|crore|lakh)?"
    r"|Rs\.?\s?[\d,]+(?:\.\d+)?\s*(?:crore|lakh|million|billion)?"
    r"|[\d,]+(?:\.\d+)?\s*(?:million|billion)\s*(?:dollars|euros|usd)"
    r")",
    re.IGNORECASE,
)


def _clean_money(mentions):
    cleaned = []
    for m in mentions:
        m = m.strip()
        if len(m) < 4:
            continue
        if re.match(r"^(rs|eur|usd)[.,\s]*$", m, re.I):
            continue
        nums = re.findall(r"[\d,]+(?:\.\d+)?", m)
        if nums:
            try:
                val = float(nums[0].replace(",", ""))
                has_unit = bool(re.search(r"(?:million|billion|crore|lakh|bn|m)", m, re.I))
                if val < 100 and not has_unit:
                    continue
            except ValueError:
                pass
        cleaned.append(m)
    return cleaned


# -- Analysis Functions --------------------------------------------------------

def analyse_sentiment(articles, corpus_text):
    """Full sentiment analysis with tone classification."""
    headline_data = []
    for art in articles:
        title = art.get("title", "")
        if not title:
            continue
        blob = TextBlob(title)
        pol = blob.sentiment.polarity
        sub = blob.sentiment.subjectivity
        # Classify
        if sub > 0.4 or pol < -0.1:
            tone = "alarmist"
        elif sub < 0.3 and abs(pol) < 0.1:
            tone = "analytical"
        else:
            tone = "neutral"
        headline_data.append({
            "title": title[:80],
            "polarity": round(pol, 4),
            "subjectivity": round(sub, 4),
            "tone": tone,
        })

    # Corpus paragraph analysis
    corpus_pols = []
    if corpus_text:
        paragraphs = [p.strip() for p in corpus_text.split("\n") if len(p.strip()) > 50]
        for para in paragraphs[:300]:
            corpus_pols.append(TextBlob(para).sentiment.polarity)

    headline_pols = [h["polarity"] for h in headline_data]
    all_pols = headline_pols + corpus_pols

    # Tone breakdown
    tone_counts = Counter(h["tone"] for h in headline_data)
    total = len(headline_data) or 1

    # Most alarmist and most analytical headlines
    sorted_alarming = sorted(headline_data, key=lambda x: x["polarity"])
    sorted_positive = sorted(headline_data, key=lambda x: x["polarity"], reverse=True)

    return {
        "headline_count": len(headline_data),
        "headline_polarity_mean": round(statistics.mean(headline_pols), 4) if headline_pols else None,
        "headline_polarity_std": round(statistics.stdev(headline_pols), 4) if len(headline_pols) > 1 else 0,
        "corpus_polarity_mean": round(statistics.mean(corpus_pols), 4) if corpus_pols else None,
        "combined_polarity_mean": round(statistics.mean(all_pols), 4) if all_pols else None,
        "combined_polarity_std": round(statistics.stdev(all_pols), 4) if len(all_pols) > 1 else 0,
        "polarity_min": round(min(all_pols), 4) if all_pols else None,
        "polarity_max": round(max(all_pols), 4) if all_pols else None,
        "tone_alarmist": tone_counts.get("alarmist", 0),
        "tone_analytical": tone_counts.get("analytical", 0),
        "tone_neutral": tone_counts.get("neutral", 0),
        "tone_alarmist_pct": round(tone_counts.get("alarmist", 0) / total * 100, 1),
        "tone_analytical_pct": round(tone_counts.get("analytical", 0) / total * 100, 1),
        "most_negative_headlines": [
            {"title": h["title"], "polarity": h["polarity"]}
            for h in sorted_alarming[:5]
        ],
        "most_positive_headlines": [
            {"title": h["title"], "polarity": h["polarity"]}
            for h in sorted_positive[:5]
        ],
    }


def analyse_ner(articles, corpus_text):
    """Full NER: organisations, governments, deaths, displaced, damage, relief funds."""
    all_texts = []
    for art in articles:
        if art.get("title"):
            all_texts.append(art["title"])
        if art.get("description"):
            all_texts.append(art["description"])
    if corpus_text:
        all_texts.append(corpus_text)
    combined = "\n".join(all_texts)

    # Organisations
    org_counter = Counter()
    for org in _ORGS_LONG:
        count = len(re.findall(re.escape(org), combined, re.I))
        if count:
            org_counter[org] += count
    for org in _ORGS_SHORT:
        count = len(re.findall(r'\b' + re.escape(org) + r'\b', combined))
        if count:
            org_counter[org] += count

    # Government mentions
    gov_counter = Counter()
    for gov in _GOV_ORGS:
        count = len(re.findall(re.escape(gov), combined, re.I))
        if count:
            gov_counter[gov] += count

    # Deaths
    death_values = []
    for m in _DEATH_RE.finditer(combined):
        num = m.group(1) or m.group(2) or m.group(3)
        if num:
            death_values.append(int(num.replace(",", "")))

    # Displaced / Evacuated
    displaced_values = []
    for m in _DISPLACED_RE.finditer(combined):
        val = m.group(1).replace(",", "")
        try:
            n = float(val)
            # Check if "million" is near
            ctx = combined[m.start():m.end()+20]
            if "million" in ctx.lower():
                n *= 1_000_000
            displaced_values.append(int(n))
        except ValueError:
            pass

    # Damage
    damage_items = []
    for m in _DAMAGE_RE.finditer(combined):
        val = m.group(1) or m.group(2) or m.group(3)
        if val:
            damage_items.append(val.strip())

    # Money / Relief Funds
    raw_money = list(set(_MONEY_RE.findall(combined)))
    money_clean = _clean_money(raw_money)

    return {
        "organisations": {
            "total_mentions": sum(org_counter.values()),
            "unique_orgs": len(org_counter),
            "top_15": org_counter.most_common(15),
        },
        "government_mentions": {
            "total": sum(gov_counter.values()),
            "breakdown": dict(gov_counter.most_common(10)),
        },
        "casualties": {
            "death_extractions": len(death_values),
            "max_reported": max(death_values) if death_values else 0,
            "min_reported": min(death_values) if death_values else 0,
            "all_values": sorted(set(death_values), reverse=True)[:10],
        },
        "displaced": {
            "extractions": len(displaced_values),
            "max_reported": max(displaced_values) if displaced_values else 0,
            "values": sorted(set(displaced_values), reverse=True)[:5],
        },
        "damage": {
            "mentions": len(damage_items),
            "items": list(set(damage_items))[:10],
        },
        "relief_funds": {
            "count": len(money_clean),
            "mentions": money_clean[:15],
        },
    }


def compute_response_delta(cap, articles):
    """Task 2.1: Delta_T = T_MediaPeak - T_SystemAlert."""
    from email.utils import parsedate_to_datetime

    # Event onset (system alert)
    onset_str = cap.get("fromdate") or cap.get("sent")
    onset_dt = None
    if onset_str:
        try:
            onset_dt = datetime.fromisoformat(onset_str)
        except Exception:
            try:
                onset_dt = parsedate_to_datetime(onset_str)
            except Exception:
                pass

    # Media peak
    day_counts = Counter()
    for art in articles:
        try:
            dt = datetime.fromisoformat(art.get("pubdate", ""))
            day_counts[dt.strftime("%Y-%m-%d")] += 1
        except (ValueError, TypeError):
            pass

    if not day_counts:
        return {"delta_hours": None, "event_onset": str(onset_dt), "media_peak_date": None}

    peak_day, peak_count = day_counts.most_common(1)[0]
    peak_dt = datetime.strptime(peak_day, "%Y-%m-%d")

    delta_hours = None
    if onset_dt:
        delta = peak_dt - onset_dt.replace(tzinfo=None)
        delta_hours = round(delta.total_seconds() / 3600, 1)

    # Daily timeline
    timeline = sorted(day_counts.items())

    return {
        "event_onset": str(onset_dt),
        "media_peak_date": peak_day,
        "media_peak_count": peak_count,
        "delta_hours": delta_hours,
        "delta_days": round(delta_hours / 24, 1) if delta_hours else None,
        "daily_article_counts": timeline,
        "total_coverage_days": len(day_counts),
    }


def compute_forgotten_crisis(media_count, population_millions):
    """Task 3.1: News Volume / Population Impact."""
    if population_millions <= 0:
        return {"index": None}
    ratio = round(media_count / population_millions, 2)
    coverage_class = (
        "OVER-COVERED" if ratio > 500
        else "ADEQUATELY COVERED" if ratio > 100
        else "UNDER-REPORTED" if ratio > 10
        else "SEVERELY UNDER-REPORTED"
    )
    return {
        "media_count": media_count,
        "population_millions": population_millions,
        "index": ratio,
        "coverage_classification": coverage_class,
    }


def compute_vulnerability_benchmark(cap):
    """Task 3.2: Coping Capacity vs Alert vs Magnitude."""
    vuln_map = {"high": 3, "medium": 2, "low": 1}
    alert_map = {"red": 3, "orange": 2, "green": 1}

    vuln = cap.get("vulnerability", "").lower()
    alert = cap.get("alertlevel", "").lower()
    magnitude = cap.get("max_wind_kmh")
    pop = cap.get("population_millions", 0)

    vuln_score = vuln_map.get(vuln, 0)
    alert_score = alert_map.get(alert, 0)
    coping_capacity = (4 - vuln_score) if vuln_score > 0 else None

    return {
        "vulnerability": cap.get("vulnerability", ""),
        "vulnerability_score": vuln_score,
        "alert_level": cap.get("alertlevel", ""),
        "alert_score": alert_score,
        "coping_capacity": coping_capacity,
        "coping_label": (
            "HIGH" if coping_capacity and coping_capacity >= 3
            else "MODERATE" if coping_capacity and coping_capacity >= 2
            else "LOW" if coping_capacity else "UNKNOWN"
        ),
        "magnitude_kmh": magnitude,
        "population_millions": pop,
        "severity_text": cap.get("severity_text", ""),
    }


# -- Main Analysis -------------------------------------------------------------

def run_analysis():
    print("=" * 70)
    print("  NLP ANALYSIS: Amphan (Event A) vs Gezani (Event B)")
    print("=" * 70, flush=True)

    # Load cache
    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found. Run run_pipeline.py first.")
        sys.exit(1)

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    if EVENT_A_KEY not in cache or EVENT_B_KEY not in cache:
        print(f"ERROR: {EVENT_A_KEY} or {EVENT_B_KEY} not in cache.")
        sys.exit(1)

    results = {}

    for label, key in [("Event_A_Amphan", EVENT_A_KEY), ("Event_B_Gezani", EVENT_B_KEY)]:
        ev = cache[key]
        cap = ev["cap"]
        media = ev["media"]
        corpus_text = ev.get("corpus_summary", {}).get("corpus_text", "")
        corpus_stats = ev.get("corpus_summary", {}).get("sources", {})

        print(f"\n{'-' * 70}")
        print(f"  Analysing: {label} ({cap.get('eventname', key)})")
        print(f"  Country: {cap.get('country', '?')}")
        print(f"  Period: {cap.get('fromdate', '?')} to {cap.get('todate', '?')}")
        print(f"  Articles: {ev.get('media_count', 0)}, Corpus: {ev.get('corpus_summary', {}).get('total_words', 0)} words")
        print(f"{'-' * 70}", flush=True)

        # Task 2.1: Response Delta
        print("\n  [2.1] Response Delta...", flush=True)
        rd = compute_response_delta(cap, media)
        print(f"    System Alert : {rd['event_onset']}")
        print(f"    Media Peak   : {rd['media_peak_date']} ({rd['media_peak_count']} articles)")
        print(f"    Delta        : {rd['delta_hours']}h ({rd.get('delta_days', '?')} days)")
        print(f"    Coverage span: {rd['total_coverage_days']} days", flush=True)

        # Task 2.2: NER
        print("\n  [2.2] Named Entity Recognition...", flush=True)
        ner = analyse_ner(media, corpus_text)
        print(f"    Organisations: {ner['organisations']['total_mentions']} mentions, "
              f"{ner['organisations']['unique_orgs']} unique")
        top5 = ner['organisations']['top_15'][:5]
        print(f"    Top 5: {', '.join(f'{n}({c})' for n,c in top5)}")
        print(f"    Govt mentions: {ner['government_mentions']['total']}")
        print(f"    Deaths: {ner['casualties']['max_reported']} max "
              f"({ner['casualties']['death_extractions']} mentions)")
        print(f"    Displaced: {ner['displaced']['max_reported']} max")
        print(f"    Damage: {ner['damage']['items'][:3]}")
        print(f"    Relief funds: {ner['relief_funds']['count']} mentions", flush=True)

        # Task 2.3: Sentiment
        print("\n  [2.3] Sentiment Analysis...", flush=True)
        sent = analyse_sentiment(media, corpus_text)
        print(f"    Headlines: mean={sent['headline_polarity_mean']}, "
              f"std={sent['headline_polarity_std']}")
        print(f"    Corpus:    mean={sent['corpus_polarity_mean']}")
        print(f"    Tone: {sent['tone_alarmist_pct']}% alarmist, "
              f"{sent['tone_analytical_pct']}% analytical")
        print(f"    Range: [{sent['polarity_min']}, {sent['polarity_max']}]", flush=True)

        # Task 3.1: Forgotten Crisis Index
        print("\n  [3.1] Forgotten Crisis Index...", flush=True)
        fc = compute_forgotten_crisis(ev.get("media_count", 0), cap.get("population_millions", 0))
        print(f"    Index: {fc['index']} ({fc.get('coverage_classification', '?')})", flush=True)

        # Task 3.2: Vulnerability Benchmark
        print("\n  [3.2] Vulnerability Benchmark...", flush=True)
        vb = compute_vulnerability_benchmark(cap)
        print(f"    Vulnerability: {vb['vulnerability']} (score={vb['vulnerability_score']})")
        print(f"    Alert Level:   {vb['alert_level']} (score={vb['alert_score']})")
        print(f"    Coping:        {vb['coping_label']} ({vb['coping_capacity']})")
        print(f"    Magnitude:     {vb['magnitude_kmh']} km/h", flush=True)

        # Store
        results[label] = {
            "event_id": ev.get("event_id"),
            "event_name": cap.get("eventname", ""),
            "country": cap.get("country", ""),
            "period": f"{cap.get('fromdate', '?')} to {cap.get('todate', '?')}",
            "media_count": ev.get("media_count", 0),
            "corpus_words": ev.get("corpus_summary", {}).get("total_words", 0),
            "corpus_sources": corpus_stats,
            "response_delta": rd,
            "ner": ner,
            "sentiment": sent,
            "forgotten_crisis": fc,
            "vulnerability_benchmark": vb,
        }

    # -- Comparative Analysis --------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  COMPARATIVE ANALYSIS: Event A vs Event B")
    print(f"{'=' * 70}", flush=True)

    a = results["Event_A_Amphan"]
    b = results["Event_B_Gezani"]

    # Response speed comparison
    dt_a = a["response_delta"]["delta_hours"]
    dt_b = b["response_delta"]["delta_hours"]
    if dt_a is not None and dt_b is not None:
        if dt_b < dt_a:
            speed_insight = f"Response {dt_a - dt_b:.0f}h FASTER for Event B (Gezani)"
        else:
            speed_insight = f"Response {dt_b - dt_a:.0f}h SLOWER for Event B (Gezani)"
    else:
        speed_insight = "Cannot compare — missing delta data"

    # Forgotten Crisis comparison
    fc_a = a["forgotten_crisis"]["index"]
    fc_b = b["forgotten_crisis"]["index"]
    if fc_a and fc_b:
        if fc_a > fc_b:
            coverage_insight = (f"Amphan received {fc_a/fc_b:.1f}x more media coverage per capita. "
                                f"Gezani is relatively UNDER-REPORTED.")
        else:
            coverage_insight = (f"Gezani received {fc_b/fc_a:.1f}x more media coverage per capita.")
    else:
        coverage_insight = "Cannot compare"

    # Tone comparison
    tone_a = a["sentiment"]["tone_alarmist_pct"]
    tone_b = b["sentiment"]["tone_alarmist_pct"]
    if tone_a > tone_b:
        tone_insight = f"Amphan coverage was MORE ALARMIST ({tone_a}% vs {tone_b}%)"
    else:
        tone_insight = f"Gezani coverage was MORE ALARMIST ({tone_b}% vs {tone_a}%)"

    # Vulnerability comparison
    coping_a = a["vulnerability_benchmark"]["coping_capacity"]
    coping_b = b["vulnerability_benchmark"]["coping_capacity"]
    alert_a = a["vulnerability_benchmark"]["alert_level"]
    alert_b = b["vulnerability_benchmark"]["alert_level"]
    mag_a = a["vulnerability_benchmark"]["magnitude_kmh"]
    mag_b = b["vulnerability_benchmark"]["magnitude_kmh"]

    vuln_insight = (
        f"Both regions have {'similar' if coping_a == coping_b else 'different'} "
        f"coping capacity (A={coping_a}, B={coping_b}). "
        f"Alert levels: A={alert_a}, B={alert_b}. "
        f"Magnitudes: A={mag_a}km/h, B={mag_b}km/h."
    )
    if coping_a == coping_b and mag_a and mag_b and mag_a > mag_b:
        vuln_insight += " Despite similar coping capacity, Amphan was stronger — higher magnitude."

    comparative = {
        "response_speed": {
            "delta_a_hours": dt_a,
            "delta_b_hours": dt_b,
            "insight": speed_insight,
        },
        "forgotten_crisis": {
            "index_a": fc_a,
            "index_b": fc_b,
            "insight": coverage_insight,
        },
        "tone_evolution": {
            "alarmist_pct_a": tone_a,
            "alarmist_pct_b": tone_b,
            "analytical_pct_a": a["sentiment"]["tone_analytical_pct"],
            "analytical_pct_b": b["sentiment"]["tone_analytical_pct"],
            "insight": tone_insight,
        },
        "vulnerability": {
            "coping_a": coping_a,
            "coping_b": coping_b,
            "insight": vuln_insight,
        },
        "deaths_comparison": {
            "deaths_a": a["ner"]["casualties"]["max_reported"],
            "deaths_b": b["ner"]["casualties"]["max_reported"],
        },
        "overall_question": "Is the world responding faster today? Is media focus aligning with vulnerability?",
        "answer": (
            f"Response Delta: A={dt_a}h, B={dt_b}h — "
            f"{'YES, faster media response' if dt_b is not None and dt_a is not None and dt_b < dt_a else 'Comparable or slower'}. "
            f"Media alignment: Amphan (FCI={fc_a}) was {'over-covered' if fc_a and fc_a > 500 else 'adequately covered'}, "
            f"Gezani (FCI={fc_b}) was {'under-reported' if fc_b and fc_b < 100 else 'adequately covered'}. "
            f"Vulnerability alignment: {'Poor' if fc_b and fc_b < 100 else 'Good'} — "
            f"low-coping regions still receive less coverage."
        ),
    }

    results["comparative_analysis"] = comparative

    print(f"\n  Response Speed: {speed_insight}")
    print(f"  Coverage Gap:  {coverage_insight}")
    print(f"  Tone Shift:    {tone_insight}")
    print(f"  Vulnerability: {vuln_insight}")
    print(f"\n  KEY FINDING: {comparative['answer']}", flush=True)

    # -- Save Results ----------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  Saving results...", flush=True)

    # JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [OK] {OUTPUT_JSON} ({os.path.getsize(OUTPUT_JSON):,} bytes)")

    # Human-readable report
    _write_report(results, a, b, comparative)
    print(f"  [OK] {OUTPUT_REPORT} ({os.path.getsize(OUTPUT_REPORT):,} bytes)")
    print(f"{'=' * 70}", flush=True)


def _write_report(results, a, b, comp):
    """Write the human-readable intelligence brief."""
    lines = []
    w = lines.append

    w("=" * 72)
    w("  DISASTER EVOLUTION & RESPONSE PIPELINE — NLP INTELLIGENCE BRIEF")
    w("  Event A: Cyclone Amphan (2020) — India, Bangladesh")
    w("  Event B: Cyclone Gezani (2026) — Madagascar")
    w("=" * 72)

    w("\n--- TASK 2.1: RESPONSE DELTA ---")
    rd_a = a["response_delta"]
    rd_b = b["response_delta"]
    w(f"  Event A — System Alert: {rd_a['event_onset']}")
    w(f"            Media Peak:   {rd_a['media_peak_date']} ({rd_a['media_peak_count']} articles)")
    w(f"            Delta_T:      {rd_a['delta_hours']}h ({rd_a.get('delta_days','?')} days)")
    w(f"            Coverage:     {rd_a['total_coverage_days']} days")
    w(f"  Event B — System Alert: {rd_b['event_onset']}")
    w(f"            Media Peak:   {rd_b['media_peak_date']} ({rd_b['media_peak_count']} articles)")
    w(f"            Delta_T:      {rd_b['delta_hours']}h ({rd_b.get('delta_days','?')} days)")
    w(f"            Coverage:     {rd_b['total_coverage_days']} days")
    w(f"  >> {comp['response_speed']['insight']}")

    w("\n--- TASK 2.2: NAMED ENTITY RECOGNITION ---")
    for lbl, ev in [("Event A (Amphan)", a), ("Event B (Gezani)", b)]:
        nr = ev["ner"]
        w(f"\n  {lbl}:")
        w(f"    Organisations: {nr['organisations']['total_mentions']} mentions, "
          f"{nr['organisations']['unique_orgs']} unique")
        top = nr['organisations']['top_15'][:8]
        w(f"    Top: {', '.join(f'{n} ({c})' for n,c in top)}")
        w(f"    Government:   {nr['government_mentions']['total']} mentions")
        w(f"    Deaths:       {nr['casualties']['max_reported']} max reported "
          f"({nr['casualties']['death_extractions']} extractions)")
        if nr['casualties']['all_values']:
            w(f"    Death values:  {nr['casualties']['all_values'][:5]}")
        w(f"    Displaced:    {nr['displaced']['max_reported']} max")
        w(f"    Damage:       {nr['damage']['items'][:5]}")
        w(f"    Relief funds: {nr['relief_funds']['mentions'][:5]}")

    w("\n--- TASK 2.3: SENTIMENT VOLATILITY ---")
    for lbl, ev in [("Event A (Amphan)", a), ("Event B (Gezani)", b)]:
        s = ev["sentiment"]
        w(f"\n  {lbl}:")
        w(f"    Headline polarity: mean={s['headline_polarity_mean']}, std={s['headline_polarity_std']}")
        w(f"    Corpus polarity:   mean={s['corpus_polarity_mean']}")
        w(f"    Range:             [{s['polarity_min']}, {s['polarity_max']}]")
        w(f"    Tone breakdown:    {s['tone_alarmist_pct']}% alarmist, "
          f"{s['tone_analytical_pct']}% analytical, "
          f"{100 - s['tone_alarmist_pct'] - s['tone_analytical_pct']:.1f}% neutral")
        w(f"    Most negative headline: {s['most_negative_headlines'][0]['title']}")
        w(f"      polarity = {s['most_negative_headlines'][0]['polarity']}")
    w(f"\n  >> {comp['tone_evolution']['insight']}")

    w("\n--- TASK 3.1: FORGOTTEN CRISIS INDEX ---")
    w(f"  Event A: {a['forgotten_crisis']['index']} "
      f"(media={a['forgotten_crisis']['media_count']}, "
      f"pop={a['forgotten_crisis']['population_millions']}M) "
      f"— {a['forgotten_crisis']['coverage_classification']}")
    w(f"  Event B: {b['forgotten_crisis']['index']} "
      f"(media={b['forgotten_crisis']['media_count']}, "
      f"pop={b['forgotten_crisis']['population_millions']}M) "
      f"— {b['forgotten_crisis']['coverage_classification']}")
    w(f"  >> {comp['forgotten_crisis']['insight']}")

    w("\n--- TASK 3.2: VULNERABILITY BENCHMARK ---")
    for lbl, ev in [("Event A (Amphan)", a), ("Event B (Gezani)", b)]:
        v = ev["vulnerability_benchmark"]
        w(f"\n  {lbl}:")
        w(f"    Vulnerability: {v['vulnerability']} (score={v['vulnerability_score']})")
        w(f"    Coping Cap:    {v['coping_label']} ({v['coping_capacity']})")
        w(f"    Alert Level:   {v['alert_level']} (score={v['alert_score']})")
        w(f"    Magnitude:     {v['magnitude_kmh']} km/h")
        w(f"    Population:    {v['population_millions']}M exposed")
    w(f"\n  >> {comp['vulnerability']['insight']}")

    w(f"\n{'=' * 72}")
    w("  KEY FINDING")
    w(f"{'=' * 72}")
    w(f"  {comp['answer']}")
    w("")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_analysis()
