"""
run_pipeline.py -- Orchestrator with validation gates.

Flow:  FETCH GDACS -> validate_fetch -> ENRICH TEXT -> validate_text
       -> COMPUTE METRICS -> validate_metrics -> CACHE

Usage:
    python -u run_pipeline.py
"""

import json
import sys
import os
from datetime import datetime

# Force unbuffered output so terminal updates in real-time
os.environ["PYTHONUNBUFFERED"] = "1"

from data_fetcher import fetch_all
from text_enrichment import build_nlp_corpus
from metrics import compute_all_metrics

# -- Event Configuration (3 Historical + 3 Recent) ----------------------------

EVENTS = {
    # Historical (2018-2021)
    "Historical_Mangkhut":  (1000498, "Typhoon Mangkhut"),
    "Historical_Idai":      (1000552, "Cyclone Idai"),
    "Historical_Amphan":    (1000667, "Cyclone Amphan"),
    # Recent (last 12 months)
    "Recent_Kalmaegi":      (1001233, "Typhoon Kalmaegi"),
    "Recent_Fytia":         (1001254, "Cyclone Fytia"),
    "Recent_Gezani":        (1001256, "Cyclone Gezani"),
}

CACHE_FILE = "pipeline_cache.json"


# -- JSON serialiser helper ----------------------------------------------------

def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# -- Validation Gates ----------------------------------------------------------

def validate_fetch(label: str, raw: dict) -> list[str]:
    """Gate 1: Check that fetched data has minimum required fields."""
    warnings = []
    cap = raw.get("cap", {})

    if not cap.get("sent"):
        warnings.append("CAP: missing 'sent' timestamp")
    if not cap.get("population_millions") and not cap.get("population_text"):
        warnings.append("CAP: missing population data")
    if not cap.get("vulnerability"):
        warnings.append("CAP: missing vulnerability")
    if not cap.get("alertlevel"):
        warnings.append("CAP: missing alertlevel")

    mc = raw.get("media_count", 0)
    if mc == 0:
        warnings.append("MEDIA: zero articles returned")
    elif mc < 20:
        warnings.append(f"MEDIA: low article count ({mc}) - NLP may be thin")

    # Check text depth in descriptions
    descs = [a.get("description", "") for a in raw.get("media", []) if a.get("description")]
    if descs:
        avg_len = sum(len(d) for d in descs) / len(descs)
        if avg_len < 50:
            warnings.append(f"MEDIA: descriptions very short (avg {avg_len:.0f} chars)")
    else:
        warnings.append("MEDIA: no descriptions available in articles")

    return warnings


def validate_text(label: str, corpus: dict) -> list[str]:
    """Gate 2: Check that text enrichment produced usable NLP corpus."""
    warnings = []
    total = corpus.get("total_words", 0)

    if total == 0:
        warnings.append("TEXT: no text enrichment data at all")
    elif total < 500:
        warnings.append(f"TEXT: very small corpus ({total} words)")

    sources = corpus.get("sources", {})
    if not sources.get("reliefweb", {}).get("available"):
        warnings.append("TEXT: ReliefWeb returned no reports")
    if not sources.get("wikipedia", {}).get("available"):
        warnings.append("TEXT: Wikipedia article not found")

    return warnings


def validate_metrics(label: str, metrics: dict) -> list[str]:
    """Gate 3: Check that computed metrics are populated."""
    warnings = []

    rd = metrics.get("response_delta", {})
    if rd.get("delta_hours") is None:
        warnings.append("METRIC: Response Delta is None (missing onset or media peak)")

    fc = metrics.get("forgotten_crisis", {})
    if fc.get("index") is None:
        warnings.append("METRIC: Forgotten Crisis Index is None (missing population)")

    sv = metrics.get("sentiment", {})
    if sv.get("n", 0) == 0:
        warnings.append("METRIC: Sentiment has zero data points")

    nr = metrics.get("ner", {})
    if nr.get("total_orgs_found", 0) == 0 and not nr.get("money_mentions"):
        warnings.append("METRIC: NER found no entities at all")

    return warnings


def _print_gate(gate_name: str, warnings: list[str]):
    """Print validation gate results."""
    if warnings:
        print(f"    [{gate_name}] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"      [!] {w}")
    else:
        print(f"    [{gate_name}] All checks passed")


# -- Main Pipeline -------------------------------------------------------------

def run():
    results = {}
    all_warnings = {}

    for label, (eid, event_name) in EVENTS.items():
        print(f"\n{'='*60}")
        print(f"  {label}: {event_name} (ID: {eid})")
        print(f"{'='*60}")
        event_warnings = []

        # ── Stage 1: Fetch GDACS data ──
        print(f"\n  [1/3] Fetching GDACS data...")
        try:
            raw = fetch_all(eid)
            print(f"    [OK] CAP XML     : {raw['cap'].get('eventname', '?')}")
            print(f"    [OK] Media API   : {raw['media_count']} articles")
            mc_with_desc = sum(1 for a in raw["media"] if a.get("description"))
            avg_desc = 0
            if mc_with_desc > 0:
                avg_desc = sum(len(a.get("description", "")) for a in raw["media"]
                               if a.get("description")) // mc_with_desc
            print(f"         Descriptions: {mc_with_desc}/{raw['media_count']} articles "
                  f"(avg {avg_desc} chars)")
            print(f"    [OK] GeoJSON     : source={raw['geojson'].get('source', '?')}")
        except Exception as e:
            print(f"    [FAIL] Fetch error: {e}")
            event_warnings.append(f"FATAL: fetch failed - {e}")
            all_warnings[label] = event_warnings
            continue

        # Validate Stage 1
        w1 = validate_fetch(label, raw)
        event_warnings.extend(w1)
        _print_gate("VALIDATE_FETCH", w1)

        # ── Stage 2: Text Enrichment ──
        print(f"\n  [2/3] Enriching text corpus...")
        try:
            corpus = build_nlp_corpus(eid, event_name, raw["media"])
            src = corpus["sources"]
            print(f"    [OK] ReliefWeb   : {src['reliefweb'].get('reports', 0)} reports, "
                  f"{src['reliefweb'].get('words', 0)} words")
            print(f"    [OK] Wikipedia   : {src['wikipedia'].get('words', 0)} words")
            scrp = src.get('article_scrape', {})
            print(f"    [OK] Article Body: {scrp.get('scraped', 0)}/{scrp.get('attempted', 0)} "
                  f"scraped, {scrp.get('words', 0)} words")
            print(f"    [OK] GDACS Media : {src['gdacs_media'].get('words', 0)} words")
            print(f"    --- Total corpus : {corpus['total_words']} words")
        except Exception as e:
            print(f"    [WARN] Enrichment error: {e}")
            corpus = {"total_words": 0, "corpus": "", "sources": {}}

        # Validate Stage 2
        w2 = validate_text(label, corpus)
        event_warnings.extend(w2)
        _print_gate("VALIDATE_TEXT", w2)

        raw["corpus"] = corpus

        # ── Stage 3: Compute Metrics ──
        print(f"\n  [3/3] Computing metrics...")
        try:
            metrics = compute_all_metrics(raw)

            rd = metrics["response_delta"]
            fc = metrics["forgotten_crisis"]
            sv = metrics["sentiment"]
            vb = metrics["vulnerability_benchmark"]
            nr = metrics["ner"]

            print(f"    [OK] Response Delta (dT)      : {rd['delta_hours']} hours "
                  f"(onset: {rd.get('event_onset', '?')})")
            print(f"    [OK] Forgotten Crisis Index    : {fc['index']}")
            print(f"    [OK] Sentiment (mean/std)      : {sv['mean']} / {sv['std']}")
            print(f"         Tone: {sv.get('tone_alarmist_pct', 0)}% alarmist, "
                  f"{sv.get('tone_analytical_pct', 0)}% analytical")
            print(f"    [OK] Vulnerability Benchmark   : coping={vb.get('coping_capacity')}, "
                  f"alert={vb['alertlevel']}")
            if vb.get("insight"):
                print(f"         Insight: {vb['insight']}")
            print(f"    [OK] NER Orgs Found            : {nr['total_orgs_found']}")
            if nr["top_organisations"]:
                top5 = nr["top_organisations"][:5]
                print(f"         Top: {', '.join(f'{n} ({c})' for n, c in top5)}")
            print(f"         Deaths: {nr.get('deaths_max_reported', 0)} max reported "
                  f"({nr.get('death_extractions', 0)} mentions)")
            if nr.get("damage_mentions"):
                print(f"         Damage: {nr['damage_mentions'][:3]}")
        except Exception as e:
            print(f"    [FAIL] Metrics error: {e}")
            event_warnings.append(f"FATAL: metrics failed - {e}")
            all_warnings[label] = event_warnings
            continue

        # Validate Stage 3
        w3 = validate_metrics(label, metrics)
        event_warnings.extend(w3)
        _print_gate("VALIDATE_METRICS", w3)

        all_warnings[label] = event_warnings

        # ── Store Results ──
        media_serialisable = []
        for art in raw["media"]:
            art_copy = {k: v for k, v in art.items() if k != "pubdate_dt"}
            media_serialisable.append(art_copy)

        # Corpus: store full text plus summaries
        corpus_for_cache = {
            "total_words": corpus.get("total_words", 0),
            "sources": corpus.get("sources", {}),
            "corpus_text": corpus.get("corpus", ""),  # Full text for NLP
        }

        results[label] = {
            "event_id": eid,
            "event_name": event_name,
            "cap": raw["cap"],
            "media_count": raw["media_count"],
            "media": media_serialisable,
            "geojson": raw["geojson"],
            "corpus_summary": corpus_for_cache,
            "metrics": metrics,
        }

    # ── Write cache ──
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default, ensure_ascii=False)

    # ── Final Report ──
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE - {len(results)}/{len(EVENTS)} events processed")
    print(f"  Cache -> {CACHE_FILE}")
    print(f"{'='*60}")

    total_warnings = sum(len(w) for w in all_warnings.values())
    if total_warnings > 0:
        print(f"\n  WARNING SUMMARY ({total_warnings} total):")
        for label, ws in all_warnings.items():
            if ws:
                print(f"    {label}: {len(ws)} warnings")
                for w in ws:
                    print(f"      - {w}")
    else:
        print(f"\n  All validation gates passed for all events!")

    return results


if __name__ == "__main__":
    run()
