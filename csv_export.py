"""
csv_export.py -- Export cleaned dataset from pipeline cache to CSV.

Produces a single CSV with consistent headers for both events.
Submission requirement: "Cleaned Dataset" with consistent headers.

Usage:
    python csv_export.py
"""

import json
import csv
import os

CACHE_FILE = "pipeline_cache.json"
OUTPUT_CSV = "disaster_events_dataset.csv"


def export_csv():
    if not os.path.exists(CACHE_FILE):
        print(f"Cache file not found: {CACHE_FILE}")
        print("Run 'python run_pipeline.py' first.")
        return

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for label, ev in data.items():
        cap = ev.get("cap", {})
        met = ev.get("metrics", {})
        cs = ev.get("corpus_summary", {})
        rd = met.get("response_delta", {})
        fc = met.get("forgotten_crisis", {})
        sv = met.get("sentiment", {})
        vb = met.get("vulnerability_benchmark", {})
        nr = met.get("ner", {})
        geo = ev.get("geojson", {})
        sources = cs.get("sources", {})

        rows.append({
            # Identity
            "event_label": label,
            "event_id": ev.get("event_id", ""),
            "event_name": ev.get("event_name", ""),
            "category": "Historical" if "Historical" in label else "Recent",

            # Summary Tab (Task 1.3)
            "magnitude_kmh": cap.get("max_wind_kmh", ""),
            "alert_level": cap.get("alertlevel", ""),
            "country": cap.get("country", ""),

            # Impact Tab (Task 1.3)
            "population_millions": cap.get("population_millions", ""),
            "vulnerability": cap.get("vulnerability", ""),
            "severity_text": cap.get("severity_text", ""),

            # Media Tab (Task 1.3)
            "media_article_count": ev.get("media_count", 0),

            # Event Timing
            "event_onset": rd.get("event_onset", ""),
            "event_fromdate": cap.get("fromdate", ""),
            "event_todate": cap.get("todate", ""),

            # Task 2.1: Response Delta
            "response_delta_hours": rd.get("delta_hours", ""),
            "media_peak_date": rd.get("media_peak_date", ""),
            "media_peak_count": rd.get("media_peak_count", ""),

            # Task 2.2: NER
            "ner_orgs_found": nr.get("total_orgs_found", 0),
            "ner_top_org": nr["top_organisations"][0][0] if nr.get("top_organisations") else "",
            "ner_deaths_max": nr.get("deaths_max_reported", 0),
            "ner_death_mentions": nr.get("death_extractions", 0),
            "ner_damage_mentions": "; ".join(nr.get("damage_mentions", [])[:5]),
            "ner_money_mentions": "; ".join(nr.get("money_mentions", [])[:5]),

            # Task 2.3: Sentiment
            "sentiment_mean": sv.get("mean", ""),
            "sentiment_std": sv.get("std", ""),
            "sentiment_min": sv.get("min", ""),
            "sentiment_max": sv.get("max", ""),
            "sentiment_n": sv.get("n", 0),
            "tone_alarmist_pct": sv.get("tone_alarmist_pct", ""),
            "tone_analytical_pct": sv.get("tone_analytical_pct", ""),

            # Task 3.1: Forgotten Crisis Index
            "forgotten_crisis_index": fc.get("index", ""),

            # Task 3.2: Vulnerability Benchmark
            "vulnerability_score": vb.get("vulnerability_score", ""),
            "coping_capacity": vb.get("coping_capacity", ""),
            "alert_score": vb.get("alert_score", ""),
            "vuln_alert_ratio": vb.get("ratio", ""),
            "vulnerability_insight": vb.get("insight", ""),

            # Text Corpus
            "corpus_total_words": cs.get("total_words", 0),
            "corpus_reliefweb_words": sources.get("reliefweb", {}).get("words", 0),
            "corpus_wikipedia_words": sources.get("wikipedia", {}).get("words", 0),
            "corpus_scraped_words": sources.get("article_scrape", {}).get("words", 0),
            "corpus_gdacs_words": sources.get("gdacs_media", {}).get("words", 0),

            # GeoJSON
            "geojson_source": geo.get("source", ""),
            "geojson_country": geo.get("country", ""),
            "geojson_coordinates": str(geo.get("coordinates", "")),
        })

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Exported {len(rows)} events to {OUTPUT_CSV}")
        print(f"Columns: {len(fieldnames)}")
        print(f"Headers: {', '.join(fieldnames[:10])}...")
    else:
        print("No data to export.")


if __name__ == "__main__":
    export_csv()
