"""
metrics.py -- Compute all analytical metrics from fetched GDACS data.

Metrics (per assignment):
  Task 2.1: Response Delta (dT)
  Task 2.2: NER Intelligence (NGOs, Governments, Deaths, Losses, Relief Funds)
  Task 2.3: Sentiment Volatility (alarmist vs analytical comparison)
  Task 3.1: Forgotten Crisis Index
  Task 3.2: Vulnerability Benchmark (Coping Capacity vs Alert vs Magnitude)
"""

import re
import statistics
from collections import Counter
from datetime import datetime
from email.utils import parsedate_to_datetime

from textblob import TextBlob

# Try spaCy; graceful fallback if model not downloaded or import fails
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    nlp = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_sent_datetime(sent_str: str) -> datetime | None:
    """Parse the CAP XML <sent> timestamp (ISO-like format)."""
    if not sent_str:
        return None
    try:
        return datetime.fromisoformat(sent_str)
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(sent_str)
    except Exception:
        return None


def _find_media_peak(articles: list[dict]) -> tuple[str, int, datetime | None]:
    """
    Find the day with the most articles.
    Returns (peak_date_str, count, peak_datetime).
    """
    day_counts = Counter()
    for art in articles:
        try:
            dt = datetime.fromisoformat(art.get("pubdate", ""))
            day_counts[dt.strftime("%Y-%m-%d")] += 1
        except (ValueError, TypeError):
            pass
    if not day_counts:
        return ("", 0, None)
    peak_day, count = day_counts.most_common(1)[0]
    peak_dt = datetime.strptime(peak_day, "%Y-%m-%d")
    return (peak_day, count, peak_dt)


VULN_MAP = {"high": 3, "medium": 2, "low": 1}
ALERT_MAP = {"red": 3, "orange": 2, "green": 1}


# ── Task 2.1: Response Delta ─────────────────────────────────────────────────

def response_delta(cap_data: dict, articles: list[dict]) -> dict:
    """
    Time between event onset and the media coverage peak day.
    Formula: Delta_T = T_MediaPeak - T_SystemAlert
    Uses `fromdate` (event onset / system alert) over `sent` (last CAP update).
    """
    onset_str = cap_data.get("fromdate") or cap_data.get("sent")
    onset_dt = None
    if onset_str:
        try:
            onset_dt = parsedate_to_datetime(onset_str)
        except Exception:
            onset_dt = _parse_sent_datetime(onset_str)

    sent_dt = _parse_sent_datetime(cap_data.get("sent"))
    peak_day, peak_count, peak_dt = _find_media_peak(articles)

    delta_hours = None
    if onset_dt and peak_dt:
        delta = peak_dt - onset_dt.replace(tzinfo=None)
        delta_hours = round(delta.total_seconds() / 3600, 1)

    return {
        "event_onset": str(onset_dt),
        "cap_sent": str(sent_dt),
        "media_peak_date": peak_day,
        "media_peak_count": peak_count,
        "delta_hours": delta_hours,
    }


# ── Task 3.1: Forgotten Crisis Index ─────────────────────────────────────────

def forgotten_crisis_index(media_count: int, population_millions: float) -> dict:
    """
    News Volume / Population Exposure (in millions).
    Higher → more media per capita ("over-covered").
    Lower → more "forgotten" / "under-reported".
    """
    if population_millions <= 0:
        ratio = None
    else:
        ratio = round(media_count / population_millions, 2)

    return {
        "media_count": media_count,
        "population_millions": population_millions,
        "index": ratio,
    }


# ── Task 2.3: Sentiment Volatility ──────────────────────────────────────────

def sentiment_volatility(articles: list[dict], corpus_text: str = "") -> dict:
    """
    Sentiment analysis on news headlines + corpus.
    Classifies each headline as 'alarmist' or 'analytical' based on
    polarity and subjectivity, then compares tone distribution.

    Alarmist = high subjectivity (>0.4) OR strong negative polarity (<-0.1)
    Analytical = low subjectivity (<0.3) AND near-neutral polarity
    """
    # Headlines
    headline_scores = []
    alarmist_count = 0
    analytical_count = 0
    neutral_count = 0

    for art in articles:
        title = art.get("title", "")
        if title:
            blob = TextBlob(title)
            pol = blob.sentiment.polarity
            sub = blob.sentiment.subjectivity
            headline_scores.append({
                "polarity": pol,
                "subjectivity": sub,
            })
            # Classify tone
            if sub > 0.4 or pol < -0.1:
                alarmist_count += 1
            elif sub < 0.3 and abs(pol) < 0.1:
                analytical_count += 1
            else:
                neutral_count += 1

    # Enriched corpus paragraphs
    corpus_pols = []
    if corpus_text:
        paragraphs = [p.strip() for p in corpus_text.split("\n") if len(p.strip()) > 50]
        for para in paragraphs[:200]:
            corpus_pols.append(TextBlob(para).sentiment.polarity)

    # Combined polarity list
    headline_pols = [s["polarity"] for s in headline_scores]
    all_pols = headline_pols + corpus_pols

    if not all_pols:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0,
                "headline_mean": None, "corpus_mean": None,
                "tone_alarmist": 0, "tone_analytical": 0, "tone_neutral": 0,
                "tone_ratio": None}

    h_mean = round(statistics.mean(headline_pols), 4) if headline_pols else None
    c_mean = round(statistics.mean(corpus_pols), 4) if corpus_pols else None
    mean_p = round(statistics.mean(all_pols), 4)
    std_p = round(statistics.stdev(all_pols), 4) if len(all_pols) > 1 else 0.0

    total_classified = alarmist_count + analytical_count + neutral_count
    tone_ratio = None
    if total_classified > 0:
        tone_ratio = round(alarmist_count / total_classified, 3)

    return {
        "mean": mean_p,
        "std": std_p,
        "min": round(min(all_pols), 4),
        "max": round(max(all_pols), 4),
        "n": len(all_pols),
        "headline_mean": h_mean,
        "corpus_mean": c_mean,
        "tone_alarmist": alarmist_count,
        "tone_analytical": analytical_count,
        "tone_neutral": neutral_count,
        "tone_alarmist_pct": round(alarmist_count / total_classified * 100, 1) if total_classified else 0,
        "tone_analytical_pct": round(analytical_count / total_classified * 100, 1) if total_classified else 0,
        "tone_ratio": tone_ratio,
    }


# ── Task 3.2: Vulnerability Benchmark ────────────────────────────────────────

def vulnerability_benchmark(vulnerability: str, alertlevel: str,
                            magnitude: int | None = None,
                            population_millions: float = 0) -> dict:
    """
    Compare Coping Capacity (inverse of vulnerability) with Alert Level.
    Insight: Did higher coping capacity lead to lower alert despite higher magnitude?

    Coping Capacity = 4 - vulnerability_score (so high vuln = low coping)
    """
    vuln_score = VULN_MAP.get(vulnerability.lower(), 0)
    alert_score = ALERT_MAP.get(alertlevel.lower(), 0)

    # Coping capacity: higher = better prepared
    coping_capacity = (4 - vuln_score) if vuln_score > 0 else None

    ratio = round(vuln_score / alert_score, 2) if alert_score > 0 else None

    # Insight generation
    insight = ""
    if coping_capacity and magnitude and alert_score:
        if coping_capacity >= 3 and alert_score <= 2 and magnitude and magnitude >= 150:
            insight = "High coping capacity may have reduced alert level despite strong magnitude"
        elif coping_capacity <= 1 and alert_score >= 3:
            insight = "Low coping capacity contributed to high alert level"
        elif coping_capacity >= 2 and alert_score >= 3:
            insight = "High alert despite reasonable coping capacity — magnitude dominates"

    return {
        "vulnerability": vulnerability,
        "vulnerability_score": vuln_score,
        "alertlevel": alertlevel,
        "alert_score": alert_score,
        "coping_capacity": coping_capacity,
        "ratio": ratio,
        "magnitude_kmh": magnitude,
        "population_millions": population_millions,
        "insight": insight,
    }


# ── Task 2.2: NER Intelligence ───────────────────────────────────────────────

# Known humanitarian organisations (multi-word safe + short acronyms need \b)
_KNOWN_ORGS_LONG = [
    "Red Cross", "Red Crescent", "Oxfam", "Save the Children",
    "Doctors Without Borders", "World Vision", "Action Contre la Faim",
    "European Union", "United Nations", "European Commission",
    "Indian Army", "Indian Navy",
]
_KNOWN_ORGS_SHORT = [
    "UNICEF", "UNDP", "UNHCR", "WHO", "WFP", "OCHA", "IFRC", "ICRC",
    "MSF", "CARE", "NDRF", "NDMA", "FEMA", "BNGRC", "USAID", "EU", "UN",
]

# Casualty patterns
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

# Damage/loss patterns
_DAMAGE_RE = re.compile(
    r"(\$[\d,.]+\s*(?:billion|million|bn|m)?)\s+(?:in\s+)?(?:damage|loss|destruction|cost)"
    r"|(?:damage|loss|cost)(?:s|ed)?\s+(?:of\s+|estimated\s+at\s+|worth\s+)?(\$[\d,.]+\s*(?:billion|million|bn|m)?)"
    r"|(\d[\d,]*)\s+(?:homes?|houses?|buildings?|structures?)\s+(?:destroy(?:ed)?|damag(?:ed)?|flatten(?:ed)?)",
    re.IGNORECASE,
)

# Improved money regex — filters out false matches
_MONEY_RE = re.compile(
    r"(?:"
    r"\$\s?[\d,]+(?:\.\d+)?\s*(?:million|billion|bn|m|k|trillion)?"
    r"|EUR\s?[\d,]+(?:\.\d+)?\s*(?:million|billion)?"
    r"|USD\s?[\d,]+(?:\.\d+)?\s*(?:million|billion|crore|lakh)?"
    r"|Rs\.?\s?[\d,]+(?:\.\d+)?\s*(?:crore|lakh|million|billion)?"
    r"|[\d,]+(?:\.\d+)?\s*(?:million|billion)\s*(?:dollars|euros|usd|eur)"
    r")",
    re.IGNORECASE,
)


def _clean_money(mentions: list[str]) -> list[str]:
    """Filter out false positive money matches (too short, just currency codes)."""
    cleaned = []
    for m in mentions:
        m = m.strip()
        # Skip if too short or just a currency prefix
        if len(m) < 4:
            continue
        # Skip if it's just "rs," or "eur," etc.
        if re.match(r"^(rs|eur|usd)[.,\s]*$", m, re.IGNORECASE):
            continue
        # Skip amounts less than $100 (noise)
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


def ner_intelligence(articles: list[dict], corpus_text: str = "") -> dict:
    """
    Task 2.2: Extract entities from headlines + corpus.
    - Organizations (NGOs, Governments, Private Agencies)
    - Deaths/casualties
    - Damage/losses
    - Relief funds (money mentions)
    """
    org_counter: Counter = Counter()
    money_mentions: list[str] = []
    death_mentions: list[str] = []
    damage_mentions: list[str] = []

    # Gather all text
    all_texts = []
    for art in articles:
        if art.get("title"):
            all_texts.append(art["title"])
        if art.get("description"):
            all_texts.append(art["description"])
    if corpus_text:
        all_texts.append(corpus_text)

    combined = "\n".join(all_texts)

    if SPACY_AVAILABLE and nlp:
        for doc in nlp.pipe(all_texts[:500], batch_size=50):
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    org_counter[ent.text] += 1
                elif ent.label_ == "MONEY":
                    money_mentions.append(ent.text)
    else:
        # Regex-based NER fallback — word boundaries for short acronyms
        for org in _KNOWN_ORGS_LONG:
            count = len(re.findall(re.escape(org), combined, re.IGNORECASE))
            if count > 0:
                org_counter[org] += count
        for org in _KNOWN_ORGS_SHORT:
            pattern = r'\b' + re.escape(org) + r'\b'
            count = len(re.findall(pattern, combined))
            if count > 0:
                org_counter[org] += count

        # Money extraction
        raw_money = _MONEY_RE.findall(combined)
        money_mentions = _clean_money(list(set(raw_money)))

    # Death toll extraction
    for match in _DEATH_RE.finditer(combined):
        num = match.group(1) or match.group(2) or match.group(3)
        if num:
            death_mentions.append({
                "count": int(num.replace(",", "")),
                "context": combined[max(0, match.start()-20):match.end()+20].strip(),
            })

    # Damage extraction
    for match in _DAMAGE_RE.finditer(combined):
        val = match.group(1) or match.group(2) or match.group(3)
        if val:
            damage_mentions.append({
                "value": val.strip(),
                "context": combined[max(0, match.start()-20):match.end()+20].strip(),
            })

    # Deduplicate and get max death toll
    max_deaths = 0
    if death_mentions:
        max_deaths = max(d["count"] for d in death_mentions)

    return {
        "top_organisations": org_counter.most_common(15),
        "money_mentions": money_mentions[:10],
        "total_orgs_found": sum(org_counter.values()),
        "spacy_used": SPACY_AVAILABLE,
        "deaths_max_reported": max_deaths,
        "death_extractions": len(death_mentions),
        "damage_mentions": [d["value"] for d in damage_mentions][:10],
    }


# ── Aggregate ─────────────────────────────────────────────────────────────────

def compute_all_metrics(event_data: dict) -> dict:
    """Compute all metrics for a single event's fetched data."""
    cap = event_data["cap"]
    media = event_data["media"]
    media_count = event_data["media_count"]
    pop = cap.get("population_millions", 0)

    # Enriched corpus text
    corpus_text = ""
    corpus_data = event_data.get("corpus", {})
    if corpus_data:
        corpus_text = corpus_data.get("corpus", "")

    return {
        "event_id": event_data["event_id"],
        "event_name": cap.get("eventname", ""),
        "response_delta": response_delta(cap, media),
        "forgotten_crisis": forgotten_crisis_index(media_count, pop),
        "sentiment": sentiment_volatility(media, corpus_text),
        "vulnerability_benchmark": vulnerability_benchmark(
            cap.get("vulnerability", ""),
            cap.get("alertlevel", ""),
            magnitude=cap.get("max_wind_kmh"),
            population_millions=pop,
        ),
        "ner": ner_intelligence(media, corpus_text),
    }
