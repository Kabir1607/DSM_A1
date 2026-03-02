"""
text_enrichment.py – Fetch rich text content for NLP analysis.

Sources (no Selenium needed):
  1. ReliefWeb     : UN OCHA situation reports via REST API (primary)
  2. Wikipedia     : Full article text via MediaWiki API (secondary)
  3. GDACS Media   : Description text from data_fetcher.py
  4. Article Body  : Full text from news article URLs via newspaper3k
"""

import re
import json
import requests

try:
    from newspaper import Article as NewsArticle
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

HEADERS = {"User-Agent": "DSM-A1-Pipeline/1.0 (academic research)"}
TIMEOUT = 20


# ── 1. Wikipedia API ──────────────────────────────────────────────────────────

# Known Wikipedia article titles for cyclone events
# Format: event_id -> Wikipedia article title
WIKI_TITLES = {
    1000498: "Typhoon_Mangkhut",
    1000552: "Cyclone_Idai",
    1000667: "Cyclone_Amphan",
    1001233: "Typhoon_Kalmaegi_(2025)",
    1001254: "Cyclone_Fytia",
    1001256: "Cyclone_Gezani",
}


def fetch_wikipedia(event_id: int, custom_title: str = None) -> dict:
    """
    Fetch full article text from Wikipedia using the MediaWiki API.
    No Selenium required — returns plain text directly.

    Returns dict with:
        title, extract (full text), word_count, char_count, url
    """
    title = custom_title or WIKI_TITLES.get(event_id)
    if not title:
        return {"source": "wikipedia", "error": f"No Wikipedia title mapped for event {event_id}"}

    # MediaWiki API - get plain text extract
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",    # Plain text, no HTML
        "exsectionformat": "plain",
        "format": "json",
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))

        if "missing" in page:
            # Try alternative title formats
            alt_title = title.replace("_", " ")
            return {
                "source": "wikipedia",
                "title": title,
                "extract": "",
                "word_count": 0,
                "char_count": 0,
                "error": f"Article not found. Try: https://en.wikipedia.org/wiki/{title}",
            }

        extract = page.get("extract", "")
        words = len(extract.split())

        return {
            "source": "wikipedia",
            "title": page.get("title", title),
            "extract": extract,
            "word_count": words,
            "char_count": len(extract),
            "url": f"https://en.wikipedia.org/wiki/{title}",
        }

    except Exception as e:
        return {"source": "wikipedia", "title": title, "error": str(e)}


# ── 2. ReliefWeb API ─────────────────────────────────────────────────────────

def fetch_reliefweb(cyclone_name: str, limit: int = 10) -> dict:
    """
    Fetch situation reports from ReliefWeb (UN OCHA) for a disaster.
    Returns structured report data with full body text.

    ReliefWeb is arguably the best source for disaster text data:
    - Official humanitarian situation reports
    - Standardised format
    - English language
    - Detailed damage, response, and needs assessments
    """
    url = "https://api.reliefweb.int/v1/reports"

    # Try with full name first, then just the proper noun
    queries_to_try = [cyclone_name]
    parts = cyclone_name.split()
    if len(parts) > 1:
        queries_to_try.append(parts[-1])  # e.g. "Amphan" from "Cyclone Amphan"

    all_reports = []
    seen_titles = set()

    for query in queries_to_try:
        params = {
            "appname": "Ashoka_Research_1607k7meRJvYGv8m6M9zMuCx",
            "query[value]": query,
            "limit": limit,
            "fields[include][]": ["title", "body", "date.created", "source", "url"],
            "sort[]": "date:desc",
        }

        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", []):
                fields = item.get("fields", {})
                title = fields.get("title", "")
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                body = fields.get("body", "")
                # Strip HTML tags from body
                clean_body = re.sub(r"<[^>]+>", " ", body)
                clean_body = re.sub(r"\s+", " ", clean_body).strip()
                word_count = len(clean_body.split())

                all_reports.append({
                    "title": title,
                    "body": clean_body,
                    "word_count": word_count,
                    "date": fields.get("date", {}).get("created", ""),
                    "source": [s.get("name", "") for s in fields.get("source", [])],
                    "url": fields.get("url", ""),
                })
        except Exception:
            pass

    total_words = sum(r["word_count"] for r in all_reports)

    return {
        "source": "reliefweb",
        "query": cyclone_name,
        "report_count": len(all_reports),
        "total_words": total_words,
        "reports": all_reports,
    }


# ── 3. GDACS Media Descriptions ──────────────────────────────────────────────

def extract_media_text(media_articles: list[dict]) -> dict:
    """
    Extract and combine text from GDACS media descriptions
    (already fetched by data_fetcher.py). This provides additional
    NLP corpus from news article snippets.
    """
    titles = []
    descriptions = []
    for art in media_articles:
        if art.get("title"):
            titles.append(art["title"])
        if art.get("description"):
            descriptions.append(art["description"])

    combined = "\n".join(descriptions)
    return {
        "source": "gdacs_media",
        "title_count": len(titles),
        "description_count": len(descriptions),
        "total_words": len(combined.split()),
        "total_chars": len(combined),
        "titles": titles,
        "combined_text": combined,
    }


# ── 4. Article Body Scraping (requests + BeautifulSoup) ──────────────────────

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def scrape_article_bodies(media_articles: list[dict], max_articles: int = 30) -> dict:
    """
    Fetch full article body text from media article URLs.
    Uses requests + BeautifulSoup with a browser user-agent.
    Samples up to `max_articles` English articles.
    """
    from bs4 import BeautifulSoup

    # Filter to English-looking articles with links
    candidates = [
        art for art in media_articles
        if art.get("link") and art.get("title", "").isascii()
    ][:max_articles]

    scraped_articles = []
    failed = 0

    for i, art in enumerate(candidates):
        url = art["link"]
        if (i + 1) % 5 == 0 or i == 0:
            print(f"      Scraping article {i+1}/{len(candidates)}...", flush=True)
        try:
            resp = requests.get(
                url, headers={"User-Agent": _BROWSER_UA},
                timeout=10, allow_redirects=True,
            )
            if resp.status_code != 200:
                failed += 1
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove script/style/nav/footer elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Try <article> first, then fall back to all <p> tags
            article = soup.find("article")
            if article:
                paragraphs = article.find_all("p")
            else:
                paragraphs = soup.find_all("p")

            # Filter to substantial paragraphs
            texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40]
            body = "\n".join(texts)

            if len(body) > 200:  # Only keep substantial articles
                scraped_articles.append({
                    "title": art.get("title", ""),
                    "url": url,
                    "body": body,
                    "word_count": len(body.split()),
                })
        except Exception:
            failed += 1

    total_words = sum(a["word_count"] for a in scraped_articles)

    return {
        "source": "article_scrape",
        "scraped": len(scraped_articles),
        "failed": failed,
        "attempted": len(candidates),
        "total_words": total_words,
        "articles": scraped_articles,
    }


# ── Aggregate: Build Full NLP Corpus ──────────────────────────────────────────

def build_nlp_corpus(event_id: int, event_name: str, media_articles: list[dict]) -> dict:
    """
    Build a complete NLP corpus for an event by combining all text sources.
    Returns word counts per source and the merged full text.
    """
    # Fetch from all sources
    wiki = fetch_wikipedia(event_id)
    relief = fetch_reliefweb(event_name)
    media = extract_media_text(media_articles)
    scraped = scrape_article_bodies(media_articles)

    # Merge all text
    corpus_parts = []
    if wiki.get("extract"):
        corpus_parts.append(f"=== WIKIPEDIA: {wiki.get('title', '')} ===\n{wiki['extract']}")
    for r in relief.get("reports", []):
        if r.get("body"):
            corpus_parts.append(f"=== RELIEFWEB: {r['title']} ===\n{r['body']}")
    for a in scraped.get("articles", []):
        if a.get("body"):
            corpus_parts.append(f"=== ARTICLE: {a['title']} ===\n{a['body']}")
    if media.get("combined_text"):
        corpus_parts.append(f"=== GDACS MEDIA DESCRIPTIONS ===\n{media['combined_text']}")

    full_corpus = "\n\n".join(corpus_parts)

    summary = {
        "event_id": event_id,
        "event_name": event_name,
        "sources": {
            "wikipedia": {
                "words": wiki.get("word_count", 0),
                "available": bool(wiki.get("extract")),
                "url": wiki.get("url", ""),
            },
            "reliefweb": {
                "reports": relief.get("report_count", 0),
                "words": relief.get("total_words", 0),
                "available": relief.get("report_count", 0) > 0,
            },
            "article_scrape": {
                "scraped": scraped.get("scraped", 0),
                "attempted": scraped.get("attempted", 0),
                "failed": scraped.get("failed", 0),
                "words": scraped.get("total_words", 0),
                "available": scraped.get("scraped", 0) > 0,
            },
            "gdacs_media": {
                "articles": media.get("description_count", 0),
                "words": media.get("total_words", 0),
                "available": media.get("total_words", 0) > 0,
            },
        },
        "total_words": (
            wiki.get("word_count", 0)
            + relief.get("total_words", 0)
            + scraped.get("total_words", 0)
            + media.get("total_words", 0)
        ),
        "corpus": full_corpus,
    }

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test for both pipeline events
    events = [
        (1000667, "Cyclone Amphan"),
        (1001256, "Cyclone Gezani"),
    ]

    for eid, name in events:
        print(f"\n{'='*60}")
        print(f"  Text Enrichment: {name} (ID: {eid})")
        print(f"{'='*60}")

        # Wikipedia
        wiki = fetch_wikipedia(eid)
        if wiki.get("extract"):
            print(f"  [OK] Wikipedia : {wiki['word_count']} words")
            print(f"       URL       : {wiki.get('url', '')}")
            print(f"       Preview   : {wiki['extract'][:150]}...")
        else:
            print(f"  [--] Wikipedia : {wiki.get('error', 'No data')}")

        # ReliefWeb
        relief = fetch_reliefweb(name)
        print(f"  [OK] ReliefWeb : {relief.get('report_count', 0)} reports, "
              f"{relief.get('total_words', 0)} words")
        if relief.get("reports"):
            for r in relief["reports"][:3]:
                src = ", ".join(r["source"][:2]) if r["source"] else "?"
                print(f"       - [{src}] {r['title'][:70]}...")

        print(f"\n  Total text available: ~{wiki.get('word_count', 0) + relief.get('total_words', 0)} words")
        print(f"  (Plus GDACS media descriptions from pipeline cache)")
