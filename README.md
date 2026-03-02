# Disaster Evolution & Response Pipeline (DSM Assignment 1)

This repository contains an automated data pipeline and dashboard to scrape, analyze, and compare disaster events using Natural Language Processing (NLP). It specifically compares the historical **Cyclone Amphan (2020)** with the recent **Cyclone Gezani (2026)** to evaluate changes in "Media Saturation" and "Humanitarian Impact".

## 🚀 Features
- **Data Fetching:** Pulls structured data from GDACS (CAP XML, GeoJSON) and ReliefWeb APIs.
- **Web Scraping:** Custom BeautifulSoup scrapers to extract full news article bodies, bypassing anti-bot measures.
- **NLP Analysis:** Calculates Response Delta (ΔT), Sentiment Volatility (alarmist vs analytical), Named Entity Recognition (deaths, damage, organizations, relief funds).
- **Comparative Metrics:** Generates a "Forgotten Crisis" Index and Vulnerability Benchmarks.
- **Dashboard:** Interactive Streamlit dashboard with dual-timeline media overlays and resilience radar charts.

## 📂 Project Structure
- `data_fetcher.py`: Extracts summary, impact, and media metadata from GDACS endpoints.
- `text_enrichment.py`: Scrapes and compiles the full text NLP corpus from ReliefWeb, Wikipedia, and external news articles.
- `metrics.py`: Computes the 5 core analytical metrics (Response Delta, Sentiment, NER, Forgotten Crisis, Vulnerability).
- `run_pipeline.py`: The orchestrator that runs the full 3-stage pipeline (Fetch -> Enrich -> Compute) across 6 events, saving to `pipeline_cache.json`.
- `nlp_analysis.py`: Focused comparative analysis script for Event A (Amphan) vs Event B (Gezani). Outputs `nlp_report.txt` and `nlp_results.json`.
- `dashboard.py`: Interactive Streamlit dashboard with "Overview" and "Comparative" views.
- `csv_export.py`: Exports the raw enriched pipeline data to `disaster_events_dataset.csv`.
- `Final_Report.txt`: The final 500-word intelligence brief discussing methodology and comparative insights.

## ⚙️ Setup & Execution

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Data Pipeline:**
   *(Note: This takes ~5-10 minutes as it scrapes dozens of live news articles)*
   ```bash
   python run_pipeline.py
   python nlp_analysis.py
   python csv_export.py
   ```

3. **Launch the Dashboard:**
   ```bash
   streamlit run dashboard.py
   ```
