# Context: Disaster Evolution & Intelligence Pipeline (Density-Optimized)

## 1. Event Selection Logic
The pipeline must select the most "data-dense" events within the following parameters to ensure robust NLP and metric analysis.

* **EVENT A (Historical - 2018-2021):** Must have >50 news headlines in Media API.
    * *Top Candidate:* **AMPHAN-20 (ID: 1000667)** - 33M exposure, massive coverage.
* **EVENT B (Recent - Last 12 Months):** Must have confirmed Population Impact & Vulnerability metrics.
    * *Top Candidate:* **GEZANI-26 (ID: 1001256)** - Very high recent humanitarian documentation.

## 2. Dynamic Source Mapping
Instead of static scraping, the pipeline dynamically fetches data from these endpoints:
1.  **Media API:** `getemmnewsbykey`
    * *Target:* Count total items (News Volume) and collect `title` strings (Sentiment/NER).
2.  **GeoJSON Summary:** `geteventdata`
    * *Target:* Magnitude, Alert Level, and Country.
3.  **CAP XML:** `cap.aspx`
    * *Target:* `<population>` for denominator in "Forgotten Crisis" Index and `<vulnerability>` for benchmarking.

## 3. Metric Calculations (Required)
1.  **Response Delta (ΔT):** Time between `sent` (XML) and Media Peak (API).
2.  **"Forgotten Crisis" Index:** Media Count / Population Exposure.
3.  **Sentiment Volatility:** Mean sentiment score of headlines (Historical vs. Recent).
4.  **Vulnerability Benchmark:** Coping Capacity vs. Alert Level.
5.  **NER Intelligence:** Extracting NGO presence and relief fund figures.

## 4. Visual Dashboard (Streamlit/Plotly)
* **Timeline:** News volume over time for both selected IDs.
* **Radar:** Normalized Comparison of Magnitude, Exposure, News Count, and Vulnerability.