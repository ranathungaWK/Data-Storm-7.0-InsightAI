# Data Storm 7.0 — InsightAI: Project Outcomes Report

> **Competition:** Data Storm 7.0 — Storming Round (36-hour Hackathon)
> **Team:** InsightAI
> **Date:** May 2026
> **Objective:** Estimate the latent maximum monthly volume potential (in liters) for ~20,000 retail outlets for January 2026.

---

## 1. Executive Summary

InsightAI developed a full end-to-end **Outlet Intelligence Platform** that combines a Lakehouse data pipeline (Bronze → Silver → Gold), geospatial intelligence from Overture Maps, unsupervised KMeans clustering, and a multi-factor opportunity scoring engine to identify high-potential retail outlets across Sri Lanka.

### Key Findings

| Metric | Value |
|---|---|
| Total Outlets Analyzed | **20,000** |
| Transaction Records Processed | **2,344,094** |
| Date Range | Jan 2023 – Dec 2025 (36 months) |
| Unique SKUs | 10 |
| Unique Distributors | 10 |
| Outlet Clusters Identified | **8** |
| Very High Opportunity Outlets | **6** |
| High Opportunity Outlets | **10** |
| Average Monthly Volume | **32.20 liters** |
| Total Historical Volume | **124,153,205 liters** |
| Overture POIs Ingested | **125,872** |

---

## 2. Data Architecture — Medallion Lakehouse Pipeline

The project follows a **Medallion Architecture** with three layers:

```
Bronze (Raw CSV → Parquet)  →  Silver (Cleaned + DQ-Checked)  →  Gold (Feature-Engineered)
```

### 2.1 Bronze Layer — Raw Ingestion

All raw CSV files were ingested as-is into Parquet format with zero transformations.

| Dataset | Rows | Columns | File |
|---|---|---|---|
| Transactions History | 2,376,389 | 7 | `transactions_history_final.csv` (169 MB) |
| Outlet Master | 20,000 | 4 | `outlet_master.csv` |
| Outlet Coordinates | 20,000 | 3 | `outlet_coordinates.csv` |
| Holiday List | 349 | 3 | `holiday_list.csv` |
| Distributor Seasonality | 360 | 4 | `distributor_seasonality_details.csv` |

### 2.2 Silver Layer — Forensic Cleaning & Data Quality

A reusable **Data Quality (DQ) Framework** was built with 7 parameterizable check functions:

| Check | Rule ID | Type | Description |
|---|---|---|---|
| Duplicate Detection | DQ_001 | CRITICAL | Composite key duplicate detection |
| Null Detection | DQ_002 | CRITICAL | Mandatory field null/empty detection |
| Referential Integrity | DQ_003 | CRITICAL | Foreign key validation |
| Value Range | DQ_004 | CRITICAL | Min/max boundary checks |
| Format Validation | DQ_005 | CRITICAL | Regex pattern compliance |
| Negative Volumes | FQ_001 | WARNING | Return/reversal tagging (not dropped) |
| Zero Volumes | FQ_002 | INFO | System adjustment tagging |

#### Data Quality Audit Results

| Dataset | Raw Rows | Duplicates Found | Missing Values |
|---|---|---|---|
| Transactions | 2,376,389 | 0 | 0 |
| Outlet Master | 20,000 | 0 | 196 |
| Coordinates | 20,000 | 0 | 0 |
| Holidays | 349 | 93 | 0 |
| Seasonality | 360 | 0 | 0 |

#### Cleaning Outcomes

**Transactions:** 2,344,094 records retained after cleaning. Date column was constructed, censored volume was computed.

**Outlet Master — Typo Corrections Applied:**

| Original Value | Corrected To | Records Affected |
|---|---|---|
| `Bakry` | `Bakery` | 395 |
| `Grocry` | `Grocery` | 390 |
| ` Eatery` (leading space) | `Eatery` | 200 |
| `small` (lowercase) | `Small` | 600 |

**Outlet Type Distribution (Post-Cleaning):**

| Outlet Type | Count |
|---|---|
| Hotel | 2,797 |
| Grocery | 2,768 |
| SMMT | 2,723 |
| Pharmacy | 2,691 |
| Kiosk | 2,691 |
| Bakery | 2,678 |
| Eatery | 2,667 |

**Outlet Size Distribution:**

| Size | Count |
|---|---|
| Small | 9,672 |
| Medium | 5,702 |
| Large | 2,887 |
| Extra Large | 943 |

**Cooler Count Distribution:**

| Coolers | Outlets |
|---|---|
| 0 | 7,040 |
| 1 | 6,004 |
| 2 | 3,984 |
| 3 | 1,020 |
| 4 | 972 |
| 5 | 980 |

**Coordinates:** 19,760 valid coordinates retained. Geo-validated against Sri Lanka bounding box (lat: 5.9–9.9, lon: 79.5–81.9). Lat/lon bucketing applied for spatial aggregation.

**Holidays:** 256 unique holidays retained after deduplication (93 duplicates removed).

---

## 3. Feature Engineering (Gold Layer)

### 3.1 Transaction Features (per outlet)

Aggregated from 2.3M+ cleaned transaction records:

| Feature | Description | Mean | Std | Min | Max |
|---|---|---|---|---|---|
| `total_volume` | Sum of all volume (liters) | 6,207.66 | 10,650.08 | 109.72 | 55,479.57 |
| `avg_monthly_volume` | Average monthly volume | 32.20 | 30.19 | 6.49 | 173.37 |
| `avg_monthly_revenue` | Average monthly bill value | 8,433.01 | 7,892.83 | 2,315.95 | 38,740.54 |
| `volume_std` | Volume standard deviation | 42.04 | 39.77 | 3.85 | 552.85 |
| `volume_cv` | Coefficient of variation | 1.30 | 0.10 | 0.55 | 3.19 |
| `transaction_count` | Total transactions | 117.20 | 85.49 | 15 | 360 |
| `sku_diversity` | Unique SKUs purchased | 9.98 | 0.16 | 7 | 10 |
| `active_months` | Months with activity | 22.53 | 6.24 | 7 | 36 |
| `sales_frequency` | Active months / 12 | 1.88 | 0.52 | 0.58 | 3.00 |

### 3.2 Geospatial Intelligence — Overture Maps POI Features

**125,872 Points of Interest** from Overture Maps were spatially joined to outlet locations within a **300-meter buffer radius**.

**Top POI Categories in Sri Lanka (Overture):**

| Category | Count |
|---|---|
| Hotel | 10,204 |
| Fashion & Apparel Store | 6,686 |
| Restaurant | 6,297 |
| Personal/Beauty Service | 5,521 |
| Professional Service | 5,320 |
| Electronics Store | 4,386 |
| Place of Learning | 3,349 |
| Hardware/Home/Garden | 3,315 |
| Travel Service | 3,263 |

**Derived Spatial Scores:**

| Score | Derived From | Mean |
|---|---|---|
| `commercial_score` | shopping + restaurant + accommodation | 0.22 |
| `accessibility_score` | transport POI count | 0.00 |
| `education_score` | education POI count | 0.00 |
| `healthcare_score` | healthcare POI count | 0.00 |
| `lifestyle_score` | entertainment POI count | 0.00 |
| `mobility_score` | transport + travel | 0.00 |

> **Note:** 1,594 outlets (8.0%) had POIs within 300m radius. The remaining outlets had zero POI presence, which itself is an informative signal for rural/low-density areas.

### 3.3 Outlet Profile Features

Enhanced behavioral features including:
- **Volatility & Stability Score** — Demand consistency measure
- **Peak-to-Average Ratio** — Latent demand headroom proxy
- **Demand Headroom Proxy** — Gap between peak and average performance
- **Consistency Index** — Average volume normalized by volatility
- **Master Data Completeness** — Data quality as a feature (mean: 99.8%)

### 3.4 Seasonality Features

- Calendar decomposition (year, month, day-of-week, week-of-year)
- Holiday proximity features (binary + distance to nearest holiday)
- Weekend indicator
- Monthly seasonality index
- Rolling 7-day and 30-day mean volumes
- Trend signal (short-term vs. long-term momentum)

---

## 4. Outlet Segmentation — KMeans Clustering

### 4.1 Methodology

- **Algorithm:** KMeans (scikit-learn)
- **k = 8** clusters
- **n_init = 20** (multiple random initializations)
- **Features Used:** commercial_score, accessibility_score, lifestyle_score, cooler_count, avg_monthly_volume, avg_monthly_revenue, sku_diversity, active_months
- **Preprocessing:** StandardScaler normalization, missing values filled with 0
- **Dimensionality Reduction:** PCA (2 components) for visualization

### 4.2 Cluster Profiles

| Cluster | Outlets | Avg Monthly Vol | Total Volume | Commercial Score | Sales Frequency | SKU Diversity | Cluster Score |
|---|---|---|---|---|---|---|---|
| **3** | 912 | **137.98** | 47,279.40 | 0.12 | 2.86 | 10.00 | **138.58** |
| **6** | 140 | 28.95 | 5,179.39 | **12.24** | 1.90 | 9.99 | **90.15** |
| **2** | 2,935 | **68.76** | 16,833.27 | 0.14 | 2.55 | 10.00 | 69.46 |
| **0** | 2,670 | 29.34 | 4,074.30 | 0.15 | 2.12 | 10.00 | 30.09 |
| **7** | 3,171 | 29.10 | 3,974.57 | 0.14 | 2.11 | 10.00 | 29.80 |
| **5** | 452 | 13.87 | 561.52 | 0.20 | 1.23 | 8.94 | 14.87 |
| **1** | 5,492 | 13.93 | 736.18 | 0.14 | 1.47 | 10.00 | 14.63 |
| **4** | 4,228 | 13.92 | 739.11 | 0.12 | 1.47 | 10.00 | 14.52 |

### 4.3 Cluster Interpretation

| Cluster | Label | Description |
|---|---|---|
| **3** | 🏆 Premium Performers | Highest volume outlets (138 L/month avg). Small group of 912 elite outlets with strong sales frequency. |
| **6** | 🏙️ Commercial Hotspots | Highest commercial POI density (12.24). Only 140 outlets but located in commercially rich areas. Key opportunity zone. |
| **2** | 📈 Growth Leaders | High volume (69 L/month), large cluster of 2,935. Strong sales consistency (2.55 frequency). |
| **0** | ⚖️ Steady Mid-Tier | Moderate volume (29 L/month), 2,670 outlets. Reliable performance with room for growth. |
| **7** | 📊 Balanced Base | Similar to Cluster 0. Largest mid-tier segment (3,171 outlets). |
| **5** | ⚠️ Low Activity Niche | Smallest average volume (13.87 L), lowest SKU diversity (8.94). 452 underperforming outlets. |
| **1** | 🏪 Mass Market (A) | Largest cluster (5,492). Low volume (13.93 L) but broad market coverage. |
| **4** | 🏪 Mass Market (B) | Second largest (4,228). Nearly identical to Cluster 1 in behavior. |

---

## 5. Opportunity Scoring Engine

### 5.1 Methodology

A multi-factor opportunity score was computed for each outlet:

```
Raw Score = (Volume Gap × 0.6) + (Commercial Score × 10) + (Mobility Score × 5) + (Lifestyle Score × 5)
```

Where:
- **Volume Gap** = Cluster average volume − outlet's actual average volume
- **Commercial/Mobility/Lifestyle Scores** = POI-derived spatial intelligence

Scores were normalized to 0–100 using MinMaxScaler.

### 5.2 Opportunity Tiers

| Tier | Score Range | Outlet Count | % of Total |
|---|---|---|---|
| **Very High** | ≥ 80 | 6 | 0.03% |
| **High** | 60–79 | 10 | 0.05% |
| **Medium** | 40–59 | 20 | 0.10% |
| **Low** | < 40 | 19,964 | 99.82% |

### 5.3 Opportunity Score Distribution

| Statistic | Value |
|---|---|
| Mean | 7.59 |
| Std Dev | 3.28 |
| Median | 7.16 |
| 25th Percentile | 6.87 |
| 75th Percentile | 7.48 |
| Maximum | 100.00 |

### 5.4 Top 10 Opportunity Outlets

| Rank | Outlet ID | Cluster | Avg Monthly Vol (L) | Opportunity Score | Tier |
|---|---|---|---|---|---|
| 1 | OUT_18247 | 6 | 16.44 | **100.00** | Very High |
| 2 | OUT_03556 | 6 | 129.24 | **95.76** | Very High |
| 3 | OUT_05406 | 6 | 29.46 | **95.73** | Very High |
| 4 | OUT_08956 | 6 | 18.73 | **92.48** | Very High |
| 5 | OUT_17774 | 6 | 13.03 | **90.90** | Very High |
| 6 | OUT_06884 | 6 | 23.29 | **84.62** | Very High |
| 7 | OUT_06433 | 6 | 20.73 | 77.80 | High |
| 8 | OUT_06602 | 6 | 8.94 | 77.09 | High |
| 9 | OUT_02546 | 6 | 13.91 | 76.38 | High |
| 10 | OUT_18913 | 6 | 70.58 | 75.42 | High |

> **Key Insight:** All top opportunity outlets belong to **Cluster 6** (Commercial Hotspots). These outlets are in commercially dense areas but have below-average sales volumes — representing the biggest untapped potential.

---

## 6. Potential Scoring (Hidden Opportunity Detection)

A separate **Potential Scoring** model was built to detect hidden growth opportunities using cluster-relative benchmarking.

### 6.1 Expected Performance Formula

```
Expected Volume = (Cluster Avg × 0.40)
                + (Commercial Ratio × Cluster Avg × 0.20)
                + (Mobility Ratio × Cluster Avg × 0.15)
                + (SKU Ratio × Cluster Avg × 0.15)
                + (Frequency Ratio × Cluster Avg × 0.10)
```

### 6.2 Potential Tier Distribution

| Tier | Count |
|---|---|
| Very High (≥80) | 1 |
| High (60–79) | 3 |
| Medium (40–59) | 4 |
| Low (20–39) | 23 |
| Very Low (<20) | 19,969 |

---

## 7. Geospatial Visualization

### 7.1 Opportunity Heatmap

An interactive **Folium heatmap** was generated showing the geographic distribution of high-opportunity outlets across Sri Lanka. The map is centered at coordinates [7.8731, 80.7718] with:

- **Heatmap layer** for all outlets with opportunity score ≥ 70
- **Circle markers** for the top 20 outlets with popup details (outlet ID, cluster, score, tier)
- **Base tiles:** CartoDB Positron (clean, light-themed map)

Output: `reports/maps/opportunity_heatmap.html`

### 7.2 PCA Cluster Visualization

A 2D PCA projection of the 8 outlet clusters was generated, showing clear separation between high-volume clusters (3, 2) and the mass-market base (1, 4).

Output: `reports/figures/cluster_pca.png`

---

## 8. Interactive Dashboard

A **Streamlit dashboard** was built providing real-time exploration of the outlet intelligence data:

### Dashboard Components:
1. **KPI Row** — Total outlets, high-opportunity count, average volume, cluster count
2. **Cluster Distribution** — Bar chart of outlets per cluster
3. **Opportunity Score Distribution** — Histogram of all scores
4. **Top Opportunity Outlets** — Sortable data table of top 20
5. **Cluster Profiles** — Full profile table with all metrics
6. **Commercial vs Sales Scatter** — Interactive scatter plot colored by cluster

**Launch:** `streamlit run output/dashboard/app.py`

---

## 9. Technical Architecture Summary

### 9.1 Pipeline Components

```
src/
├── bronze/           # Raw CSV → Parquet ingestion
├── silver/           # DQ checks, forensic cleaning, quarantine
│   ├── dq_checks.py          # 7 reusable DQ check functions
│   ├── quarantine.py          # QuarantineManager class
│   ├── clean_transactions.py  # Transaction cleaning
│   ├── clean_outlet_master.py # Outlet master cleaning
│   ├── clean_coordinates.py   # Geo-validation
│   └── clean_holidays.py      # Holiday deduplication
├── gold/             # Feature engineering
│   ├── build_outlet_features.py  # Master feature table builder
│   ├── feature_poi.py            # Overture Maps POI features
│   ├── feature_outlet_profile.py # Behavioral profile features
│   ├── feature_transaction.py    # Transaction aggregations
│   └── feature_seasonality.py    # Temporal features
├── models/           # ML models
│   ├── clustering.py                  # KMeans (k=8)
│   ├── cluster_profiles.py           # Cluster profiling
│   ├── opportunity_recommendations.py # Opportunity scoring
│   └── potential_scoring.py           # Hidden potential detection
├── visualizations/   # Charts & maps
│   ├── plot_clusters.py        # PCA visualization
│   └── opportunity_heatmap.py  # Folium heatmap
└── utils/            # Shared utilities
    ├── config.py     # YAML config loader
    ├── logger.py     # Structured logging
    └── io.py         # Parquet/CSV I/O helpers
```

### 9.2 Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn (KMeans, PCA, StandardScaler, MinMaxScaler) |
| Geospatial | GeoPandas, Folium, Overture Maps |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Data Format | Apache Parquet |
| Configuration | YAML |
| Version Control | Git + GitHub |

### 9.3 Data Flow

```
Raw CSVs (Bronze)
    ↓
DQ Checks + Forensic Cleaning (Silver)
    ↓  ↘ Quarantined Records
Feature Engineering (Gold)
    ↓
outlet_features.parquet (20,000 × 30+ features)
    ↓
┌──────────────┬────────────────────┬──────────────────┐
│  Clustering  │ Opportunity Scoring│ Potential Scoring │
│  (KMeans k=8)│ (Multi-factor)     │ (Cluster-relative)│
└──────┬───────┴────────┬───────────┴────────┬─────────┘
       ↓                ↓                    ↓
  8 Segments    Ranked Outlets       Hidden Opportunities
       ↓                ↓                    ↓
       └────────────────┴────────────────────┘
                        ↓
              Dashboard + Heatmap + Report
```

---

## 10. Key Recommendations

### 10.1 Immediate Actions (Quick Wins)

1. **Prioritize Cluster 6 outlets** — All 140 outlets in this commercially rich cluster should receive targeted sales interventions. The top 6 "Very High" opportunity outlets are all in this cluster.

2. **Focus on OUT_18247** (Score: 100) — This outlet has the highest opportunity score with only 16.44 L/month average volume despite being in the most commercially dense area.

3. **Investigate Cluster 3 underperformers** — The 912 premium outlets averaging 138 L/month have peers reaching 173+ L/month. Closing this gap represents significant volume upside.

### 10.2 Strategic Recommendations

4. **Expand cooler deployment** — 35.2% of outlets (7,040) have zero coolers. Cooler installation in high-commercial-score locations could unlock latent demand.

5. **SKU optimization** — Most outlets carry 10 SKUs. Cluster 5 averages only 8.94 — investigate whether limited SKU availability constrains volume.

6. **Geographic expansion** — The POI analysis shows only 8% of outlets have commercial POIs within 300m. Target outlet acquisition in POI-dense areas currently without coverage.

### 10.3 Data-Driven Next Steps

7. **Build predictive model** — Use the cluster profiles and opportunity scores as features in a supervised demand forecasting model for January 2026.

8. **Expand POI radius** — Consider increasing the 300m buffer to 500m–1km to capture broader catchment area effects.

9. **Temporal deep-dive** — Leverage the 36-month transaction history to build outlet-level time series forecasts with seasonality adjustment.

---

## 11. Appendix

### A. Configuration Parameters

| Parameter | Value |
|---|---|
| KMeans Clusters | 8 |
| KMeans n_init | 20 |
| Random Seed | 42 |
| POI Search Radius | 300 meters |
| Geo Bounding Box (Lat) | 5.9 – 9.9 |
| Geo Bounding Box (Lon) | 79.5 – 81.9 |
| Volume Range Check | -10,000 to 10,000 L |
| Outlier Z-Score Threshold | 3.5 |
| Censoring CV Threshold | 0.15 |
| Opportunity Very High | ≥ 80 |
| Opportunity High | ≥ 60 |
| Opportunity Medium | ≥ 40 |

### B. Output Artifacts

| Artifact | Path | Description |
|---|---|---|
| Outlet Features | `data/gold/outlet_features.parquet` | 20,000 × 30+ feature table |
| Cluster Assignments | `data/models/outlet_clusters.parquet` | Outlet → cluster mapping |
| Cluster Profiles | `data/models/cluster_profiles.parquet` | 8 cluster summaries |
| Opportunity Scores | `data/models/opportunity_recommendations.parquet` | Ranked outlet opportunities |
| Potential Scores | `data/models/outlet_potential_scores.parquet` | Hidden potential analysis |
| POI Features | `data/gold/poi_features.parquet` | 1,594 outlet POI profiles |
| PCA Visualization | `reports/figures/cluster_pca.png` | Cluster scatter plot |
| Opportunity Heatmap | `reports/maps/opportunity_heatmap.html` | Interactive Sri Lanka map |
| Dashboard | `output/dashboard/app.py` | Streamlit application |

---

*Report generated by InsightAI — Data Storm 7.0*
*Built with Machine Learning, Geospatial Intelligence, and Overture Maps.*
