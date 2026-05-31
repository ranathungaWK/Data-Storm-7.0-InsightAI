# Section A — Data Engineering & Scraping Pipeline

## How we acquired external data

Primary external source: Overpass API (OpenStreetMap). We used programmatic Overpass QL queries to collect POIs around each outlet coordinate.

Example Overpass query pattern used (radius parameter substituted per run):

```
[out:json][timeout:25];
(
  node(around:RADIUS, LAT, LON)["amenity"~"supermarket|school|bank|restaurant|bus_stop|hospital|pharmacy|fuel_station|shopping_mall"];
  way(around:RADIUS, LAT, LON)["amenity"~"supermarket|school|bank|restaurant|bus_stop|hospital|pharmacy|fuel_station|shopping_mall"];
  relation(around:RADIUS, LAT, LON)["amenity"~"supermarket|school|bank|restaurant|bus_stop|hospital|pharmacy|fuel_station|shopping_mall"];
);
out center;
```

We ran queries for radii: 500 (meters) and 1000 (meters) depending on the POI aggregation required.

## POI categories collected

The following categories were extracted and normalized into `poi_type` values:

```
supermarket
school
bank
restaurant
bus_stop
hospital
pharmacy
fuel_station
shopping_mall
```

## Engineered features (collected)

```
poi_count
competitor_count
competitor_count_500m
competitor_count_1km
nearest_competitor_distance
commercial_score
accessibility_score
mobility_score
saturation_score
poi_influence_score
```

## Why each feature matters (one-line explanations)

```
Higher bus stop density -> higher pedestrian traffic
Higher commercial score -> larger catchment demand
Lower competitor distance -> stronger competitive pressure
Higher poi_count -> more footfall and retail activity
Higher saturation_score -> more competitive cannibalization (negative)
```

## Evidence table (feature -> business meaning)

| Feature | Business Meaning |
|---|---|
| `competitor_count` | local competition |
| `accessibility_score` | ease of reaching outlet |
| `mobility_score` | human movement proxy |
| `commercial_score` | business activity proxy |

## Files copied into this section

- Overpass query examples: `section_A/overpass_examples.txt` (generated)
- Notes: see `report_assets/poi_pipeline_diagram.png` for the POI pipeline (placeholder)
