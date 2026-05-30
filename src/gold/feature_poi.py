from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree

from src.utils.logger import get_logger

logger = get_logger("gold.feature_poi_local")


# ============================================================
# PATHS
# ============================================================

OUTLET_PATH = Path("data/silver/cleaned/coordinates_cleaned.parquet")
POI_PATH    = Path("data/overture/sri_lanka_places.parquet")
OUTPUT_PATH = Path("data/gold/poi_features.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

SEARCH_RADIUS_METERS = 500
DECAY_RADIUS_METERS  = 1000
DECAY_SCALE_METERS   = 200


# ============================================================
# CATEGORY MAP
# Built from actual basic_category + taxonomy hierarchy values
# found in sri_lanka_places.parquet.
#
# Strategy: use basic_category (fastest, already flat string).
# For anything not covered, fall back to taxonomy.primary.
# ============================================================

CATEGORY_MAP = {

    # ── RESTAURANT ──────────────────────────────────────────
    "restaurant":                   "restaurant",
    "casual_eatery":                "restaurant",
    "fast_food_restaurant":         "restaurant",
    "cafe":                         "restaurant",
    "coffee_shop":                  "restaurant",
    "bar":                          "restaurant",
    "bakery":                       "restaurant",
    "food_and_drink":               "restaurant",
    "food_court":                   "restaurant",
    "food_truck_stand":             "restaurant",
    "non_alcoholic_beverage_venue": "restaurant",
    "buffet_restaurant":            "restaurant",
    "seafood_restaurant":           "restaurant",
    "pizza_restaurant":             "restaurant",
    "sri_lankan_restaurant":        "restaurant",
    "indian_restaurant":            "restaurant",
    "chinese_restaurant":           "restaurant",
    "burger_restaurant":            "restaurant",
    "sandwich_shop":                "restaurant",
    "tea_room":                     "restaurant",
    "smoothie_juice_bar":           "restaurant",
    "ice_cream_shop":               "restaurant",
    "dessert_shop":                 "restaurant",
    "patisserie_cake_shop":         "restaurant",
    "donut_shop":                   "restaurant",
    "bubble_tea_shop":              "restaurant",
    "bar_and_grill_restaurant":     "restaurant",
    "sports_bar":                   "restaurant",
    "gastropub":                    "restaurant",
    "pub":                          "restaurant",
    "hotel_bar":                    "restaurant",
    "beach_bar":                    "restaurant",
    "cocktail_bar":                 "restaurant",
    "beer_garden":                  "restaurant",
    "bistro":                       "restaurant",
    "diner":                        "restaurant",
    "cafeteria":                    "restaurant",
    "caterer":                      "restaurant",
    "night_market":                 "restaurant",

    # ── SHOPPING ─────────────────────────────────────────────
    "shopping":                          "shopping",
    "fashion_and_apparel_store":         "shopping",
    "electronics_store":                 "shopping",
    "hardware_home_and_garden_store":    "shopping",
    "food_and_beverage_store":           "shopping",
    "books_music_and_video_store":       "shopping",
    "flowers_and_gifts_store":           "shopping",
    "arts_crafts_and_hobby_store":       "shopping",
    "personal_care_and_beauty_store":    "shopping",
    "sporting_goods_store":              "shopping",
    "vehicle_parts_store":               "shopping",
    "shopping_mall":                     "shopping",
    "specialty_store":                   "shopping",
    "warehouse_club_store":              "shopping",
    "animal_and_pet_store":              "shopping",
    "pharmacy_and_drug_store":           "shopping",
    "grocery_store":                     "shopping",
    "convenience_store":                 "shopping",
    "supermarket":                       "shopping",
    "department_store":                  "shopping",
    "furniture_store":                   "shopping",
    "home_goods_store":                  "shopping",
    "home_improvement_store":            "shopping",
    "home_decor_store":                  "shopping",
    "appliance_store":                   "shopping",
    "shoe_store":                        "shopping",
    "jewelry_store":                     "shopping",
    "toy_store":                         "shopping",
    "toys_and_games_store":              "shopping",
    "gift_shop":                         "shopping",
    "souvenir_store":                    "shopping",
    "discount_store":                    "shopping",
    "outlet_store":                      "shopping",
    "clothing_store":                    "shopping",
    "bookstore":                         "shopping",
    "market":                            "shopping",
    "farmers_market":                    "shopping",
    "flea_market":                       "shopping",
    "public_market":                     "shopping",
    "duty_free_store":                   "shopping",
    "second_hand_store":                 "shopping",
    "liquor_store":                      "shopping",
    "candy_store":                       "shopping",
    "health_food_store":                 "shopping",
    "eyewear_store":                     "shopping",
    "mobile_phone_store":                "shopping",
    "computer_store":                    "shopping",
    "auto_parts_store":                  "shopping",
    "vitamin_and_supplement_store":      "shopping",
    "saree_shop":                        "shopping",

    # ── HEALTHCARE ───────────────────────────────────────────
    "hospital":                          "healthcare",
    "outpatient_care_facility":          "healthcare",
    "pharmacy":                          "healthcare",
    "pharmacy_and_drug_store":           "healthcare",
    "dental_clinic":                     "healthcare",
    "doctor":                            "healthcare",
    "doctors_office":                    "healthcare",
    "pediatric_clinic":                  "healthcare",
    "emergency_or_urgent_care_facility": "healthcare",
    "animal_hospital":                   "healthcare",
    "diagnostics_imaging_or_lab_service":"healthcare",
    "dialysis_clinic":                   "healthcare",
    "medical_service":                   "healthcare",
    "medical_spa":                       "healthcare",
    "primary_care_or_general_clinic":    "healthcare",
    "specialized_medical_facility":      "healthcare",
    "surgery_center":                    "healthcare",
    "eye_care":                          "healthcare",
    "vision_or_eye_care_clinic":         "healthcare",
    "health_retreat":                    "healthcare",
    "health_spa":                        "healthcare",
    "wellness_service":                  "healthcare",
    "behavioral_or_mental_health_clinic":"healthcare",
    "fertility_clinic":                  "healthcare",
    "hospice":                           "healthcare",
    "maternity_center":                  "healthcare",
    "rehabilitation_center":             "healthcare",
    "physical_therapy":                  "healthcare",
    "mental_health":                     "healthcare",
    "health_and_wellness_club":          "healthcare",

    # ── EDUCATION ────────────────────────────────────────────
    "education":                         "education",
    "place_of_learning":                 "education",
    "college_university":                "education",
    "preschool":                         "education",
    "high_school":                       "education",
    "specialty_school":                  "education",
    "educational_service":               "education",
    "tutoring_service":                  "education",
    "school":                            "education",
    "elementary_school":                 "education",
    "middle_school":                     "education",
    "private_school":                    "education",
    "public_school":                     "education",
    "religious_school":                  "education",
    "language_school":                   "education",
    "art_school":                        "education",
    "music_school":                      "education",
    "cooking_school":                    "education",
    "driving_school":                    "education",
    "vocational_and_technical_school":   "education",
    "library":                           "education",
    "educational_facility":              "education",
    "educational_research_institute":    "education",
    "montessori_school":                 "education",
    "day_care_preschool":                "education",
    "child_care_and_day_care":           "education",
    "bartending_school":                 "education",
    "cosmetology_school":                "education",
    "flight_school":                     "education",
    "engineering_school":                "education",
    "medical_school":                    "education",
    "nursing_school":                    "education",

    # ── TRANSPORT ────────────────────────────────────────────
    "train_station":                          "transport",
    "public_transit_facility_or_service":     "transport",
    "bus_station":                            "transport",
    "airport":                                "transport",
    "travel_and_transportation":              "transport",
    "travel_service":                         "transport",
    "ground_transport_facility_or_service":   "transport",
    "air_transport_facility_or_service":      "transport",
    "rail_facility_or_service":               "transport",
    "b2b_transportation_and_storage_service": "transport",
    "taxi_service":                           "transport",
    "taxi_or_ride_share_service":             "transport",
    "ride_share_service":                     "transport",
    "parking":                                "transport",
    "ferry_boat_company":                     "transport",
    "heliport":                               "transport",
    "seaplane_base":                          "transport",
    "airport_terminal":                       "transport",
    "airport_lounge":                         "transport",
    "airport_shuttle":                        "transport",
    "car_rental_service":                     "transport",
    "vehicle_rental_service":                 "transport",
    "limo_service":                           "transport",
    "courier_and_delivery_service":           "transport",
    "freight_and_cargo_service":              "transport",
    "shipping_center":                        "transport",
    "shipping_or_delivery_service":           "transport",
    "post_office":                            "transport",

    # ── ACCOMMODATION ────────────────────────────────────────
    "hotel":               "accommodation",
    "lodging":             "accommodation",
    "resort":              "accommodation",
    "hostel":              "accommodation",
    "bed_and_breakfast":   "accommodation",
    "private_lodging":     "accommodation",
    "guest_house":         "accommodation",
    "motel":               "accommodation",
    "inn":                 "accommodation",
    "lodge":               "accommodation",
    "cabin":               "accommodation",
    "cottage":             "accommodation",
    "apartment":           "accommodation",
    "beach_resort":        "accommodation",
    "service_apartment":   "accommodation",
    "holiday_rental_home": "accommodation",
    "country_house":       "accommodation",

    # ── ENTERTAINMENT ────────────────────────────────────────
    "gym":                       "entertainment",
    "sport_or_fitness_facility": "entertainment",
    "sport_or_recreation_club":  "entertainment",
    "historic_site":             "entertainment",
    "park":                      "entertainment",
    "beach":                     "entertainment",
    "arts_and_entertainment":    "entertainment",
    "museum":                    "entertainment",
    "movie_theater":             "entertainment",
    "amusement_park":            "entertainment",
    "bowling_alley":             "entertainment",
    "casino":                    "entertainment",
    "karaoke_venue":             "entertainment",
    "gaming_venue":              "entertainment",
    "escape_room":               "entertainment",
    "comedy_club":               "entertainment",
    "dance_club":                "entertainment",
    "dance_studio":              "entertainment",
    "nightlife_venue":           "entertainment",
    "event_venue":               "entertainment",
    "performing_arts_venue":     "entertainment",
    "music_venue":               "entertainment",
    "theatre_venue":             "entertainment",
    "art_gallery":               "entertainment",
    "zoo":                       "entertainment",
    "aquarium":                  "entertainment",
    "water_park":                "entertainment",
    "sports_and_recreation":     "entertainment",
    "golf_course":               "entertainment",
    "swimming_pool":             "entertainment",
    "tennis_court":              "entertainment",
    "national_park":             "entertainment",
    "botanical_garden":          "entertainment",
    "nature_reserve":            "entertainment",
    "adventure_sports_center":   "entertainment",
    "fitness_studio":            "entertainment",
    "yoga_studio":               "entertainment",
    "pilates_studio":            "entertainment",
    "laser_tag":                 "entertainment",
    "paintball":                 "entertainment",
    "go_kart_track":             "entertainment",
    "arcade":                    "entertainment",
    "skate_park":                "entertainment",
    "sport_court":               "entertainment",
    "sport_field":               "entertainment",
    "stadium":                   "entertainment",
    "auditorium":                "entertainment",
    "fairgrounds":               "entertainment",
    "festival_venue":            "entertainment",
    "hot_springs":               "entertainment",
    "wildlife_sanctuary":        "entertainment",
    "science_attraction":        "entertainment",
    "amusement_attraction":      "entertainment",
}

TARGET_BUCKETS = [
    "restaurant",
    "shopping",
    "healthcare",
    "education",
    "transport",
    "accommodation",
    "entertainment",
]


# ============================================================
# LOAD OUTLETS
# ============================================================

logger.info("Loading outlet coordinates...")
outlets = pd.read_parquet(OUTLET_PATH)
outlets_gdf = gpd.GeoDataFrame(
    outlets,
    geometry=gpd.points_from_xy(outlets.longitude, outlets.latitude),
    crs="EPSG:4326",
)
logger.info(f"Loaded outlets: {len(outlets_gdf):,}")


# ============================================================
# LOAD & FILTER POIs
# ============================================================

logger.info("Loading Overture POIs...")
pois = gpd.read_parquet(POI_PATH)
logger.info(f"Loaded POIs raw: {len(pois):,}")

# Use basic_category (flat string, already clean in this dataset)
# Fall back to taxonomy.primary for rows where basic_category is null
pois["raw_category"] = pois["basic_category"].str.lower().fillna("")

null_mask = pois["raw_category"] == ""
if null_mask.sum() > 0:
    fallback = pois.loc[null_mask, "taxonomy"].apply(
        lambda x: x.get("primary", "").lower() if isinstance(x, dict) else ""
    )
    pois.loc[null_mask, "raw_category"] = fallback
    logger.info(f"Filled {null_mask.sum():,} nulls from taxonomy.primary")

pois["poi_category"] = pois["raw_category"].map(CATEGORY_MAP)

matched = pois["poi_category"].notna().sum()
logger.info(f"POIs matched to a bucket: {matched:,} / {len(pois):,}")

# Log unmatched top categories so you can extend the map if needed
unmatched_top = (
    pois.loc[pois["poi_category"].isna(), "raw_category"]
    .value_counts()
    .head(20)
)
logger.info(f"Top unmatched categories (extend CATEGORY_MAP if important):\n{unmatched_top.to_string()}")

pois = pois[pois["poi_category"].notna()].copy()
logger.info(f"Filtered POIs after mapping: {len(pois):,}")


# ============================================================
# PROJECT CRS
# ============================================================

outlets_gdf = outlets_gdf.to_crs(epsg=3857)
pois        = pois.to_crs(epsg=3857)
outlet_points = outlets_gdf.geometry.copy()


# ============================================================
# BUFFER OUTLETS FOR SPATIAL JOIN
# ============================================================

outlets_gdf["geometry"] = outlets_gdf.geometry.buffer(SEARCH_RADIUS_METERS)
logger.info(f"Created outlet buffers ({SEARCH_RADIUS_METERS}m)")


# ============================================================
# DISTANCE-DECAY FEATURES  (KDTree, DECAY_RADIUS_METERS)
# ============================================================

logger.info("Computing distance-decay POI influence scores...")

poi_coords    = np.column_stack((pois.geometry.x.to_numpy(), pois.geometry.y.to_numpy()))
outlet_coords = np.column_stack((outlet_points.x.to_numpy(), outlet_points.y.to_numpy()))
poi_categories = pois["poi_category"].to_numpy()

poi_tree       = cKDTree(poi_coords)
neighbor_pairs = poi_tree.query_ball_point(outlet_coords, r=DECAY_RADIUS_METERS)

poi_influence_scores        = []
accessibility_weighted_scores = []

for i, poi_idx_list in enumerate(neighbor_pairs):
    if not poi_idx_list:
        poi_influence_scores.append(0.0)
        accessibility_weighted_scores.append(0.0)
        continue

    outlet_xy  = outlet_coords[i]
    nearby_xy  = poi_coords[poi_idx_list]
    distances  = np.sqrt(((nearby_xy - outlet_xy) ** 2).sum(axis=1))
    weights    = np.exp(-distances / DECAY_SCALE_METERS)

    poi_influence_scores.append(float(weights.sum()))

    transport_mask = np.array([poi_categories[idx] == "transport" for idx in poi_idx_list])
    accessibility_weighted_scores.append(float(weights[transport_mask].sum()))

feature_df_decay = pd.DataFrame({
    "outlet_id":                    outlets_gdf["outlet_id"].to_numpy(),
    "poi_influence_score":          poi_influence_scores,
    "accessibility_weighted_score": accessibility_weighted_scores,
})


# ============================================================
# SPATIAL JOIN  (count POIs within SEARCH_RADIUS_METERS)
# ============================================================

logger.info("Running spatial join...")
joined = gpd.sjoin(pois, outlets_gdf, predicate="intersects", how="inner")
logger.info(f"Spatial matches: {len(joined):,}")

if len(joined) == 0:
    logger.warning("Spatial join returned 0 matches — count features will be zero.")


# ============================================================
# AGGREGATE RAW COUNTS PER BUCKET
# ============================================================

if len(joined) > 0:
    count_df = (
        joined.groupby(["outlet_id", "poi_category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
else:
    count_df = pd.DataFrame({"outlet_id": outlets_gdf["outlet_id"].to_numpy()})

# Ensure every bucket column exists
for col in TARGET_BUCKETS:
    if col not in count_df.columns:
        count_df[col] = 0

# Merge decay scores — right join so ALL outlets appear even with 0 counts
feature_df = feature_df_decay.merge(count_df, on="outlet_id", how="left")
for col in TARGET_BUCKETS:
    if col not in feature_df.columns:
        feature_df[col] = 0


# ============================================================
# DERIVED SCORES
# ============================================================

feature_df["commercial_score"]    = feature_df["shopping"] + feature_df["restaurant"] + feature_df["accommodation"]
feature_df["accessibility_score"] = feature_df["transport"]
feature_df["education_score"]     = feature_df["education"]
feature_df["healthcare_score"]    = feature_df["healthcare"]
feature_df["lifestyle_score"]     = feature_df["entertainment"]


# ============================================================
# FINAL COLUMN ORDER & NULL HANDLING
# ============================================================

final_cols = (
    ["outlet_id"]
    + TARGET_BUCKETS
    + [
        "poi_influence_score",
        "accessibility_weighted_score",
        "commercial_score",
        "accessibility_score",
        "education_score",
        "healthcare_score",
        "lifestyle_score",
    ]
)
feature_df = feature_df[final_cols]

numeric_cols = feature_df.select_dtypes(include=["number"]).columns
feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0)

# Quick sanity log
non_zero = (feature_df[numeric_cols] > 0).sum()
logger.info(f"Final shape: {feature_df.shape}")
logger.info(f"Non-zero outlet counts per feature:\n{non_zero.to_string()}")


# ============================================================
# SAVE
# ============================================================

feature_df.to_parquet(OUTPUT_PATH, index=False)
logger.info(f"Saved POI features -> {OUTPUT_PATH}")