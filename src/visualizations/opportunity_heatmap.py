from pathlib import Path

import folium
import pandas as pd

from folium.plugins import HeatMap

from src.utils.logger import get_logger

logger = get_logger(
    "visualization.opportunity_heatmap"
)


# ============================================================
# PATHS
# ============================================================

COORD_PATH = Path(
    "data/silver/cleaned/coordinates_cleaned.parquet"
)

OPPORTUNITY_PATH = Path(
    "data/models/opportunity_recommendations.parquet"
)

OUTPUT_PATH = Path(
    "reports/maps/opportunity_heatmap.html"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

logger.info("Loading datasets...")

coord_df = pd.read_parquet(
    COORD_PATH
)

opp_df = pd.read_parquet(
    OPPORTUNITY_PATH
)

logger.info(
    f"Coordinates: {coord_df.shape}"
)

logger.info(
    f"Opportunity: {opp_df.shape}"
)

# ============================================================
# MERGE
# ============================================================

df = coord_df.merge(

    opp_df,

    on="outlet_id",

    how="inner"

)

logger.info(
    f"Merged shape: {df.shape}"
)

# ============================================================
# FILTER TOP OPPORTUNITIES
# ============================================================

top_df = df[
    df["opportunity_score"] >= 70
]

logger.info(
    f"Top opportunities: {len(top_df):,}"
)

# ============================================================
# CREATE BASE MAP
# ============================================================

sri_lanka_map = folium.Map(

    location=[7.8731, 80.7718],

    zoom_start=7,

    tiles="cartodbpositron"

)

# ============================================================
# HEATMAP DATA
# ============================================================

heat_data = [

    [

        row["latitude"],
        row["longitude"],
        row["opportunity_score"]

    ]

    for _, row in top_df.iterrows()

]

# ============================================================
# ADD HEATMAP
# ============================================================

HeatMap(

    heat_data,

    radius=15,

    blur=10,

    max_zoom=12

).add_to(sri_lanka_map)

# ============================================================
# ADD TOP OUTLET MARKERS
# ============================================================

top_20 = top_df.nlargest(
    20,
    "opportunity_score"
)

for _, row in top_20.iterrows():

    popup_text = f"""
    Outlet: {row['outlet_id']}<br>
    Cluster: {row['cluster']}<br>
    Opportunity Score: {row['opportunity_score']:.2f}<br>
    Tier: {row['opportunity_tier']}
    """

    folium.CircleMarker(

        location=[
            row["latitude"],
            row["longitude"]
        ],

        radius=6,

        popup=popup_text,

        fill=True

    ).add_to(sri_lanka_map)

# ============================================================
# SAVE
# ============================================================

sri_lanka_map.save(
    OUTPUT_PATH
)

logger.info(
    f"Saved heatmap -> {OUTPUT_PATH}"
)