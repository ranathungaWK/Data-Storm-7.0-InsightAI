from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="Outlet Intelligence Dashboard",

    page_icon="📊",

    layout="wide"

)


# ============================================================
# PATHS
# ============================================================

FEATURE_PATH = Path(
    "data/gold/outlet_features.parquet"
)

CLUSTER_PATH = Path(
    "data/models/outlet_clusters.parquet"
)

OPPORTUNITY_PATH = Path(
    "data/models/opportunity_recommendations.parquet"
)

POTENTIAL_PATH = Path(
    "data/models/outlet_potential_target.parquet"
)

PROFILE_PATH = Path(
    "data/models/cluster_profiles.parquet"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    features_df = pd.read_parquet(
        FEATURE_PATH
    )

    clusters_df = pd.read_parquet(
        CLUSTER_PATH
    )

    opportunity_df = pd.read_parquet(
        OPPORTUNITY_PATH
    )

    # optional potential/gap
    potential_df = None
    if POTENTIAL_PATH.exists():
        potential_df = pd.read_parquet(POTENTIAL_PATH)

    profile_df = pd.read_parquet(
        PROFILE_PATH
    )

    return (

        features_df,

        clusters_df,

        opportunity_df,

        profile_df,
        potential_df

    )


(

    features_df,

    clusters_df,

    opportunity_df,

    profile_df,
    potential_df

) = load_data()


# ============================================================
# MERGE
# ============================================================

df = (

    features_df

    .merge(
        clusters_df,
        on="outlet_id",
        how="left"
    )

    .merge(
        opportunity_df[
            [
                "outlet_id",
                "opportunity_score",
                "opportunity_tier"
            ]
        ],
        on="outlet_id",
        how="left"
    )

)

# Merge potential if available
if potential_df is not None:
    df = df.merge(
        potential_df[
            ["outlet_id", "max_monthly_volume", "actual_avg_monthly_volume", "gap_volume", "gap_pct"]
        ],
        on="outlet_id",
        how="left"
    )

# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Outlet Intelligence Dashboard"
)

st.markdown(
    """
    AI-powered outlet segmentation,
    opportunity scoring,
    and geo-commercial intelligence.
    """
)

# ============================================================
# KPI ROW
# ============================================================

total_outlets = len(df)

high_opportunity = len(

    df[
        df["opportunity_tier"] == "Very High"
    ]

)

avg_volume = (
    df["avg_monthly_volume"]
    .mean()
)

cluster_count = (
    df["cluster"]
    .nunique()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Outlets",
    f"{total_outlets:,}"
)

col2.metric(
    "High Opportunity",
    f"{high_opportunity:,}"
)

col3.metric(
    "Avg Monthly Volume",
    f"{avg_volume:,.2f}"
)

col4.metric(
    "Clusters",
    cluster_count
)

# Potential KPIs
if potential_df is not None:
    total_upside = df["gap_volume"].clip(lower=0).sum()
    avg_gap_pct = df["gap_pct"].mean()

    col5 = st.columns(6)[4]
    col5.metric("Total Upside Volume", f"{total_upside:,.0f}")
    col6 = st.columns(6)[5]
    col6.metric("Avg Gap %", f"{avg_gap_pct:.2%}")

# Feature snapshot for the new POI / mobility layer
st.subheader("Feature Snapshot")

feature_cols = [
    "poi_influence_score",
    "accessibility_weighted_score",
    "mobility_score",
    "competitor_count_500m",
    "competitor_count_1km",
    "saturation_score",
]

feature_stats = []
for col in feature_cols:
    if col in df.columns:
        feature_stats.append((col, float(df[col].mean())))

feature_metric_cols = st.columns(min(3, max(1, len(feature_stats))))
for idx, (label, value) in enumerate(feature_stats[: len(feature_metric_cols)]):
    feature_metric_cols[idx].metric(label, f"{value:,.2f}")

if feature_stats:
    st.dataframe(
        df[[c for c in feature_cols if c in df.columns]].describe().T,
        use_container_width=True,
    )

# ============================================================
# CLUSTER DISTRIBUTION
# ============================================================

st.subheader(
    "Cluster Distribution"
)

cluster_counts = (

    df["cluster"]
    .value_counts()
    .reset_index()

)

cluster_counts.columns = [
    "cluster",
    "count"
]

fig_cluster = px.bar(

    cluster_counts,

    x="cluster",

    y="count",

    color="cluster",

    title="Outlet Count by Cluster"

)

st.plotly_chart(
    fig_cluster,
    width=900
)

# ============================================================
# OPPORTUNITY SCORE DISTRIBUTION
# ============================================================

st.subheader(
    "Opportunity Score Distribution"
)

fig_score = px.histogram(

    df,

    x="opportunity_score",

    nbins=40,

    title="Opportunity Score Distribution"

)

st.plotly_chart(
    fig_score,
    width=900
)

# ============================================================
# TOP OPPORTUNITIES
# ============================================================

st.subheader(
    "Top Opportunity Outlets"
)

top_df = (

    df

    .sort_values(
        by="opportunity_score",
        ascending=False
    )

    .head(20)

)

st.dataframe(

    top_df[
        [
            "outlet_id",
            "cluster",
            "avg_monthly_volume",
            "opportunity_score",
            "opportunity_tier",
            "max_monthly_volume",
            "gap_volume",
            "gap_pct"
        ]
    ],

    use_container_width=False,
    width=1000

)

# ============================================================
# CLUSTER PROFILE TABLE
# ============================================================

st.subheader(
    "Cluster Profiles"
)

st.dataframe(
    profile_df,
    width=1000
)

# ============================================================
# SCATTER PLOT
# ============================================================

st.subheader(
    "Commercial vs Sales Performance"
)

fig_scatter = px.scatter(

    df,

    x="commercial_score",

    y="avg_monthly_volume",

    color="cluster",

    hover_data=["outlet_id"],

    title="Commercial Score vs Sales"

)

st.plotly_chart(
    fig_scatter,
    width=900
)

# ============================================================
# MOBILITY VS OPPORTUNITY
# ============================================================

if "mobility_score" in df.columns:
    st.subheader("Mobility vs Opportunity")

    fig_mobility = px.scatter(
        df,
        x="mobility_score",
        y="opportunity_score",
        color="cluster",
        hover_data=["outlet_id", "gap_volume"],
        title="Mobility Score vs Opportunity Score",
    )

    st.plotly_chart(fig_mobility, width=900)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    "Built with Machine Learning, "
    "Geospatial Intelligence, "
    "and Overture Maps."
)