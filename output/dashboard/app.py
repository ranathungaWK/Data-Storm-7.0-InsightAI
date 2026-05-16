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

    profile_df = pd.read_parquet(
        PROFILE_PATH
    )

    return (

        features_df,

        clusters_df,

        opportunity_df,

        profile_df

    )


(

    features_df,

    clusters_df,

    opportunity_df,

    profile_df

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
    use_container_width=True
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
    use_container_width=True
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
            "opportunity_tier"
        ]
    ],

    use_container_width=True

)

# ============================================================
# CLUSTER PROFILE TABLE
# ============================================================

st.subheader(
    "Cluster Profiles"
)

st.dataframe(
    profile_df,
    use_container_width=True
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
    use_container_width=True
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    "Built with Machine Learning, "
    "Geospatial Intelligence, "
    "and Overture Maps."
)