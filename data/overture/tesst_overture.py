import geopandas as gpd

df = gpd.read_parquet(
    "data/overture/sri_lanka_places.parquet"
)

print(df.shape)
print(df.columns)
print(df.head())