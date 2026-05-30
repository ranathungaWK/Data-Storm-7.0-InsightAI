import pandas as pd

txn = pd.read_parquet('data/silver/cleaned/transactions_cleaned.parquet')
master = pd.read_parquet('data/silver/cleaned/outlet_master_cleaned.parquet')
coord = pd.read_parquet('data/silver/cleaned/coordinates_cleaned.parquet')
hol = pd.read_parquet('data/silver/cleaned/holidays_cleaned.parquet')

print('TXN SHAPE:', txn.shape)
print('TXN COLS:', list(txn.columns))
print()
print('MASTER SHAPE:', master.shape)
print('MASTER COLS:', list(master.columns))
print()
print('COORD SHAPE:', coord.shape)
print('COORD COLS:', list(coord.columns))
print()
print('HOL SHAPE:', hol.shape)
print('HOL COLS:', list(hol.columns))
print()
print('UNIQUE OUTLETS IN TXN:', txn['outlet_id'].nunique())
print('DATE RANGE:', txn['date'].min(), 'to', txn['date'].max())
print('UNIQUE SKUS:', txn['sku_id'].nunique())
print('UNIQUE DISTRIBUTORS:', txn['distributor_id'].nunique())
print()
print('MASTER OUTLET TYPES:')
print(master['outlet_type'].value_counts().to_string())
print()
if 'outlet_size' in master.columns:
    print('MASTER OUTLET SIZES:')
    print(master['outlet_size'].value_counts().to_string())
print()
if 'cooler_count' in master.columns:
    print('COOLER DISTRIBUTION:')
    print(master['cooler_count'].value_counts().sort_index().to_string())

# POI features
poi = pd.read_parquet('data/gold/poi_features.parquet')
print()
print('POI SHAPE:', poi.shape)
print('POI COLS:', list(poi.columns))
print('POI HEAD:')
print(poi.head(5).to_string())

# Overture POI data
import os
overture_path = 'data/overture/sri_lanka_places.parquet'
if os.path.exists(overture_path):
    import geopandas as gpd
    pois_data = gpd.read_parquet(overture_path)
    print()
    print('OVERTURE POIS:', len(pois_data))
    if 'basic_category' in pois_data.columns:
        print('POI CATEGORIES:')
        print(pois_data['basic_category'].value_counts().head(20).to_string())
