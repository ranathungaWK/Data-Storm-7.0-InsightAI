import pyarrow.parquet as pq


FILES = [
    'data/models/peak_sales_predictions.parquet',
    'data/models/model_comparison.parquet',
    'data/models/outlet_potential_scores.parquet',
    'data/models/outlet_potential_predictions.parquet',
    'data/models/model_c_market_demand_predictions.parquet',
    'data/models/model_c_opportunity_ranking.parquet',
    'data/silver/cleaned/outlet_master_cleaned.parquet',
    'data/silver/cleaned/coordinates_cleaned.parquet',
    'data/silver/cleaned/poi_cleaned.parquet',
]


def main():
    for path in FILES:
        print(f'\nFILE: {path}')
        try:
            schema = pq.read_schema(path)
            print(schema)
        except Exception as exc:
            print(f'ERROR: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
    main()
