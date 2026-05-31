import os
import shutil
import csv
import pyarrow.parquet as pq


def main():
    os.makedirs('submissions', exist_ok=True)
    # copy pipeline and README
    shutil.copy('run_pipeline.py', 'submissions/run_pipeline.py')
    shutil.copy('README.md', 'submissions/README.md')

    # read prediction columns from parquet
    src = 'data/models/peak_sales_predictions.parquet'
    table = pq.read_table(src, columns=['outlet_id', 'predicted_value'])
    outlet_ids = table.column('outlet_id').to_pylist()
    preds = table.column('predicted_value').to_pylist()

    out_path = 'submissions/InsightAI_predictions.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['outlet_id', 'predicted_peak_sales', 'team_name'])
        for oid, p in zip(outlet_ids, preds):
            writer.writerow([oid, p, 'InsightAI'])

    print('WROTE', out_path)


if __name__ == '__main__':
    main()
