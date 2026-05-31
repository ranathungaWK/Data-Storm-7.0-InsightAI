import csv
from collections import Counter
from pathlib import Path

root = Path.cwd()
master = root / 'src' / 'bronze' / 'outlet_master.csv'
coords = root / 'src' / 'bronze' / 'outlet_coordinates.csv'
out_dir = root / 'report_assets'
out_dir.mkdir(exist_ok=True)

# master stats
rows = []
with master.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

total = len(rows)
missing_outlet_size = sum(1 for r in rows if (r.get('Outlet_Size') or '').strip()=='' )
missing_cooler = sum(1 for r in rows if (r.get('Cooler_Count') or '').strip()=='' )

# duplicates
ids = [r['Outlet_ID'] for r in rows]
dup_counts = Counter(ids)
duplicates = sum(1 for k,v in dup_counts.items() if v>1)

# coords stats
coord_rows = []
with coords.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        coord_rows.append(r)

coord_total = len(coord_rows)
invalid_coords = 0
for r in coord_rows:
    try:
        lat = float(r.get('Latitude') or 0)
        lon = float(r.get('Longitude') or 0)
        if lat==0 or lon==0 or not (-90<=lat<=90) or not (-180<=lon<=180):
            invalid_coords += 1
    except:
        invalid_coords += 1

# write missing_values.csv
with (out_dir / 'missing_values.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['column','missing_count'])
    writer.writerow(['Outlet_Size', missing_outlet_size])
    writer.writerow(['Cooler_Count', missing_cooler])
    writer.writerow(['Coordinates_Invalid', invalid_coords])

# write duplicates.csv
with (out_dir / 'duplicates.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['issue','count'])
    writer.writerow(['outlet_id_duplicates', duplicates])

# write data_quality_summary.csv
with (out_dir / 'data_quality_summary.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['metric','value'])
    writer.writerow(['master_records', total])
    writer.writerow(['coord_records', coord_total])
    writer.writerow(['rejected_records_estimate', duplicates + invalid_coords + missing_outlet_size])

# sample rejected records (write first few with reason)
with (out_dir / 'rejected_records_sample.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Outlet_ID','reason'])
    # find some missing size
    count=0
    for r in rows:
        if (r.get('Outlet_Size') or '').strip()=='':
            writer.writerow([r.get('Outlet_ID'),'missing_outlet_size'])
            count+=1
            if count>=5: break
    # add some invalid coords
    count2=0
    for r in coord_rows:
        try:
            lat = float(r.get('Latitude') or 0)
            lon = float(r.get('Longitude') or 0)
            if lat==0 or lon==0 or not (-90<=lat<=90) or not (-180<=lon<=180):
                writer.writerow([r.get('Outlet_ID'),'invalid_coordinate'])
                count2+=1
                if count2>=5: break
        except:
            writer.writerow([r.get('Outlet_ID'),'invalid_coordinate'])
            count2+=1
            if count2>=5: break

print('WROTE report_assets/missing_values.csv, duplicates.csv, data_quality_summary.csv, rejected_records_sample.csv')