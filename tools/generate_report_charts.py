import csv
import matplotlib.pyplot as plt
from pathlib import Path

root = Path.cwd()
assets = root / 'report_assets'
assets.mkdir(exist_ok=True)

# Read feature importance without pandas
fi_path = assets / 'feature_importance.csv'
features = []
importances = []
with fi_path.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            features.append(r['feature'])
            importances.append(float(r['importance']))
        except Exception:
            continue

if features and importances:
    combined = list(zip(features, importances))
    combined.sort(key=lambda x: x[1], reverse=True)
    top10 = combined[:10]
    labels = [c[0] for c in top10][::-1]
    vals = [c[1] for c in top10][::-1]
    # write a simple SVG bar chart for portability
    svg_lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400">']
    maxv = max(vals) if vals else 1
    bar_h = 30
    gap = 10
    left = 200
    for i, (label, val) in enumerate(zip(labels[::-1], vals[::-1])):
        y = 20 + i * (bar_h + gap)
        w = int((val / maxv) * 400)
        svg_lines.append(f'<text x="10" y="{y+20}" font-size="12">{label}</text>')
        svg_lines.append(f'<rect x="{left}" y="{y}" width="{w}" height="{bar_h}" fill="#1f77b4" />')
        svg_lines.append(f'<text x="{left + w + 5}" y="{y+20}" font-size="12">{val:.1f}</text>')
    svg_lines.append('</svg>')
    (assets / 'feature_importance.svg').write_text('\n'.join(svg_lines), encoding='utf-8')

# Model metrics image (simple text) without pandas
metrics_path = assets / 'model_metrics.csv'
metrics_text = ''
try:
    with metrics_path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        # assume header: metric,ModelA_peak_sales
        metrics_text = '\n'.join([f"{r[0]}: {r[1]}" for r in rows])
except Exception:
    metrics_text = 'No metrics available'

metrics_svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="400" height="120">']
for i, line in enumerate(metrics_text.split('\n')):
    metrics_svg.append(f'<text x="10" y="{20 + i*18}" font-size="14">{line}</text>')
metrics_svg.append('</svg>')
(assets / 'model_metrics.svg').write_text('\n'.join(metrics_svg), encoding='utf-8')

# Budget distribution and top100 using csv
alloc_path = root / 'submissions' / 'InsightAI_budget_allocations.csv'
if not alloc_path.exists():
    # fallback to report_assets preview
    alloc_path = assets / 'InsightAI_budget_allocations.csv'

alloc = []
with alloc_path.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            alloc.append((r['Outlet_ID'], float(r['Trade_Spend_Allocation'])))
        except Exception:
            continue

alloc_sorted = sorted(alloc, key=lambda x: x[1], reverse=True)
with (assets / 'top100_allocations.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Outlet_ID','Trade_Spend_Allocation'])
    for outlet_id, val in alloc_sorted[:100]:
        writer.writerow([outlet_id, f"{val:.2f}"])

if alloc:
    values = [v for _, v in alloc]
    # simple SVG histogram (binned)
    bins = 50
    minv, maxv = min(values), max(values)
    width = 600
    height = 300
    counts = [0]*bins
    for v in values:
        idx = int((v - minv) / (maxv - minv + 1e-9) * (bins-1))
        counts[idx] += 1
    maxc = max(counts) if counts else 1
    svgh = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    for i, c in enumerate(counts):
        bar_w = int((c / maxc) * (width-100))
        x = 50 + i * ((width-100)/bins)
        y = height - 20 - (c / maxc) * (height-80)
        h = int((c / maxc) * (height-80))
        svgh.append(f'<rect x="{x}" y="{y}" width="{(width-100)/bins - 1}" height="{h}" fill="#2ca02c" />')
    svgh.append('</svg>')
    (assets / 'budget_distribution.svg').write_text('\n'.join(svgh), encoding='utf-8')

print('WROTE feature_importance.svg, model_metrics.svg, budget_distribution.svg, top100_allocations.csv')
