from pathlib import Path
import requests

OUTPUT_PATH = Path(
    "data/external/sri-lanka-latest.osm.pbf"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

URL = (
    "https://download.geofabrik.de/"
    "asia/sri-lanka-latest.osm.pbf"
)

print("Downloading Sri Lanka OSM extract...")

response = requests.get(
    URL,
    stream=True
)

with open(OUTPUT_PATH, "wb") as f:

    for chunk in response.iter_content(
        chunk_size=1024 * 1024
    ):

        if chunk:
            f.write(chunk)

print(f"Saved -> {OUTPUT_PATH}")