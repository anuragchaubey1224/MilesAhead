import os
import time
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# configuration
INPUT_CSV = "/Users/anuragchaubey/MilesAhead/data/processed/delivery_data_cleaned.csv"
OUTPUT_CSV = "/Users/anuragchaubey/MilesAhead/data/processed/delivery_data_osrm_distance.csv"

OSRM_URL = "http://localhost:5050/route/v1/driving" 
NUM_WORKERS = 4
MAX_RETRIES = 3


# load and clean data
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude'])

print(f" Fresh run started: Total {len(df)} rows to process.\n")

# OSRM fetch logic
def fetch_osrm_distance(row):
    for _ in range(MAX_RETRIES):
        try:
            src = f"{row['Store_Longitude']},{row['Store_Latitude']}"
            dst = f"{row['Drop_Longitude']},{row['Drop_Latitude']}"
            url = f"{OSRM_URL}/{src};{dst}?overview=false"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                distance_m = data['routes'][0]['distance']
                return round(distance_m / 1000, 2)  # km
        except Exception:
            time.sleep(0.5)
    return None  

# processing
start_time = time.time()
print(" Fetching OSRM distances...\n")

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    distances = list(tqdm(
        executor.map(fetch_osrm_distance, [row for _, row in df.iterrows()]),
        total=len(df),
        desc="  Calculating Distances ..."
    ))

df['osrm_road_distance_km'] = distances

# save (overwrite mode)
df.to_csv(OUTPUT_CSV, index=False)

elapsed = round(time.time() - start_time, 2)
print(f"\n✅ Done! Processed {len(df)} rows and saved to → {OUTPUT_CSV}")
print(f"⏱️ Elapsed time: {elapsed} seconds")
