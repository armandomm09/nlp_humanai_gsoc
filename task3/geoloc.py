import pandas as pd
import re
from collections import Counter
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BEARER = os.getenv("BEARER")
# Load classified data with sentiment and risk from Task 2
df = pd.read_csv("../task2/tweets_classified.csv")

# Extract locations from tweet text
locations = []
location_pattern = re.compile(r"\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")
for text in df["text"]:
    match = location_pattern.search(text)
    if match:
        locations.append(match.group(1))

# Geocode locations using Nominatim (OpenStreetMap)
geolocator = Nominatim(user_agent="regional-distress-analysis")
geocoded_locations = {}
for loc in set(locations):  # Use set to avoid redundant geocoding requests
    location = geolocator.geocode(loc)
    if location:
        geocoded_locations[loc] = (location.latitude, location.longitude)

# Generate heatmap points
heatmap_points = []
for loc in locations:
    if loc in geocoded_locations:
        heatmap_points.append(geocoded_locations[loc])

# Create heatmap with Folium
if heatmap_points:
    avg_lat = sum(pt[0] for pt in heatmap_points) / len(heatmap_points)
    avg_lon = sum(pt[1] for pt in heatmap_points) / len(heatmap_points)
    m = folium.Map(location=[avg_lat, avg_lon], tiles="CartoDB dark_matter", zoom_start=2)
    HeatMap(heatmap_points).add_to(m)
    m.save("index.html")
    print(f"Heatmap generated with {len(heatmap_points)} points. File: tweet_heatmap.html")
else:
    print("No geographic locations found for the tweets.")

# Identify top 5 regions with the highest volume of crisis tweets
if locations:
    loc_counts = Counter(locations)
    top5 = loc_counts.most_common(5)
    print("\nTop 5 regions with the highest number of crisis tweets:")
    for place, count in top5:
        print(f"{place}: {count} tweets")
