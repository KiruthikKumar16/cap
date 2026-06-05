import requests
import json

def fetch_signals():
    overpass_url = "https://lz4.overpass-api.de/api/interpreter"
    query = """
    [out:json][timeout:60]; 
    ( 
      node["highway"="traffic_signals"](8.7,78.0,8.9,78.2); 
      way["highway"="traffic_signals"](8.7,78.0,8.9,78.2); 
    ); 
    out body; 
    >; 
    out skel qt;
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = requests.post(overpass_url, data={'data': query}, headers=headers, timeout=90)
        response.raise_for_status()
        data = response.json()
        with open('signals_area_verified.json', 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully fetched {len(data.get('elements', []))} elements.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_signals()
