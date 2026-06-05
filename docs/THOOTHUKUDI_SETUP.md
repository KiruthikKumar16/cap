# Thoothukudi Smart City Map Setup

This guide explains how to generate the high-resolution Thoothukudi map used in the Zero-Shot Generalization test.

## 1. Download High-Resolution OSM Data
Run this command to download the high-node-count data directly from the Overpass API:

```bash
curl -X POST "https://overpass-api.de/api/interpreter" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d 'data=[out:xml][timeout:90];(way(8.78,78.10,8.85,78.18)[highway];);(._;>;);out;' \
     -o thoothukudi.osm
```

## 2. Convert to SUMO Network
Transform the OSM XML into a simulation-ready road network.
**Critical Note**: We use the `--lefthand` flag because Thoothukudi follows left-hand traffic rules.

```bash
mkdir -p data/raw && \
netconvert --osm-files thoothukudi.osm \
           --output-file data/raw/thoothukudi.net.xml \
           --geometry.remove --roundabouts.guess --junctions.join --tls.guess-signals \
           --tls.discard-simple --tls.join --lefthand \
           --keep-edges.by-vclass passenger
```

## 3. Generate Research-Grade Traffic Demand
Create 3600 seconds (1 hour) of realistic traffic flow across the city.

```bash
export SUMO_HOME="/usr/share/sumo" && \
python $SUMO_HOME/tools/randomTrips.py -n data/raw/thoothukudi.net.xml \
                                       -r data/raw/thoothukudi.rou.xml \
                                       -e 3600 -p 0.5 --fringe-factor 10
```

## 4. Run the Validation Test
Execute the generalization script to see how the MAPPO-STGNN model performs on this unseen real-world map.

```bash
python scripts/evaluate_generalization.py --checkpoint marl_ppo_traffic.zip
```
