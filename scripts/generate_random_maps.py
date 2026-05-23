import argparse
import os
import subprocess
import shutil
from pathlib import Path

def generate_maps(output_dir: Path, count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    types = ["grid", "spider", "random"]
    
    sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    netgenerate_bin = Path(sumo_home) / "bin" / "netgenerate"
    if not netgenerate_bin.exists():
        # Fallback to PATH
        netgenerate_bin = "netgenerate"
        
    for i in range(count):
        map_type = types[i % len(types)]
        prefix = f"{map_type}_{i}"
        net_file = output_dir / f"{prefix}.net.xml"
        
        cmd = [str(netgenerate_bin)]
        if map_type == "grid":
            cmd.extend(["--grid", "--grid.x-number", "5", "--grid.y-number", "5"])
        elif map_type == "spider":
            cmd.extend(["--spider", "--spider.arm-number", "5", "--spider.circle-number", "3"])
        else:
            cmd.extend(["--rand", "--rand.iterations", "200"])
            
        cmd.extend(["--output-file", str(net_file), "--no-turnarounds", "true", "--tls.guess", "true", "--tls.guess.threshold", "0"])
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Generated {net_file}")
            
            # Generate random trips to create a route file
            route_file = output_dir / f"{prefix}.rou.xml"
            trips_file = output_dir / f"{prefix}.trips.xml"
            
            # We use randomTrips.py from SUMO_HOME/tools
            sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
            random_trips_script = Path(sumo_home) / "tools" / "randomTrips.py"
            
            if random_trips_script.exists():
                trip_cmd = [
                    "python", str(random_trips_script),
                    "-n", str(net_file),
                    "-e", "3600",
                    "-o", str(trips_file),
                    "--route-file", str(route_file)
                ]
                subprocess.run(trip_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Generated routes {route_file}")
            else:
                print(f"Warning: SUMO_HOME not set or randomTrips.py not found at {random_trips_script}")
                
        except Exception as e:
            print(f"Failed to generate {map_type} map: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate procedural SUMO maps.")
    parser.add_argument("--count", type=int, default=10, help="Number of maps to generate")
    parser.add_argument("--output_dir", type=str, default="data/raw/procedural/", help="Output directory")
    args = parser.parse_args()
    
    generate_maps(Path(args.output_dir), args.count)
