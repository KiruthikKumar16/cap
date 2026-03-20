import os
import subprocess


def create_sumo_scenario(grid_size: int, demand: str):
    """Creates a SUMO scenario with a given grid size and traffic demand."""
    
    net_file = f"data/raw/grid_{grid_size}x{grid_size}.net.xml"
    route_file = f"data/raw/grid_{grid_size}x{grid_size}_{demand}.rou.xml"
    
    # Create network file
    netconvert_cmd = [
        "netconvert",
        "--grid-network", str(grid_size),
        "-o", net_file
    ]
    subprocess.run(netconvert_cmd, check=True)
    
    # Create route file
    trip_cmd = [
        "python", os.environ.get("SUMO_HOME", "") + "/tools/trip/randomTrips.py",
        "-n", net_file,
        "-r", route_file,
        "-e", "1000",
        "--period", str(1.0 / {"low": 0.1, "medium": 0.5, "high": 1.0}[demand])
    ]
    subprocess.run(trip_cmd, check=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create SUMO scenario")
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size")
    parser.add_argument("--demand", type=str, default="medium", choices=["low", "medium", "high"], help="Traffic demand")
    args = parser.parse_args()
    
    create_sumo_scenario(args.grid_size, args.demand)
