import os
import subprocess


def create_sumo_scenario(grid_size: int, demand: str):
    """Creates a SUMO scenario with a given grid size and traffic demand."""

    net_file = f"data/raw/grid_{grid_size}x{grid_size}.net.xml"
    route_file = f"data/raw/grid_{grid_size}x{grid_size}_{demand}.rou.xml"

    # Ensure output directory exists
    os.makedirs("data/raw", exist_ok=True)

    # ✅ Generate network using netgenerate ONLY if it doesn't exist
    if not os.path.exists(net_file):
        print(f"Generating new network: {net_file}")
        netgenerate_cmd = [
            "netgenerate",
            "--grid",
            "--grid.number", str(grid_size),
            "--output-file", net_file,
            "--grid.traffic-lights", "true"  # Ensure we have traffic lights!
        ]
        subprocess.run(netgenerate_cmd, check=True)
    else:
        print(f"Using existing network: {net_file}")

    # ✅ Get randomTrips.py path correctly
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME is not set. Please set it to your SUMO installation path.")

    trip_script = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")

    if not os.path.exists(trip_script):
        raise FileNotFoundError(f"randomTrips.py not found at: {trip_script}")

    # ✅ Generate routes
    trip_cmd = [
        "python",
        trip_script,
        "-n", net_file,
        "-r", route_file,
        "-e", "1000",
        "--period", str(1.0 / {"low": 0.1, "medium": 0.5, "high": 1.0}[demand])
    ]

    subprocess.run(trip_cmd, check=True)

    print("\n[OK] SUMO scenario created successfully!")
    print(f"Network file: {net_file}")
    print(f"Route file:   {route_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create SUMO scenario")
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size")
    parser.add_argument(
        "--demand",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Traffic demand"
    )

    args = parser.parse_args()

    create_sumo_scenario(args.grid_size, args.demand)