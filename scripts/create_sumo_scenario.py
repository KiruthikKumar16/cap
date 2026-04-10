import os
import subprocess
import sys
from pathlib import Path

# Reuse grid net builder from create_sumo_network (netconvert --grid-network was removed in recent SUMO)
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
import create_sumo_network as csn  # noqa: E402


def create_sumo_scenario(grid_size: int, demand: str):
    """Creates a SUMO scenario with a given grid size and traffic demand."""

    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    net_file = data_dir / f"grid_{grid_size}x{grid_size}.net.xml"
    route_file = data_dir / f"grid_{grid_size}x{grid_size}_{demand}.rou.xml"

    if not csn.create_net_generic(data_dir, grid_size, net_file):
        raise RuntimeError(
            f"Could not build {net_file}. Install SUMO, set SUMO_HOME, and ensure netconvert is available."
        )
    csn._patch_net_four_phases(net_file)

    sumo_home = os.environ.get("SUMO_HOME", "").strip()
    if not sumo_home:
        raise RuntimeError("SUMO_HOME is not set; cannot find tools/randomTrips.py")
    random_trips = Path(sumo_home) / "tools" / "randomTrips.py"
    if not random_trips.is_file():
        raise RuntimeError(f"randomTrips.py not found at {random_trips}")

    period = str(1.0 / {"low": 0.1, "medium": 0.5, "high": 1.0}[demand])
    trip_cmd = [
        sys.executable,
        str(random_trips),
        "-n",
        str(net_file),
        "-r",
        str(route_file),
        "-e",
        "1000",
        "--period",
        period,
    ]
    subprocess.run(trip_cmd, check=True)
    print(f"[OK] Scenario routes: {route_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create SUMO scenario")
    parser.add_argument("--grid-size", type=int, default=5, help="Grid size")
    parser.add_argument(
        "--demand",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Traffic demand",
    )
    args = parser.parse_args()

    create_sumo_scenario(args.grid_size, args.demand)
