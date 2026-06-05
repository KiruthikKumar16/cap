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

    # ✅ Generate routes with Mixed Traffic (Rickshaws/Bikes)
    # (Patent Angle: Heterogeneous traffic simulation for developing world urban scenarios)
    trip_cmd = [
        "python",
        trip_script,
        "-n", net_file,
        "-r", route_file,
        "-e", "1000",
        "--period", str(1.0 / {"low": 0.1, "medium": 0.5, "high": 1.0}[demand]),
        "--fringe-factor", "10",
        "--validate"
    ]
    
    # Add vehicle classes if requested
    # Standard: passenger, truck, bus, motorcycle, bicycle
    # In SUMO, we can define custom types for Rickshaws by overriding motorcycle parameters
    vtype_args = [
        '--vtype-output', 'data/raw/vtypes.add.xml',
        '--vclass', 'passenger',
        '--vehicle-class', 'passenger'
    ]
    # We'll use a post-processing step to inject heterogeneous traffic types
    subprocess.run(trip_cmd, check=True)
    
    # [NEW] Mitigation: Inject mixed traffic types into the route file
    _inject_mixed_traffic(route_file)

def _inject_mixed_traffic(route_file: str):
    """
    Injects rickshaws and bicycles into the generated route file.
    Simulates Sim-to-Real heterogeneity.
    """
    import random
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(route_file)
    root = tree.getroot()
    
    # Define custom types
    rickshaw_type = ET.Element('vType', {
        'id': 'rickshaw', 'vClass': 'motorcycle', 'length': '3.0', 'width': '1.5', 
        'maxSpeed': '12.0', 'accel': '1.5', 'decel': '3.0', 'sigma': '0.9', 'guiShape': 'three_wheeler'
    })
    bike_type = ET.Element('vType', {
        'id': 'bicycle', 'vClass': 'bicycle', 'length': '1.8', 'width': '0.6', 
        'maxSpeed': '5.0', 'accel': '0.8', 'decel': '2.0', 'sigma': '0.7', 'guiShape': 'bicycle'
    })
    
    root.insert(0, rickshaw_type)
    root.insert(1, bike_type)
    
    # Randomly change vehicle types
    for veh in root.findall('vehicle'):
        rand = random.random()
        if rand < 0.15: # 15% Rickshaws
            veh.set('type', 'rickshaw')
        elif rand < 0.25: # 10% Bicycles
            veh.set('type', 'bicycle')
            
    tree.write(route_file)
    print(f"[OK] Injected mixed traffic into {route_file}")

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