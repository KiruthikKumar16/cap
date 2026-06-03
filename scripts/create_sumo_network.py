"""
Create SUMO grid networks (3x3 and 6x6) for Phase 1.

Uses netgenerate + netconvert (SUMO tools) to produce valid .net.xml files with
traffic lights. If SUMO is not on PATH, only the routes and config are written;
use existing .net.xml files or run netgenerate/netconvert manually.

Removed: 2x2 grid (use 3x3 minimum for research/patent credibility)
Active: 3x3 (baseline for publication) and 6x6 (scalability proof)
"""

import os
import shutil
import subprocess
import argparse
from pathlib import Path


def find_sumo_bin():
    """Return path to SUMO bin directory, or None if not found."""
    # First try PATH
    nc = shutil.which("netconvert")
    if nc:
        return str(Path(nc).parent)
    
    # Then try SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME", "").strip()
    if sumo_home:
        bin_path = Path(sumo_home) / "bin"
        if (bin_path / "netconvert.exe").exists() or (bin_path / "netconvert").exists():
            return str(bin_path)
    
    # Common Windows install
    for prefix in [r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Program Files\Eclipse\Sumo"]:
        bin_path = Path(prefix) / "bin"
        if (bin_path / "netconvert.exe").exists():
            return str(bin_path)
    
    return None


def create_net_generic(data_dir: Path, grid_size: int, net_file: Path) -> bool:
    """Generate grid_NxN.net.xml by manually creating nod and edg files."""
    bin_dir = find_sumo_bin()
    if not bin_dir:
        return False
    netconvert = os.path.join(bin_dir, "netconvert.exe" if os.name == "nt" else "netconvert")
    
    nod_file = data_dir / f"grid_{grid_size}x{grid_size}.nod.xml"
    edg_file = data_dir / f"grid_{grid_size}x{grid_size}.edg.xml"
    
    # Generate nodes
    nodes = ['<nodes>']
    for row in range(grid_size):
        for col in range(grid_size):
            node_id = f"{chr(65+col)}{row}"
            x = col * 100
            y = row * 100
            # Internal nodes are traffic lights, boundary nodes are priority
            node_type = "traffic_light" if 0 < row < grid_size-1 and 0 < col < grid_size-1 else "priority"
            nodes.append(f'    <node id="{node_id}" x="{x}" y="{y}" type="{node_type}"/>')
    nodes.append('</nodes>')
    nod_file.write_text('\n'.join(nodes), encoding="utf-8")
    
    # Generate edges
    edges = ['<edges>']
    for row in range(grid_size):
        for col in range(grid_size):
            curr = f"{chr(65+col)}{row}"
            # Horizontal
            if col < grid_size - 1:
                nxt = f"{chr(65+col+1)}{row}"
                edges.append(f'    <edge id="{curr}{nxt}" from="{curr}" to="{nxt}" priority="1" numLanes="2" speed="13.89"/>')
                edges.append(f'    <edge id="{nxt}{curr}" from="{nxt}" to="{curr}" priority="1" numLanes="2" speed="13.89"/>')
            # Vertical
            if row < grid_size - 1:
                nxt = f"{chr(65+col)}{row+1}"
                edges.append(f'    <edge id="{curr}{nxt}" from="{curr}" to="{nxt}" priority="1" numLanes="2" speed="13.89"/>')
                edges.append(f'    <edge id="{nxt}{curr}" from="{nxt}" to="{curr}" priority="1" numLanes="2" speed="13.89"/>')
    edges.append('</edges>')
    edg_file.write_text('\n'.join(edges), encoding="utf-8")
    
    try:
        subprocess.run(
            [netconvert, "-n", str(nod_file), "-e", str(edg_file), "-o", str(net_file)],
            check=True,
            capture_output=True,
        )
        # Cleanup
        nod_file.unlink()
        edg_file.unlink()
        return True
    except Exception as e:
        print(f"Error during network generation: {e}")
        return False


def _patch_net_four_phases(net_file: Path) -> None:
    """Replace single-phase tlLogic with 4 phases (GG, yy, rr, rr) so RL setPhase(0..3) is valid."""
    text = net_file.read_text(encoding="utf-8")
    one_phase = '        <phase duration="90" state="GG"/>'
    four_phases = """        <phase duration="31" state="GG"/>
        <phase duration="5" state="yy"/>
        <phase duration="31" state="rr"/>
        <phase duration="5" state="rr"/>"""
    if one_phase in text:
        text = text.replace(one_phase, four_phases)
        net_file.write_text(text, encoding="utf-8")


def create_route_file_generic(output_path: str, grid_size: int):
    """Create routes for NxN grid."""
    # Create routes that traverse the grid in multiple directions
    routes = []
    flows = []
    flow_id = 0
    veh_per_hour = max(300, 2000 // (grid_size * grid_size))  # Adjust density by grid size
    
    # Horizontal routes (left to right and right to left)
    for row in range(grid_size):
        edges_lr = [f"{chr(65+col)}{row}{chr(65+col+1)}{row}" for col in range(grid_size-1)]
        edges_rl = [f"{chr(65+col+1)}{row}{chr(65+col)}{row}" for col in range(grid_size-2, -1, -1)]
        if edges_lr:
            routes.append(f'    <route id="h_r{row}_lr" edges="{" ".join(edges_lr)}"/>')
            flows.append(f'    <flow id="flow_{flow_id}" type="car" route="h_r{row}_lr" begin="0" end="3600" vehsPerHour="{veh_per_hour}" departLane="optimized" departSpeed="max"/>')
            flow_id += 1
        if edges_rl:
            routes.append(f'    <route id="h_r{row}_rl" edges="{" ".join(edges_rl)}"/>')
            flows.append(f'    <flow id="flow_{flow_id}" type="car" route="h_r{row}_rl" begin="0" end="3600" vehsPerHour="{veh_per_hour}" departLane="optimized" departSpeed="max"/>')
            flow_id += 1
    
    # Vertical routes (top to bottom and bottom to top)
    for col in range(grid_size):
        edges_tb = [f"{chr(65+col)}{row}{chr(65+col)}{row+1}" for row in range(grid_size-1)]
        edges_bt = [f"{chr(65+col)}{row+1}{chr(65+col)}{row}" for row in range(grid_size-2, -1, -1)]
        if edges_tb:
            routes.append(f'    <route id="v_c{col}_tb" edges="{" ".join(edges_tb)}"/>')
            flows.append(f'    <flow id="flow_{flow_id}" type="car" route="v_c{col}_tb" begin="0" end="3600" vehsPerHour="{veh_per_hour}" departLane="optimized" departSpeed="max"/>')
            flow_id += 1
        if edges_bt:
            routes.append(f'    <route id="v_c{col}_bt" edges="{" ".join(edges_bt)}"/>')
            flows.append(f'    <flow id="flow_{flow_id}" type="car" route="v_c{col}_bt" begin="0" end="3600" vehsPerHour="{veh_per_hour}" departLane="optimized" departSpeed="max"/>')
            flow_id += 1
    
    route_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>
{chr(10).join(routes)}
{chr(10).join(flows)}
</routes>
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(route_content)
    print(f"[OK] Created route file: {output_path}")


def create_config_file_generic(output_path: str, net_file: str, route_file: str, grid_size: int):
    """Create SUMO configuration file (.sumocfg) for NxN grid."""
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
    <processing>
        <lateral-resolution value="0.8"/>
    </processing>
    <report>
        <no-warnings value="true"/>
    </report>
</configuration>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    print(f"[OK] Created config file: {output_path}")


def create_grid_network(grid_size: int, data_dir: Path):
    """Creates all SUMO files for a grid of a given size."""
    print(f"Creating {grid_size}x{grid_size} grid network...")

    net_file = data_dir / f"grid_{grid_size}x{grid_size}.net.xml"
    route_file = data_dir / f"grid_{grid_size}x{grid_size}.rou.xml"
    config_file = data_dir / f"grid_{grid_size}x{grid_size}.sumocfg"

    if create_net_generic(data_dir, grid_size, net_file):
        print(f"[OK] Created {grid_size}x{grid_size} network file: {net_file}")
    else:
        if net_file.exists():
            print(f"[INFO] Leaving existing {grid_size}x{grid_size} network file as-is: {net_file}")
        else:
            print(f"[WARN] Could not create {grid_size}x{grid_size} network with netgenerate/netconvert.")
            print("       Install SUMO or set SUMO_HOME, then run this script again.")
            return  # Can't proceed without a net file

    create_route_file_generic(str(route_file), grid_size)
    create_config_file_generic(
        str(config_file), f"grid_{grid_size}x{grid_size}.net.xml", f"grid_{grid_size}x{grid_size}.rou.xml", grid_size
    )
    print()

    print(f"--- {grid_size}x{grid_size} Files Ready ---")
    print(f"Network:  {net_file}")
    print(f"Routes:   {route_file}")
    print(f"Config:   {config_file}")
    print()
    print("Test with:")
    print(f"  sumo-gui -c {config_file.relative_to(data_dir.parent.parent)}")
    print()


def main():
    """Create SUMO files for a configurable grid size."""
    import argparse

    parser = argparse.ArgumentParser(description="Create SUMO grid network files.")
    parser.add_argument(
        "--grid-size", type=int, default=10, help="Size of the grid (e.g., 10 for a 10x10 grid)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"SUMO Network Generation: {args.grid_size}x{args.grid_size}")
    print("=" * 70)
    print("NOTE: SUMO is MANDATORY. This script uses netgenerate/netconvert.")
    print()

    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    create_grid_network(args.grid_size, data_dir)

    print("=" * 70)
    print("SUMO Network Files Generation Complete")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SUMO network files.")
    parser.add_argument("--grid-size", type=int, default=10, help="Size of the grid (e.g., 10 for a 10x10 grid)")
    parser.add_argument("--veh-per-hour", type=int, default=300, help="Vehicles per hour per flow")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory for network files")
    args = parser.parse_args()
    
    create_grid_network(args.grid_size, Path(args.output_dir))
