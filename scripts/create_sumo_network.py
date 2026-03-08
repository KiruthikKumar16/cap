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
from pathlib import Path


def find_sumo_bin():
    """Return path to SUMO bin directory, or None if not found."""
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
    # PATH
    nc = shutil.which("netconvert")
    if nc:
        return str(Path(nc).parent)
    return None


def create_net_generic(data_dir: Path, grid_size: int, net_file: Path) -> bool:
    """Generate grid_NxN.net.xml via netgenerate and netconvert. Return True on success."""
    bin_dir = find_sumo_bin()
    if not bin_dir:
        return False
    netconvert = os.path.join(bin_dir, "netconvert.exe" if os.name == "nt" else "netconvert")
    netgenerate = os.path.join(bin_dir, "netgenerate.exe" if os.name == "nt" else "netgenerate")
    if not os.path.isfile(netconvert) or not os.path.isfile(netgenerate):
        return False
    gen_net = data_dir / f"grid_{grid_size}x{grid_size}_gen.net.xml"
    try:
        subprocess.run(
            [netgenerate, "--grid", f"--grid.number={grid_size}", "--grid.length=100", "-o", str(gen_net)],
            cwd=str(data_dir),
            capture_output=True,
            check=True,
            timeout=30,
        )
        # Create TLS ID list for all intersections
        tls_ids = ",".join([f"{chr(65+i)}{j}" for i in range(grid_size) for j in range(grid_size)])
        subprocess.run(
            [netconvert, "-s", str(gen_net), "-o", str(net_file), "--tls.set", tls_ids],
            cwd=str(data_dir),
            capture_output=True,
            check=True,
            timeout=30,
        )
        if gen_net.exists():
            gen_net.unlink()
        # Patch net to add 4 phases for RL control
        _patch_net_four_phases(net_file)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        if gen_net.exists():
            try:
                gen_net.unlink()
            except OSError:
                pass
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
        edges_rl = list(reversed(edges_lr))
        if edges_lr:
            routes.append(f'    <route id="h_r{row}_lr" edges="{" ".join(edges_lr)}"/>')
            flows.append(f'    <flow id="flow_{flow_id}" type="car" route="h_r{row}_lr" begin="0" end="3600" vehsPerHour="{veh_per_hour}" departLane="best" departSpeed="max"/>')
            flow_id += 1
        if edges_rl:
            routes.append(f'    <route id="h_r{row}_rl" edges="{" ".join(edges_rl)}"/>')
            flows.append(f'    <flow id="flow_{flow_id}" type="car" route="h_r{row}_rl" begin="0" end="3600" vehsPerHour="{veh_per_hour}" departLane="best" departSpeed="max"/>')
            flow_id += 1
    
    # Vertical routes (top to bottom and bottom to top)
    for col in range(grid_size):
        edges_tb = [f"{chr(65+col)}{row}{chr(65+col)}{row+1}" for row in range(grid_size-1)]
        edges_bt = list(reversed(edges_tb))
        if edges_tb:
            routes.append(f'    <route id="v_c{col}_tb" edges="{" ".join(edges_tb)}"/>')
            flows.append(f'    <flow id="flow_{flow_id}" type="car" route="v_c{col}_tb" begin="0" end="3600" vehsPerHour="{veh_per_hour}" departLane="best" departSpeed="max"/>')
            flow_id += 1
        if edges_bt:
            routes.append(f'    <route id="v_c{col}_bt" edges="{" ".join(edges_bt)}"/>')
            flows.append(f'    <flow id="flow_{flow_id}" type="car" route="v_c{col}_bt" begin="0" end="3600" vehsPerHour="{veh_per_hour}" departLane="best" departSpeed="max"/>')
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


def main():
    """Create SUMO files for 3x3 and 6x6 grids (no placeholder 2x2). SUMO mandatory."""
    print("=" * 70)
    print("SUMO Network Generation: 3x3 (Publication) + 6x6 (Scalability)")
    print("=" * 70)
    print("NOTE: SUMO is MANDATORY. Placeholder mode has been removed.")
    print()
    
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 3x3 grid
    print("Creating 3x3 grid network (baseline for research/patent)...")
    grid_size = 3
    net_file_3x3 = data_dir / "grid_3x3.net.xml"
    route_file_3x3 = data_dir / "grid_3x3.rou.xml"
    config_file_3x3 = data_dir / "grid_3x3.sumocfg"
    
    if create_net_generic(data_dir, grid_size, net_file_3x3):
        print(f"[OK] Created 3x3 network file (netgenerate+netconvert): {net_file_3x3}")
    else:
        if net_file_3x3.exists():
            print(f"[INFO] Leaving existing 3x3 network file as-is: {net_file_3x3}")
        else:
            print("[WARN] Could not create 3x3 network with netgenerate/netconvert.")
            print("       Install SUMO or set SUMO_HOME, then run this script again.")
    
    create_route_file_generic(str(route_file_3x3), grid_size)
    create_config_file_generic(str(config_file_3x3), "grid_3x3.net.xml", "grid_3x3.rou.xml", grid_size)
    print()
    
    # Create 6x6 grid
    print("Creating 6x6 grid network (scalability proof for real-world deployment)...")
    grid_size = 6
    net_file_6x6 = data_dir / "grid_6x6.net.xml"
    route_file_6x6 = data_dir / "grid_6x6.rou.xml"
    config_file_6x6 = data_dir / "grid_6x6.sumocfg"
    
    if create_net_generic(data_dir, grid_size, net_file_6x6):
        print(f"[OK] Created 6x6 network file (netgenerate+netconvert): {net_file_6x6}")
    else:
        if net_file_6x6.exists():
            print(f"[INFO] Leaving existing 6x6 network file as-is: {net_file_6x6}")
        else:
            print("[WARN] Could not create 6x6 network with netgenerate/netconvert.")
            print("       Install SUMO or set SUMO_HOME, then run this script again.")
    
    create_route_file_generic(str(route_file_6x6), grid_size)
    create_config_file_generic(str(config_file_6x6), "grid_6x6.net.xml", "grid_6x6.rou.xml", grid_size)
    print()
    
    print("=" * 70)
    print("SUMO Network Files Ready (3x3 + 6x6)")
    print("=" * 70)
    print(f"3x3 Network:  {net_file_3x3}")
    print(f"3x3 Routes:   {route_file_3x3}")
    print(f"3x3 Config:   {config_file_3x3}")
    print()
    print(f"6x6 Network:  {net_file_6x6}")
    print(f"6x6 Routes:   {route_file_6x6}")
    print(f"6x6 Config:   {config_file_6x6}")
    print()
    print("Test with:")
    print("  sumo -n data/raw/grid_3x3.net.xml -r data/raw/grid_3x3.rou.xml")
    print("  sumo -n data/raw/grid_6x6.net.xml -r data/raw/grid_6x6.rou.xml")
    print()
    print("REMOVED: 2x2 grid (use 3x3 minimum for research/patent credibility)")


if __name__ == "__main__":
    main()
