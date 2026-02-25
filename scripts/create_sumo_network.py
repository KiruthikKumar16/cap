"""
Create SUMO 2x2 grid network for Phase 1.

Uses netgenerate + netconvert (SUMO tools) to produce a valid .net.xml with
traffic lights. If SUMO is not on PATH, only the routes and config are written;
use an existing grid_2x2.net.xml or run netgenerate/netconvert manually.

Generated net: junctions A0, A1, B0, B1; edges A0A1, A0B0, A1A0, A1B1, B0A0, B0B1, B1A1, B1B0.
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


def create_net_with_sumo(data_dir: Path, net_file: Path) -> bool:
    """Generate grid_2x2.net.xml via netgenerate and netconvert. Return True on success."""
    bin_dir = find_sumo_bin()
    if not bin_dir:
        return False
    netconvert = os.path.join(bin_dir, "netconvert.exe" if os.name == "nt" else "netconvert")
    netgenerate = os.path.join(bin_dir, "netgenerate.exe" if os.name == "nt" else "netgenerate")
    if not os.path.isfile(netconvert) or not os.path.isfile(netgenerate):
        return False
    gen_net = data_dir / "grid_2x2_gen.net.xml"
    try:
        subprocess.run(
            [netgenerate, "--grid", "--grid.number=2", "--grid.length=100", "-o", str(gen_net)],
            cwd=str(data_dir),
            capture_output=True,
            check=True,
            timeout=30,
        )
        subprocess.run(
            [netconvert, "-s", str(gen_net), "-o", str(net_file), "--tls.set", "A0,A1,B0,B1"],
            cwd=str(data_dir),
            capture_output=True,
            check=True,
            timeout=30,
        )
        if gen_net.exists():
            gen_net.unlink()
        # Netconvert creates 1 phase per TLS; RL env expects 4. Patch net to add 4 phases.
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


def create_route_file(output_path: str):
    """Routes for netgenerate 2x2 grid: edge IDs A0A1, A0B0, A1A0, A1B1, B0A0, B0B1, B1A1, B1B0."""
    route_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>
    <route id="r0" edges="A0B0 B0B1"/>
    <route id="r1" edges="A0A1 A1B1"/>
    <route id="r2" edges="B0A0 A0A1"/>
    <route id="r3" edges="B1A1 A1A0"/>
    <flow id="flow0" type="car" route="r0" begin="0" end="3600" vehsPerHour="300" departLane="best" departSpeed="max"/>
    <flow id="flow1" type="car" route="r1" begin="0" end="3600" vehsPerHour="300" departLane="best" departSpeed="max"/>
    <flow id="flow2" type="car" route="r2" begin="0" end="3600" vehsPerHour="300" departLane="best" departSpeed="max"/>
    <flow id="flow3" type="car" route="r3" begin="0" end="3600" vehsPerHour="300" departLane="best" departSpeed="max"/>
</routes>
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(route_content)
    print(f"[OK] Created route file: {output_path}")


def create_config_file(output_path: str, net_file: str, route_file: str):
    """Create SUMO configuration file (.sumocfg)."""
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
    """Create SUMO files for 2x2 grid (net via netgenerate+netconvert if available)."""
    print("Creating SUMO network files for 2x2 grid...")
    print()
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    net_file = data_dir / "grid_2x2.net.xml"
    route_file = data_dir / "grid_2x2.rou.xml"
    config_file = data_dir / "grid_2x2.sumocfg"

    if create_net_with_sumo(data_dir, net_file):
        print(f"[OK] Created network file (netgenerate+netconvert): {net_file}")
    else:
        if net_file.exists():
            print(f"[INFO] Leaving existing network file as-is: {net_file}")
        else:
            print("[WARN] SUMO not found; network file not created.")
            print("       Install SUMO, set SUMO_HOME or add bin to PATH, then run this script again,")
            print("       or copy grid_2x2.net.xml from a machine where netgenerate/netconvert ran.")

    create_route_file(str(route_file))
    create_config_file(str(config_file), "grid_2x2.net.xml", "grid_2x2.rou.xml")

    print()
    print("=" * 60)
    print("SUMO Network Files Ready")
    print("=" * 60)
    print(f"Network file: {net_file}")
    print(f"Route file: {route_file}")
    print(f"Config file: {config_file}")
    print()
    print("Test with:")
    print("  sumo -n data/raw/grid_2x2.net.xml -r data/raw/grid_2x2.rou.xml")


if __name__ == "__main__":
    main()
