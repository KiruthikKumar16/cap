"""
Quick SUMO + TraCI connectivity check.

Run from project root (with venv activated):
  python scripts/check_sumo.py

Checks:
  1. SUMO binary is found (config or SUMO_HOME or PATH)
  2. TraCI Python module is available
  3. A short simulation runs and returns departed/arrived counts
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 60)
    print("SUMO + TraCI connectivity check")
    print("=" * 60)

    # 1) Config
    try:
        import yaml
        with open(project_root / "configs" / "phase1.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        sumo_binary = config.get("sumo", {}).get("sumo_binary")
        net_file = config.get("sumo", {}).get("net_file", "data/raw/grid_2x2.net.xml")
        route_file = config.get("sumo", {}).get("route_file", "data/raw/grid_2x2.rou.xml")
        config_file = config.get("sumo", {}).get("config_file")
    except Exception as e:
        print(f"  [WARN] Could not load config: {e}")
        sumo_binary = None
        net_file = "data/raw/grid_2x2.net.xml"
        route_file = "data/raw/grid_2x2.rou.xml"
        config_file = "data/raw/grid_2x2.sumocfg"

    if sumo_binary:
        print(f"  Config sumo_binary: {sumo_binary}")
        if not Path(sumo_binary).exists():
            print(f"  [FAIL] File not found.")
        else:
            print(f"  [OK] File exists.")
    else:
        print("  Config sumo_binary: not set (will use SUMO_HOME or PATH)")

    sumo_home = os.environ.get("SUMO_HOME", "").strip()
    if sumo_home:
        print(f"  SUMO_HOME: {sumo_home}")
    else:
        print("  SUMO_HOME: not set (optional if sumo_binary is set)")

    # 2) TraCI
    try:
        import traci
        print("  TraCI: import OK")
    except ImportError as e:
        print(f"  [FAIL] TraCI not available: {e}")
        print("  Install: pip install traci (or use SUMO's Python environment)")
        return 1

    # 3) Start SUMO and run a few steps
    try:
        import traci
        sumo_bin = sumo_binary or "sumo"
        if not sumo_binary and sumo_home:
            sumo_bin = os.path.join(sumo_home, "bin", "sumo.exe" if os.name == "nt" else "sumo")
        cmd = [sumo_bin, "--step-length", "1", "--no-warnings"]
        if config_file and (project_root / config_file).exists():
            cmd.extend(["-c", str(project_root / config_file)])
        else:
            cmd.extend(["-n", str(project_root / net_file), "-r", str(project_root / route_file)])
        print(f"  Command: {' '.join(cmd)}")
        traci.start(cmd, port=8813)
        print("  [OK] SUMO started (TraCI connected)")

        departed_total = 0
        arrived_total = 0
        for _ in range(50):
            traci.simulationStep()
            departed_total += traci.simulation.getDepartedNumber()
            arrived_total += traci.simulation.getArrivedNumber()
        print(f"  After 50 steps: departed={departed_total}, arrived={arrived_total}")
        if departed_total == 0:
            print("  [WARN] No vehicles departed — check route file and flows.")
        else:
            print("  [OK] Vehicles are flowing.")
        traci.close()
    except Exception as e:
        print(f"  [FAIL] SUMO run failed: {e}")
        try:
            traci.close()
        except Exception:
            pass
        return 1

    print("=" * 60)
    print("SUMO is working. If evaluation still shows 0 throughput/travel time,")
    print("the connection may be dropping when multiple envs reset (e.g. DQN vs fixed-time).")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
