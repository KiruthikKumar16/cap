# SUMO Troubleshooting

## Errors You Saw and Fixes Applied

### 1. **`'Net' object has no attribute 'getJunctions'`**

**Cause:** sumolib (SUMO’s Python library) uses **`getNodes()`**, not `getJunctions()`. The code was calling the wrong method.

**Fix:** `src/phase1/graph_builder.py` now uses `net.getNodes()` and `node.getType()`. Graph building works with your SUMO/sumolib version.

---

### 2. **`An unknown lane (':J0_0_0') was tried to be set as incoming to junction 'J0'`** and **`Unknown from-node 'J0' for edge 'e0'`**

**Cause:** The hand-written `grid_2x2.net.xml` used **`intLanes`** and **`<request>`** on junctions. Those refer to internal lanes that SUMO creates from connections. Our file had no such connections, so SUMO 1.26 rejected the net.

**Fix:** `scripts/create_sumo_network.py` now writes a **minimal net** without `intLanes` and without `<request>`. Junctions only have `id`, `type`, `x`, `y`, and `incLanes`.

**What you should do:** Regenerate the network and run again:

```powershell
python scripts/create_sumo_network.py
python scripts/run_phase1_demo.py --quick
```

---

### 3. **`SUMO_HOME is not set properly`**

**Cause:** SUMO uses the `SUMO_HOME` environment variable for XML validation and tools.

**What to do (optional but recommended):**

1. Find your SUMO install (e.g. `C:\Program Files (x86)\Eclipse SUMO`).
2. In **System Properties → Environment Variables**, add:
   - Variable: `SUMO_HOME`
   - Value: that path (e.g. `C:\Program Files (x86)\Eclipse SUMO`).
3. Restart PowerShell/terminal.

If you only added SUMO’s `bin` to `PATH` and not `SUMO_HOME`, the project still runs; you may see this warning and validation can be limited.

---

### 4. **`Connection 'default' is already active`**

**Cause:** When SUMO failed to start (e.g. due to the bad net), TraCI did not close the “default” connection. The next env that tried to start SUMO saw that name as still in use.

**Fix:** In `src/phase1/traffic_env.py`, when `traci.start()` fails we now call `traci.close()` so the connection is released.

With the fixed net and proper env `close()` calls, this should stop once the net and startup succeed.

---

### 5. **`Vehicle 'flow0.0' has no valid route`**

**Cause:** The net had no **connection** elements, so SUMO couldn’t route vehicles from one edge to the next (e.g. e0 → e4 via J1). Flows with `from="e0" to="e4"` need a path; without connections there is no path.

**Fix:** The net now includes **&lt;connection&gt;** elements for all allowed movements at each junction, and the route file uses **explicit &lt;route&gt;** (e.g. `edges="e0 e4"`) so each flow has a defined path. Regenerate with `python scripts/create_sumo_network.py` or use the updated `data/raw/grid_2x2.net.xml` and `grid_2x2.rou.xml`.

---

### 6. **`Found invalid logic position of a link for junction 'J0' (0, max -1)`**

**Cause:** Hand-written traffic-light nets often get **link indices** wrong. SUMO assigns controlled-link order from junction geometry; manual `tlLogic` and `linkIndex` can mismatch, so SUMO reports invalid logic (e.g. "max -1" = no valid controlled links).

**Fix:** The project now uses **netgenerate + netconvert** to build the 2×2 grid so SUMO creates correct TLS and link logic. Junctions are **A0, A1, B0, B1**; edges are **A0A1, A0B0, A1A0, A1B1, B0A0, B0B1, B1A1, B1B0**. Run `python scripts/create_sumo_network.py` (with SUMO on PATH or `SUMO_HOME` set). The script runs `netgenerate` and `netconvert --tls.set A0,A1,B0,B1` to produce `grid_2x2.net.xml` and writes `grid_2x2.rou.xml` with the correct edge IDs. Then test with: `sumo -n data/raw/grid_2x2.net.xml -r data/raw/grid_2x2.rou.xml`.

---

## Quick checklist

| Step | Action |
|------|--------|
| 1 | Install SUMO and add its **bin** to `PATH` (and optionally set `SUMO_HOME`). |
| 2 | Restart PowerShell after changing PATH/SUMO_HOME. |
| 3 | Regenerate the net: `python scripts/create_sumo_network.py`. |
| 4 | Run the demo: `python scripts/run_phase1_demo.py --quick`. |

If SUMO still does not start, run:

```powershell
sumo -n data/raw/grid_2x2.net.xml -r data/raw/grid_2x2.rou.xml
```

If that works, the net and routes are valid; the issue is then in the Python/TraCI setup.
