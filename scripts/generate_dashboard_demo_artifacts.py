"""
Generate clearly labeled dashboard demo artifacts.

This is for time-boxed UI walkthroughs only. It intentionally marks the output as
presentation_demo so the dashboard displays a warning and the JSON cannot be
mistaken for benchmark evidence.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MEDIA_DIR = OUTPUTS / "dashboard_media"


def _backup(path: Path) -> None:
    if not path.exists():
        return
    backup_dir = OUTPUTS / "dashboard_demo_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, backup_dir / f"{path.stem}_{stamp}{path.suffix}")


def _episode_block(mean_reward: float, mean_throughput: float, mean_travel_time: float, mean_waiting: float, mean_queue: float) -> dict:
    rewards = [mean_reward + d for d in [-4.2, 2.6, 1.1, -1.7, 3.4, -0.9]]
    throughputs = [mean_throughput + d for d in [-18, 9, 14, -7, 21, -3]]
    travel = [mean_travel_time + d for d in [1.4, -0.8, -1.1, 0.6, -1.5, 0.2]]
    waiting = [mean_waiting + d for d in [360, -210, -280, 140, -420, 90]]
    queue = [mean_queue + d for d in [3.8, -1.9, -2.4, 1.1, -3.2, 0.8]]
    return {
        "mean_reward": mean_reward,
        "std_reward": 2.6,
        "mean_throughput": mean_throughput,
        "std_throughput": 14.2,
        "mean_travel_time": mean_travel_time,
        "std_travel_time": 1.1,
        "mean_waiting_time": mean_waiting,
        "std_waiting_time": 285.0,
        "mean_queue_length": mean_queue,
        "std_queue_length": 2.4,
        "rewards": rewards,
        "throughputs": throughputs,
        "travel_times": travel,
        "waiting_times": waiting,
        "queue_lengths": queue,
    }


def _traffic_frame(model_name: str, frame_idx: int, frames: int, accent: tuple[int, int, int], offset: int, style: str) -> Image.Image:
    width, height = 520, 330
    bg = {"mappo": "#eff6ff", "colight": "#ecfdf5", "nstlight": "#fffbeb"}[style]
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    header = {"mappo": "#1e3a8a", "colight": "#065f46", "nstlight": "#92400e"}[style]
    draw.rectangle((0, 0, width, 44), fill=header)
    draw.text((18, 15), model_name, fill="#ffffff", font=font)
    subtitle = {"mappo": "adaptive coordination", "colight": "green-wave baseline", "nstlight": "spatial attention"}[style]
    draw.text((width - 170, 15), subtitle, fill="#e2e8f0", font=font)

    road = "#334155"
    lane = "#cbd5e1"
    if style == "mappo":
        xs = [125, 260, 395]
        ys = [98, 176, 254]
    elif style == "colight":
        xs = [95, 210, 325, 440]
        ys = [118, 222]
    else:
        xs = [150, 260, 370]
        ys = [92, 164, 236, 292]
    for x in xs:
        draw.rounded_rectangle((x - 25, 48, x + 25, height), radius=4, fill=road)
        draw.line((x, 54, x, height - 10), fill=lane, width=1)
    for y in ys:
        draw.rounded_rectangle((0, y - 22, width, y + 22), radius=4, fill=road)
        draw.line((8, y, width - 8, y), fill=lane, width=1)
    for x in xs:
        for y in ys:
            draw.rectangle((x - 29, y - 26, x + 29, y + 26), fill="#1e293b")
            if style == "colight":
                ns_green = ((frame_idx // 5) + (x // 100)) % 2 == 0
            elif style == "nstlight":
                ns_green = ((frame_idx + y // 15 + x // 30) % 24) < 8
            else:
                ns_green = ((frame_idx + offset + (x // 40) + (y // 40)) % 18) < 9
            draw.ellipse((x - 38, y - 35, x - 30, y - 27), fill="#22c55e" if ns_green else "#ef4444")
            draw.ellipse((x + 30, y + 27, x + 38, y + 35), fill="#ef4444" if ns_green else "#22c55e")

    car_colors = [accent, (245, 158, 11), (14, 165, 233), (239, 68, 68), (34, 197, 94)]
    for lane_idx, y in enumerate(ys):
        car_count = {"mappo": 5, "colight": 7, "nstlight": 4}[style]
        for car_idx in range(car_count):
            direction = -1 if style == "nstlight" and lane_idx % 2 else 1
            speed = {"mappo": 7 + lane_idx, "colight": 11 + lane_idx, "nstlight": 5 + lane_idx}[style]
            x = (direction * frame_idx * speed + car_idx * (78 if style == "colight" else 104) + offset * 13) % (width + 70) - 35
            color = car_colors[(lane_idx + car_idx) % len(car_colors)]
            draw.rounded_rectangle((x, y - 14, x + 24, y - 5), radius=3, fill=color)
            draw.rectangle((x + 16, y - 12, x + 22, y - 7), fill="#e0f2fe")
    for lane_idx, x in enumerate(xs):
        car_count = {"mappo": 4, "colight": 3, "nstlight": 5}[style]
        for car_idx in range(car_count):
            speed = {"mappo": 5 + lane_idx, "colight": 4 + lane_idx, "nstlight": 8 + lane_idx}[style]
            y = ((frame_idx * speed + car_idx * 82 + offset * 17) % (height + 70)) + 42
            if y > height + 25:
                y -= height + 70
            color = car_colors[(lane_idx + car_idx + 2) % len(car_colors)]
            draw.rounded_rectangle((x + 7, y, x + 18, y + 24), radius=3, fill=color)
            draw.rectangle((x + 9, y + 2, x + 16, y + 8), fill="#e0f2fe")

    if style == "mappo":
        draw.rounded_rectangle((350, 55, 505, 78), radius=6, fill="#dbeafe", outline="#2563eb")
        draw.text((360, 63), "lowest queue", fill="#1e3a8a", font=font)
    elif style == "colight":
        draw.line((40, 64, 480, 64), fill="#10b981", width=5)
        draw.polygon([(480, 64), (466, 56), (466, 72)], fill="#10b981")
        draw.text((56, 72), "corridor wave", fill="#065f46", font=font)
    else:
        for r in [32, 54, 76]:
            draw.ellipse((260 - r, 176 - r, 260 + r, 176 + r), outline="#f59e0b", width=2)
        draw.text((360, 62), "attention zones", fill="#92400e", font=font)

    progress = int((frame_idx + 1) / frames * 220)
    draw.rounded_rectangle((18, height - 22, 238, height - 12), radius=5, fill="#e2e8f0")
    draw.rounded_rectangle((18, height - 22, 18 + progress, height - 12), radius=5, fill=accent)
    return img


def _make_demo_gif(model_name: str, slug: str, accent: tuple[int, int, int], offset: int, style: str) -> dict:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    frames = 36
    images = [_traffic_frame(model_name, idx, frames, accent, offset, style) for idx in range(frames)]
    gif_path = MEDIA_DIR / f"{slug}_{style}_demo_v2.gif"
    poster_path = MEDIA_DIR / f"{slug}_{style}_demo_v2.png"
    images[0].save(poster_path)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=False)
    return {
        "gif_path": str(gif_path.relative_to(ROOT)),
        "poster_path": str(poster_path.relative_to(ROOT)),
        "frame_count": frames,
        "source": "presentation_demo_animation",
    }


def _parse_sumo_network() -> tuple[list[dict], list[dict]]:
    net_path = ROOT / "data" / "raw" / "grid_5x5.net.xml"
    tree = ET.parse(net_path)
    root = tree.getroot()
    lanes: list[dict] = []
    junctions: list[dict] = []

    def parse_shape(shape: str) -> list[tuple[float, float]]:
        pts = []
        for token in shape.split():
            x, y = token.split(",")
            pts.append((float(x), float(y)))
        return pts

    for edge in root.findall("edge"):
        edge_id = edge.attrib.get("id", "")
        if edge_id.startswith(":"):
            continue
        for lane in edge.findall("lane"):
            pts = parse_shape(lane.attrib["shape"])
            if len(pts) < 2:
                continue
            x0, y0 = pts[0]
            x1, y1 = pts[-1]
            lanes.append(
                {
                    "id": lane.attrib.get("id", ""),
                    "edge": edge_id,
                    "points": pts,
                    "orientation": "horizontal" if abs(x1 - x0) >= abs(y1 - y0) else "vertical",
                    "mid_x": sum(p[0] for p in pts) / len(pts),
                    "mid_y": sum(p[1] for p in pts) / len(pts),
                    "dir": (x1 - x0, y1 - y0),
                }
            )

    for junction in root.findall("junction"):
        jid = junction.attrib.get("id", "")
        if len(jid) == 2 and jid[0].isalpha() and jid[1].isdigit():
            junctions.append(
                {
                    "id": jid,
                    "x": float(junction.attrib["x"]),
                    "y": float(junction.attrib["y"]),
                    "type": junction.attrib.get("type", ""),
                }
            )
    return lanes, junctions


def _sumo_project(point: tuple[float, float], width: int, height: int) -> tuple[float, float]:
    x, y = point
    margin = 34
    sx = margin + (x / 400.0) * (width - 2 * margin)
    sy = height - margin - (y / 400.0) * (height - 2 * margin)
    return sx, sy


def _point_on_polyline(points: list[tuple[float, float]], t: float) -> tuple[float, float]:
    if len(points) == 1:
        return points[0]
    segments = []
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        length = math.dist(a, b)
        segments.append((a, b, length))
        total += length
    target = (t % 1.0) * total
    for a, b, length in segments:
        if target <= length or length == 0:
            frac = 0.0 if length == 0 else target / length
            return (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)
        target -= length
    return points[-1]


def _sumo_policy_frame(
    model_name: str,
    frame_idx: int,
    frames: int,
    accent: tuple[int, int, int],
    offset: int,
    style: str,
    lanes: list[dict],
    junctions: list[dict],
) -> Image.Image:
    width, height = 620, 430
    bg = {"mappo": "#eef6ff", "colight": "#ecfdf5", "nstlight": "#fffbeb"}[style]
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    header = {"mappo": "#1d4ed8", "colight": "#047857", "nstlight": "#b45309"}[style]
    subtitles = {
        "mappo": "coordinated pressure relief",
        "colight": "corridor progression",
        "nstlight": "local spatial attention",
    }
    draw.rectangle((0, 0, width, 42), fill=header)
    draw.text((16, 14), model_name, fill="#ffffff", font=font)
    draw.text((width - 190, 14), subtitles[style], fill="#e2e8f0", font=font)

    for lane in lanes:
        pts = [_sumo_project(p, width, height) for p in lane["points"]]
        color = "#475569" if lane["orientation"] == "horizontal" else "#334155"
        draw.line(pts, fill=color, width=5)
        draw.line(pts, fill="#cbd5e1", width=1)

    for j in junctions:
        x, y = _sumo_project((j["x"], j["y"]), width, height)
        if j["type"] == "traffic_light":
            phase_seed = (ord(j["id"][0]) + int(j["id"][1]) + offset)
            if style == "colight":
                green = ((frame_idx // 5) + ord(j["id"][0])) % 2 == 0
            elif style == "nstlight":
                green = ((frame_idx + phase_seed * 3) % 24) < 10
            else:
                green = ((frame_idx + phase_seed) % 18) < 12
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#22c55e" if green else "#ef4444")
        else:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#94a3b8")

    if style == "mappo":
        selected = [lane for lane in lanes if 85 <= lane["mid_x"] <= 315 or 85 <= lane["mid_y"] <= 315]
        density, speed, color_cycle = 76, 0.022, [(37, 99, 235), (14, 165, 233), (34, 197, 94)]
    elif style == "colight":
        selected = [
            lane
            for lane in lanes
            if lane["orientation"] == "horizontal"
            and any(abs(lane["mid_y"] - band) <= 12 for band in [100.0, 200.0, 300.0])
        ]
        density, speed, color_cycle = 70, 0.034, [(16, 185, 129), (5, 150, 105), (245, 158, 11)]
        for y_raw in [100, 200, 300]:
            y = _sumo_project((0, y_raw), width, height)[1]
            draw.line((38, y - 14, width - 38, y - 14), fill="#10b981", width=3)
            draw.polygon([(width - 38, y - 14), (width - 50, y - 20), (width - 50, y - 8)], fill="#10b981")
    else:
        selected = [lane for lane in lanes if abs(lane["mid_x"] - 200) <= 160 and abs(lane["mid_y"] - 200) <= 160]
        density, speed, color_cycle = 88, 0.018, [(245, 158, 11), (239, 68, 68), (14, 165, 233)]
        cx, cy = _sumo_project((200, 200), width, height)
        for r in [45, 72, 99]:
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="#f59e0b", width=2)

    if not selected:
        selected = lanes

    for i in range(density):
        lane = selected[(i * 7 + offset) % len(selected)]
        base = ((frame_idx * speed) + (i / density) + (offset * 0.013)) % 1.0
        if style == "nstlight" and i % 5 == 0:
            base = (base + 0.08 * math.sin(frame_idx / 4.0 + i)) % 1.0
        if style == "mappo" and i % 6 == 0:
            base = (base + 0.04 * math.sin(frame_idx / 3.0)) % 1.0
        x, y = _sumo_project(_point_on_polyline(lane["points"], base), width, height)
        horizontal = lane["orientation"] == "horizontal"
        car = color_cycle[i % len(color_cycle)]
        if horizontal:
            draw.rounded_rectangle((x - 7, y - 4, x + 9, y + 4), radius=2, fill=car)
            draw.rectangle((x + 2, y - 3, x + 7, y + 3), fill="#dbeafe")
        else:
            draw.rounded_rectangle((x - 4, y - 7, x + 4, y + 9), radius=2, fill=car)
            draw.rectangle((x - 3, y - 2, x + 3, y + 5), fill="#dbeafe")

    progress = int((frame_idx + 1) / frames * 230)
    draw.rounded_rectangle((16, height - 24, 246, height - 13), radius=5, fill="#dbe4ee")
    draw.rounded_rectangle((16, height - 24, 16 + progress, height - 13), radius=5, fill=accent)
    draw.text((260, height - 25), "SUMO grid_5x5 geometry", fill="#475569", font=font)
    return img


def _make_demo_gif(model_name: str, slug: str, accent: tuple[int, int, int], offset: int, style: str) -> dict:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    lanes, junctions = _parse_sumo_network()
    frames = 48
    images = [
        _sumo_policy_frame(model_name, idx, frames, accent, offset, style, lanes, junctions)
        for idx in range(frames)
    ]
    gif_path = MEDIA_DIR / f"{slug}_{style}_sumo_policy_v3.gif"
    poster_path = MEDIA_DIR / f"{slug}_{style}_sumo_policy_v3.png"
    images[0].save(poster_path)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=95, loop=0, optimize=False)
    return {
        "gif_path": str(gif_path.relative_to(ROOT)),
        "poster_path": str(poster_path.relative_to(ROOT)),
        "frame_count": frames,
        "source": "presentation_demo_sumo_geometry_animation",
    }


def main() -> None:
    now = datetime.now().isoformat(timespec="seconds")
    media = {
        "MAPPO-STGNN": _make_demo_gif("MAPPO-STGNN", "mappo_stgnn", (37, 99, 235), 2, "mappo"),
        "CoLight": _make_demo_gif("CoLight", "colight", (16, 185, 129), 5, "colight"),
        "NSTLight": _make_demo_gif("NSTLight", "nstlight", (245, 158, 11), 8, "nstlight"),
    }

    benchmark = {
        "artifact_metadata": {
            "artifact_type": "presentation_demo",
            "generated_at": now,
            "disclaimer": "Synthetic/sample dashboard output for UI walkthrough only; not benchmark evidence.",
        },
        "MAPPO-STGNN": {
            "mean_reward": -512.4,
            "mean_throughput": 2328.0,
            "mean_travel_time": 43.6,
            "mean_waiting_time": 38240.0,
            "mean_queue_length": 251.7,
        },
        "CoLight": {
            "mean_reward": -548.9,
            "mean_throughput": 2215.0,
            "mean_travel_time": 47.8,
            "mean_waiting_time": 41980.0,
            "mean_queue_length": 286.4,
            "diagnostics": {"model_type": "CoLight", "weights_loaded": False, "mode": "demo"},
        },
        "NSTLight": {
            "mean_reward": -536.2,
            "mean_throughput": 2242.0,
            "mean_travel_time": 46.2,
            "mean_waiting_time": 40790.0,
            "mean_queue_length": 274.9,
            "diagnostics": {"model_type": "NSTLight", "weights_loaded": False, "mode": "demo"},
        },
        "FixedTime": {
            "mean_reward": -589.6,
            "mean_throughput": 2096.0,
            "mean_travel_time": 52.4,
            "mean_waiting_time": 46250.0,
            "mean_queue_length": 323.5,
        },
        "Random": {
            "mean_reward": -641.8,
            "mean_throughput": 1984.0,
            "mean_travel_time": 58.9,
            "mean_waiting_time": 51870.0,
            "mean_queue_length": 371.2,
        },
        "latency_ms_per_step": [
            {"model": "MAPPO-STGNN", "device": "CPU", "n_runs": 200, "mean_ms": 0.152, "std_ms": 0.040, "p95_ms": 0.210},
            {"model": "CoLight", "device": "CPU", "n_runs": 200, "mean_ms": 0.141, "std_ms": 0.031, "p95_ms": 0.201},
            {"model": "NSTLight", "device": "CPU", "n_runs": 200, "mean_ms": 0.149, "std_ms": 0.028, "p95_ms": 0.214},
            {"model": "FixedTime", "device": "CPU", "n_runs": 200, "mean_ms": 0.013, "std_ms": 0.001, "p95_ms": 0.013},
        ],
        "action_diagnostics": {
            "MAPPO-STGNN": {"trace_steps": 32, "unique_action_vectors": 11, "dominant_vector_fraction": 0.28, "vector_change_rate": 0.74, "mean_unique_phases_per_step": 2.9, "weights_loaded": True},
            "CoLight": {"trace_steps": 32, "unique_action_vectors": 8, "dominant_vector_fraction": 0.34, "vector_change_rate": 0.61, "mean_unique_phases_per_step": 2.4, "weights_loaded": False},
            "NSTLight": {"trace_steps": 32, "unique_action_vectors": 9, "dominant_vector_fraction": 0.31, "vector_change_rate": 0.66, "mean_unique_phases_per_step": 2.6, "weights_loaded": False},
        },
        "dashboard_media": media,
    }

    eval_summary = {
        "artifact_type": "presentation_demo",
        "num_episodes": 6,
        "num_seeds": 1,
        "total_runs": 6,
        "used_sumo": False,
        "dqn": _episode_block(-512.4, 2328.0, 43.6, 38240.0, 251.7),
        "fixed_time": _episode_block(-589.6, 2096.0, 52.4, 46250.0, 323.5),
        "random": _episode_block(-641.8, 1984.0, 58.9, 51870.0, 371.2),
    }

    stress = {
        "artifact_type": "presentation_demo",
        "normal": {
            "mappo": benchmark["MAPPO-STGNN"],
            "nstlight": benchmark["NSTLight"],
        },
        "stress": {
            "mappo": {"mean_reward": -548.0, "mean_throughput": 2198.0, "mean_travel_time": 47.1, "mean_waiting_time": 42120.0, "mean_queue_length": 280.4},
            "nstlight": {"mean_reward": -591.5, "mean_throughput": 2104.0, "mean_travel_time": 51.3, "mean_waiting_time": 46540.0, "mean_queue_length": 318.7},
        },
        "degradation_limits_pct": {
            "mappo": {"throughput_drop_pct": 5.58, "waiting_time_increase_pct": 10.15, "queue_length_increase_pct": 11.40},
            "nstlight": {"throughput_drop_pct": 6.15, "waiting_time_increase_pct": 14.10, "queue_length_increase_pct": 15.93},
        },
    }

    targets = [
        OUTPUTS / "benchmark_results.json",
        OUTPUTS / "dashboard_media.json",
        OUTPUTS / "phase1" / "evaluation_summary.json",
        OUTPUTS / "phase3" / "adversarial_benchmark.json",
    ]
    for target in targets:
        _backup(target)

    (OUTPUTS / "phase1").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "phase3").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "benchmark_results.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    (OUTPUTS / "dashboard_media.json").write_text(json.dumps(media, indent=2), encoding="utf-8")
    (OUTPUTS / "phase1" / "evaluation_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    (OUTPUTS / "phase3" / "adversarial_benchmark.json").write_text(json.dumps(stress, indent=2), encoding="utf-8")

    print("Dashboard demo artifacts generated.")
    print("Open the dashboard and click 'Load Latest Results'.")
    print("These artifacts are labeled presentation_demo and are not benchmark evidence.")


if __name__ == "__main__":
    main()
