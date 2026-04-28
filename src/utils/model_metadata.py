import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _config_digest(config: Dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_metadata(
    algorithm: str,
    checkpoint_path: str,
    config: Dict[str, Any],
    observation_space_repr: str,
    action_space_repr: str,
) -> Dict[str, Any]:
    return {
        "algorithm": algorithm,
        "checkpoint_path": checkpoint_path,
        "observation_space": observation_space_repr,
        "action_space": action_space_repr,
        "config_digest": _config_digest(config),
        "sumo": config.get("sumo", {}),
        "model": config.get("model", {}),
        "reward": config.get("reward", {}),
        "evaluation": {"seeds": config.get("evaluation", {}).get("seeds", [])},
    }


def _metadata_candidates(checkpoint_path: Path) -> list:
    base = checkpoint_path
    if checkpoint_path.suffix != ".zip":
        base = checkpoint_path.with_suffix(".zip")
    return [
        checkpoint_path.with_suffix(".metadata.json"),
        Path(str(checkpoint_path) + ".metadata.json"),
        base.with_suffix(".metadata.json"),
        Path(str(base) + ".metadata.json"),
    ]


def save_metadata(metadata: Dict[str, Any], checkpoint_path: Path) -> Path:
    target = checkpoint_path.with_suffix(".metadata.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def load_metadata_for_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    for candidate in _metadata_candidates(checkpoint_path):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def validate_metadata(
    metadata: Dict[str, Any],
    expected_algorithm: str,
    observation_space_repr: str,
    action_space_repr: str,
    config: Dict[str, Any],
) -> Optional[str]:
    if not metadata:
        return None
    model_algo = str(metadata.get("algorithm", "")).upper()
    if model_algo and model_algo != expected_algorithm.upper():
        return f"Checkpoint algorithm `{model_algo}` is incompatible with expected `{expected_algorithm}`."
    if metadata.get("observation_space") and metadata["observation_space"] != observation_space_repr:
        return (
            "Observation space mismatch detected via metadata. "
            f"checkpoint={metadata.get('observation_space')} env={observation_space_repr}"
        )
    if metadata.get("action_space") and metadata["action_space"] != action_space_repr:
        return (
            "Action space mismatch detected via metadata. "
            f"checkpoint={metadata.get('action_space')} env={action_space_repr}"
        )
    expected_digest = _config_digest(config)
    saved_digest = metadata.get("config_digest")
    if saved_digest and saved_digest != expected_digest:
        return (
            "Config digest mismatch detected via metadata. "
            "Checkpoint was produced with a different configuration snapshot."
        )
    return None


def is_digest_only_mismatch(
    mismatch: Optional[str],
    metadata: Optional[Dict[str, Any]],
    expected_algorithm: str,
    observation_space_repr: str,
    action_space_repr: str,
) -> bool:
    if not mismatch or not metadata:
        return False
    if "Config digest mismatch" not in mismatch:
        return False
    algo_ok = str(metadata.get("algorithm", "")).upper() == expected_algorithm.upper()
    obs_ok = str(metadata.get("observation_space", "")) == observation_space_repr
    act_ok = str(metadata.get("action_space", "")) == action_space_repr
    return algo_ok and obs_ok and act_ok
