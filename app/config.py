from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_type: str
    path: Path
    class_names: list[str]
    image_size: tuple[int, int]


def get_raw_config() -> tuple[Path, dict]:
    config_path = Path(
        os.getenv("MODEL_CONFIG", str(DEFAULT_CONFIG_PATH))
    ).resolve()

    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    return config_path, raw


def load_selected_model(model_name: str | None = None) -> ModelConfig:
    config_path, raw = get_raw_config()
    selected = model_name or os.getenv("MODEL_NAME", raw["selected_model"])
    models = raw["models"]

    if selected not in models:
        raise ValueError(
            f"Unknown model '{selected}'. Available: {sorted(models)}"
        )

    item = models[selected]
    model_path = Path(item["path"])
    if not model_path.is_absolute():
        model_path = (config_path.parent / model_path).resolve()

    return ModelConfig(
        name=selected,
        model_type=item["type"].lower(),
        path=model_path,
        class_names=list(item["class_names"]),
        image_size=tuple(item.get("image_size", [224, 224])),
    )


def list_available_models() -> dict[str, dict]:
    config_path, raw = get_raw_config()
    result = {}
    default_name = os.getenv("MODEL_NAME", raw["selected_model"])

    for name, item in raw["models"].items():
        model_path = Path(item["path"])
        if not model_path.is_absolute():
            model_path = (config_path.parent / model_path).resolve()
        
        file_size_mb = (
            round(model_path.stat().st_size / (1024 * 1024), 3)
            if model_path.exists()
            else None
        )

        result[name] = {
            "name": name,
            "type": item["type"].lower(),
            "path": str(model_path),
            "file_exists": model_path.exists(),
            "file_size_mb": file_size_mb,
            "class_names": list(item["class_names"]),
            "image_size": list(item.get("image_size", [224, 224])),
            "is_active": (name == default_name),
        }
    return result
