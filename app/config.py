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


def load_selected_model() -> ModelConfig:
    config_path = Path(
        os.getenv("MODEL_CONFIG", str(DEFAULT_CONFIG_PATH))
    ).resolve()

    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    selected = os.getenv("MODEL_NAME", raw["selected_model"])
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
