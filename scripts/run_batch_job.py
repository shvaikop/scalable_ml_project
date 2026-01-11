import json
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent          # scripts/
REPO_ROOT = HERE.parent                         # repo root (assuming scripts/ is in repo root)

NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "lake_water_level_batch_inference_pipeline.ipynb"
CONFIG_PATH   = REPO_ROOT / "configs" / "sensors.json"

ARG_MAP = {
    "latitude": "--latitude",
    "longitude": "--longitude",
    "sensor_id": "--sensor-id",
    "sensor_name": "--sensor-name",
    "water_level_fg_version": "--water-level-fg-version",
    "weather_fg_version": "--weather-fg-version",
    "fv_version": "--fv-version",
    "model_version": "--model-version",
    "sensor_url": "--sensor-url",
}

def run_one(cfg: dict) -> None:
    cmd = ["ipython", str(NOTEBOOK_PATH), "--",]
    for k, flag in ARG_MAP.items():
        if k not in cfg:
            raise KeyError(f"Config missing key '{k}': {cfg}")
        cmd += [flag, str(cfg[k])]

    print("\nRunning:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)

def main() -> None:
    configs = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(configs, list) or not configs:
        raise ValueError("configs/sensors.json must contain a non-empty JSON list.")

    for cfg in configs:
        run_one(cfg)

if __name__ == "__main__":
    main()
