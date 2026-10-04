from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
