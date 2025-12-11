import os
import json
import hashlib

def load_json(path: str, logger:callable):
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)

def compute_config_hash(config: dict) -> str:
    """Compute a hash for a camera config."""
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
