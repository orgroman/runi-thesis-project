import json
from pathlib import Path
import logging
import os
from typing import Dict, List, Type, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def prepare_batch_request(
    custom_id: str,
    messages: List[Dict],
    schema: Type[BaseModel],
    schema_name: str = "negation_schema",
    max_tokens: int = 500,
    model: str = "gpt-4o-mini",
) -> Dict:
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
        },
    }

    return request


def get_cache_dir(dataset_dir: Union[str, Path]) -> Path:
    """
    Get the cache directory for storing datasets.

    Returns:
        str: The path to the cache directory.
    """
    env_name = "THESIS_CACHE_DIR"
    cache_dir = os.environ.get(env_name)
    logger.debug(f"Cache directory provided from env {env_name}: {cache_dir}")
    default_cache_dir = Path.home() / ".thesis_cache" / "datasets"

    target_cache_dir = default_cache_dir
    if cache_dir is None:
        logger.debug("Cache directory not provided, using default.")
    else:
        logger.debug(f"Cache directory provided from env {env_name}: {cache_dir}")
        target_cache_dir = Path(cache_dir)

    # Create the directory if it doesn't exist
    target_cache_dir = target_cache_dir / dataset_dir
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directory created at: {target_cache_dir}")

    return target_cache_dir


def load_cache_map(dataset_cache_dir: Path) -> dict:
    """
    Load the cache map from the specified directory.

    Args:
        cache_dir (Path): The path to the cache directory.

    Returns:
        dict: The loaded cache map.
    """
    logger.debug(f"Loading cache map from {dataset_cache_dir}")
    cache_map_path = dataset_cache_dir / "cache_map.json"
    if cache_map_path.exists():
        with open(cache_map_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning(
                    f"Cache map file {cache_map_path} is corrupted. Creating a new one."
                )

    return {}


def save_cache_map(cache_map: dict, dataset_cache_dir: Path) -> None:
    """
    Save the cache map to the specified directory.

    Args:
        cache_map (dict): The cache map to save.
        cache_dir (Path): The path to the cache directory.
    """
    logger.debug(f"Saving cache map: {cache_map}")
    cache_map_path = dataset_cache_dir / "cache_map.json"
    with open(cache_map_path, "w") as f:
        json.dump(cache_map, f, indent=4)
    logger.info(f"Cache map saved to {cache_map_path}")
