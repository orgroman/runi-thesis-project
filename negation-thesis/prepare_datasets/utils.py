import json
from pathlib import Path
import logging
import os
from typing import Union
logger = logging.getLogger(__name__)

def get_cache_dir(dataset_dir: Union[str, Path]) -> Path:
    """
    Get the cache directory for storing datasets.
            
    Returns:
        str: The path to the cache directory.
    """
    env_name = "THESIS_CACHE_DIR"
    cache_dir = os.environ.get(env_name)
    logger.debug(f"Cache directory provided from env {env_name}: {cache_dir}")
    target_cache_dir = Path(cache_dir)
    if cache_dir is None:
        logger.debug("No cache directory provided, using default.")
        target_cache_dir = Path.home() / ".thesis_cache" / "datasets"
                
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
    cache_map_path = dataset_cache_dir / "cache_map.json"
    if cache_map_path.exists():
        with open(cache_map_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Cache map file {cache_map_path} is corrupted. Creating a new one.")
    
    return {}

def save_cache_map(cache_map: dict, dataset_cache_dir: Path) -> None:
    """
    Save the cache map to the specified directory.
    
    Args:
        cache_map (dict): The cache map to save.
        cache_dir (Path): The path to the cache directory.
    """
    cache_map_path = dataset_cache_dir / "cache_map.json"
    with open(cache_map_path, 'w') as f:
        json.dump(cache_map, f, indent=4)
    logger.info(f"Cache map saved to {cache_map_path}")
