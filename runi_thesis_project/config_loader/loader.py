from pathlib import Path
from typing import Union
import json

def load_configs(config_dir: Union[str, Path]) -> dict:
    config_dir = Path(config_dir)
    
    # rglob any config.json files in the config directory
    config_files = config_dir.rglob("config.json")
    
    # Create a dictionary where the keys are the sub paths of the config files and values are the contents of the config files
    config_dict = {}
    for config_file in config_files:
        with open(config_file, "r") as f:
            config_dict[config_file.relative_to(config_dir)] = json.load(f)
    
    return config_dict
    
    