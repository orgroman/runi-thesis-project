from pathlib import Path
from typing import Union
from pydash import set_with
import json

def load_configs(config_dir: Union[str, Path]) -> dict:
    config_dir = Path(config_dir)
    
    # rglob any config.json files in the config directory
    config_files = config_dir.rglob("config.json")
    
    # Create a dictionary where the keys are the sub paths of the config files and values are the contents of the config files
    config_dict = {}
    for config_file in config_files:
        with open(config_file, "r") as f:
            config_key = str(config_file.relative_to(config_dir).parent).replace('\\','.').replace('/','.')
            config_value = json.load(f)
            set_with(config_dict, config_key, config_value, lambda: {})
    
    return config_dict
    
    