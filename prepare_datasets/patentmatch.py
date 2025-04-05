from typing import Dict, List
import pandas as pd
from pathlib import Path
import json
import logging
logger = logging.getLogger(__name__)
from .utils import get_cache_dir, load_cache_map, save_cache_map

def standarize_dataset(dataset_file: str) -> Dict:
    """
    Transform the input data into a standardized format with caching support.
    
    Args:
        dataset_file (str): The path to the dataset file.
        cache_dir (str, optional): Directory to use for caching. If None, uses default cache.
        
    Returns:
        dict: The transformed data in a standardized format.
    """    
    dataset_file = Path(dataset_file)
    logger.info(f"Loading and standardizing dataset from {dataset_file}")
    
    # Get cache directory
    patentmatch_dir = get_cache_dir("patentmatch")
    
    standardized_file = patentmatch_dir / f"{dataset_file.stem}_standardized.json"
    
    # Initialize cache map
    cache_map = load_cache_map(patentmatch_dir)
    cache_values = cache_map.get(str(dataset_file), {})
    standardized_file = cache_values.get("standardized_file", standardized_file)
    if Path(standardized_file).is_file():
        logger.info(f"Standardized file already exists: {standardized_file}")
        # Load and return the standardized records from the cache

        with open(standardized_file, 'r') as f:
            try:
                standardized_records = json.load(f)
                # Verify cache integrity
                if len(standardized_records) > 0:
                    logger.info(f"Successfully loaded {len(standardized_records)} records from cache")
                    return standardized_records
                else:
                    logger.warning("Cache file exists but contains no records. Regenerating...")
            except json.JSONDecodeError:
                logger.warning(f"Cache file {standardized_file} is corrupted. Regenerating...")
    
    logger.info("Standardized file not found in cache. Proceeding to generate a new one.")
    
    # Load and process the dataset
    df = pd.read_csv(dataset_file, sep='\t', header=0, index_col=0)
    data_dict = df.to_dict(orient='records')
    logger.debug(f"Loaded {len(data_dict)} records from {dataset_file}")
    
    # Standardize the records
    standardized_records = {}
    for idx, record in enumerate(data_dict):
        standardized_records[str(idx)] = {
            "q1": record['text'],
            "doc1": record['text_b'],            
            "metadata": record
        }
    
    # Save to cache
    with open(standardized_file, 'w') as f:
        json.dump(standardized_records, f, indent=4)
    
    # Update cache map
    cache_map[str(dataset_file)] = {
        "standardized_file": str(standardized_file),
        "original_file": str(dataset_file),            
        "num_records": len(data_dict),
        "last_updated": pd.Timestamp.now().isoformat()
    }

    save_cache_map(cache_map, patentmatch_dir)
    
    logger.info(f"Transformed {len(data_dict)} records to standardized format and saved to {standardized_file}")
    return standardized_records

class PatentMatchDataset:
    """
    This class is used to prepare the PatentMatch dataset for training and evaluation.
    It includes methods to load the dataset, preprocess it, and split it into training,
    validation, and test sets.
    """

    def __init__(self, dataset_df: pd.DataFrame):
        self.dataset_df = dataset_df
    
    @classmethod
    def from_original_file(cls, dataset_file: str):
        """
        Load the dataset from a file and return an instance of PatentMatchDataset.
        
        Args:
            dataset_file (str): The path to the dataset file."
            """
        dataset_df = standarize_dataset(dataset_file)
        return cls(dataset_df)
            

    def load_dataset(self):
        # Load the dataset from the specified directory
        pass

    def preprocess(self):
        # Preprocess the dataset (e.g., tokenization, normalization)
        pass

    def split_dataset(self):
        # Split the dataset into training, validation, and test sets
        pass

if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Example usage
    dataset_file = r'C:\Users\orgrd\workspace\data\patentmatch_test\patentmatch_test.tsv'
    dataset_obj = PatentMatchDataset.from_original_file(dataset_file)
    print('done')
