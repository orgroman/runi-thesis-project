import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from temporalio import activity

logger = logging.getLogger(__name__)

@activity.defn
async def load_patent_data(csv_path: str) -> Dict[str, Any]:
    """
    Load and validate patent data from CSV file.
    
    Args:
        csv_path: Path to CSV file containing patent data
        
    Returns:
        Dict containing dataframe info and pickle path
        
    Raises:
        FileNotFoundError: If the CSV file does not exist
        ValueError: If the CSV schema is invalid
    """
    activity.logger.info(f"Loading patent data from {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        # Load CSV data
        df = pd.read_csv(csv_path)
        
        # Validate schema - check required columns
        required_columns = ['patent_application_id', 'text', 'text_b']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {missing_columns}")
        
        # Save dataframe to temporary pickle file for efficient passing between activities
        temp_dir = Path('/c:/Users/orgrd/workspace/repos/runi-thesis-project/temp')
        temp_dir.mkdir(exist_ok=True)
        
        pickle_path = temp_dir / f"patent_data_{pd.util.hash_pandas_object(df).sum()}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(df, f)
        
        activity.logger.info(f"Loaded {len(df)} records from CSV")
        
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dataframe_pickle": str(pickle_path)
        }
        
    except Exception as e:
        activity.logger.error(f"Error loading CSV data: {str(e)}")
        raise
