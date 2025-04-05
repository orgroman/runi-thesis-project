import pandas as pd
import logging

logger = logging.getLogger(__name__)

def transform_to_standardized_df(dataset_file: str) -> pd.DataFrame:
    """
    Transform the input data into a standardized format.
    
    Args:
        data (list): The input data to be transformed.
        
    Returns:
        list: The transformed data in a standardized format.
    """
    # Load the patentmatch tsv file into a DataFrame
    df = pd.read_csv(dataset_file, sep='\t', header=0, index_col=0)

    return [item.lower() for item in data]


if __name__ == "__main__":
    # Example usage
    dataset_file = r'C:\Users\orgrd\workspace\data\patentmatch_test\patentmatch_train.tsv'
    standardized_df = transform_to_standardized_df(dataset_file)
    print(standardized_df.head())