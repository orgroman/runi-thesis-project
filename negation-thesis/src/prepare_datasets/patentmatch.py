import random
from typing import Dict, List
from uuid import uuid4
import pandas as pd
from pathlib import Path
import json
import logging

from pydantic import BaseModel
from .utils import (
    get_cache_dir,
    load_cache_map,
    save_cache_map,
    prepare_openai_batch_request,
)

logger = logging.getLogger(__name__)


class ResponseSchema(BaseModel):
    q2: str
    doc2: str


schema_obj = ResponseSchema.model_json_schema()

ANNOTATE_NEGATION_PROMPT = """
You are provided with a query-document pair (q1, doc1) from the patent domain. Your task is to create a new query-document pair (q2, doc2) similar in style to the NevIR dataset. Specifically:
Generate a new document (doc2) that is identical to the original document (doc1) except for a key negation or reversal of meaning.
Formulate a new query (q2) that precisely aligns with the altered meaning of the new document (doc2), making it relevant exclusively to doc2 and not to the original doc1.

Here are examples:
Example 1:
q1: "The image streaming apparatus wherein the control unit compares resolutions and processes images accordingly."
doc1: "The device includes a control unit, display, and wireless transmission module."
q2: "The image streaming apparatus wherein the control unit does not compare resolutions upon request."
doc2: "The device includes a control unit that does not compare resolutions, display, and wireless transmission module."

Example 2:
q1: "A biometric apparatus comprising an instruction issuing unit to measure pulse and energy."
doc1: "Acceleration signals are analyzed equally during steps for running detection."
q2: "A biometric apparatus without an instruction issuing unit to measure pulse."
doc2: "Acceleration signals cannot be correlated to specific movements due to the absence of an instruction issuing unit."
"""

ANNOTATE_NEGATION_INPUT_PROMPT = """
Now, given the following input pair:
q1: "{original_query}"
doc1: "{original_document}"
Generate the corresponding new query-document pair (q2, doc2) following the instructions provided.
"""


def generate_standardized_records(dataset_file: str) -> List[Dict]:
    """
    Generate standardized records from the input dataset.

    Args:
        dataset_file (str): The path to the dataset file.

    Returns:
        dict: The standardized records.
    """
    logger.info(f"Loading and standardizing dataset from {dataset_file}")

    # Load the dataset
    df = pd.read_csv(dataset_file, sep="\t", header=0, index_col=0)
    data_dict = df.to_dict(orient="records")

    # Standardize the records as list of dictionaries (jsonl format)
    standardized_records = []
    for record in data_dict:
        standardized_record = {
            {
                "id": str(uuid4.uuid4()),
                "data": {
                    "q1": record["text"],
                    "doc1": record["text_b"],
                    "metadata": record                    
                }
            }

        }
        standardized_records.append(standardized_record)

    return standardized_records


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
    standardized_file = patentmatch_dir / f"{dataset_file.stem}_standardized.jsonl"

    if Path(standardized_file).is_file():
        logger.info(f"Standardized file already exists: {standardized_file}")
        # Load and return the standardized records from the cache

        with open(standardized_file, "r") as f:
            # Load the jsonl file into a list of dictionaries
            standardized_records = [json.loads(line) for line in f.readlines()]
        logger.info(f"Loaded {len(standardized_records)} records from the cache")

    logger.info(
        "Standardized file not found in cache. Proceeding to generate a new one."
    )
    standardized_records = generate_standardized_records(dataset_file)
    logger.debug(f"Generated {len(standardized_records)} records from the dataset")

    # Save the jsonl file to the cache directory
    with open(standardized_file, "w") as f:
        for record in standardized_records:
            f.write(json.dumps(record) + "\n")

    logger.info(
        f"Transformed {len(standardized_records)} records to standardized format and saved to {standardized_file}"
    )
    return standardized_records


def prepare_openai_jsonl(records: List[Dict]) -> List[str]:
    """
    Prepare the records for OpenAI API requests in JSONL format.
    """
    logger.info(f"Preparing OpenAI JSONL requests for {len(records)} records")
    openai_records = []
    for record in records:
        record_data = record["data"]
        openai_record = prepare_openai_batch_request(
            custom_id=record["id"],
            messages=[
                {
                    "role": "system",
                    "content": ANNOTATE_NEGATION_PROMPT,
                },
                {
                    "role": "user",
                    "content": ANNOTATE_NEGATION_INPUT_PROMPT.format(
                        original_query=record_data["q1"],
                        original_document=record_data["doc1"],
                    ),
                },
            ],
            schema=ResponseSchema,
            schema_name="patentmatch_negation",
            max_tokens=500,
            model="gpt-4o-mini"            
        )
        openai_records.append(openai_record)
    return openai_records


class PatentMatchAnnotator:
    """
    This class is used to handle the batch annotation by chatgpt
    It should be used to check the batch status, retry failed batches, and save the results.
    """

    def __init__(self,
                 openai_records: List[Dict],
                 batch_size: int = 1000,):
        self.openai_records = openai_records
        self.batch_size = batch_size
        
    @classmethod
    def from_records(cls, records: List[Dict], **kwargs):
        logger.info(f"Creating PatentMatchAnnotator with {kwargs}")
        openai_records = prepare_openai_jsonl(records)
        cls_obj = cls(openai_records=openai_records, **kwargs)
        return cls_obj
    
    @classmethod
    def from_file(cls, jsonl_file: str, **kwargs):
        logger.info(f"Loading records from {jsonl_file}")
        with open(jsonl_file, "r") as f:
            records = [json.loads(line) for line in f.readlines()]
        
        return cls.from_records(records, **kwargs)

    async def create_batch_requests(self):
        """
        Create the batch requests for OpenAI API.
        """
        logger.info(f"Creating batch requests for {len(self.openai_records)} records")
        get_cache_dir("patentmatch")
        # Placeholder for the actual batch request creation logic
        pass
            

    def load_jsonl_batches(self, batch_size: int = 1000):
        """
        Load the jsonl batches from the file.

        Args:
            batch_size (int): The size of each batch for annotation.
        """
        logger.info(f"Loading jsonl batches with batch size {batch_size}")
        self.openai_records = prepare_openai_jsonl(self.openai_records)
            
    def annotate_batch(self, records: List[Dict]):
        """
        Annotate a batch of records using chatgpt.

        Args:
            records (List[Dict]): The records to annotate.
        """
        # Placeholder for the actual annotation logic
        pass

    def save_results(self, output_file: str):
        """
        Save the annotated results to a file.

        Args:
            output_file (str): The path to the output file.
        """
        with open(output_file, "w") as f:
            json.dump(self.annotated_records, f, indent=4)


class PatentMatchDataset:
    """
    This class is used to prepare the PatentMatch dataset for training and evaluation.
    It includes methods to load the dataset, preprocess it, and split it into training,
    validation, and test sets.
    """

    def __init__(self, standarized_records: Dict):
        self.standarized_records = standarized_records

    @classmethod
    def from_file(cls, dataset_file: str):
        """
        Load the dataset from a file and return an instance of PatentMatchDataset.

        Args:
            dataset_file (str): The path to the dataset file."
        """
        logger.info(f"Loading dataset from {dataset_file}")
        standarized_records = standarize_dataset(dataset_file)
        return cls(standarized_records=standarized_records)
    
    def get_annotator(self, batch_size: int = 1000, sample_size: int = 0):
        """
        Get the annotator for the dataset.

        Args:
            batch_size (int): The size of each batch for annotation.
            random_sample_size (int): The size of the random sample to annotate.
            if 0, all records will be annotated.
        """
        logger.info(f"Getting annotator for the dataset with batch size {batch_size} and random sample size {random_sample_size}")
        
        # Sample records if a sample size is provided
        if sample_size > 0:
            logger.info(f"Sampling {sample_size} records from {len(self.standarized_records)} total records")    
            sampled_records = random.sample(self.standarized_records, sample_size)
            return PatentMatchAnnotator.from_records(sampled_records, batch_size=batch_size)

    def annotate_negation(self, batch_size: int = 1000, random_sample_size: int = 0):
        """
        Annotate the negation in the dataset using chatgpt.

        Args:
            batch_size (int): The size of each batch for annotation.
            random_sample_size (int): The size of the random sample to annotate.
            if 0, all records will be annotated.
        """
        logger.info(
            f"Annotating negation in the dataset with batch size {batch_size}"
            f"and random sample size {random_sample_size}"
        )

        annotator = PatentMatchAnnotator(batch_size=batch_size)

        # Split records into batches and annotate
        records = list(self.standarized_records.values())
        for i in range(0, len(records), batch_size):
            batch_records = records[i : i + batch_size]
            annotator.annotate_batch(batch_records)

        # Save the annotated results
        output_file = "annotated_patentmatch.json"
        annotator.save_results(output_file)


if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Example usage
    dataset_file = (
        r"C:\Users\orgrd\workspace\data\patentmatch_train\patentmatch_train.tsv"
    )
    dataset_obj = PatentMatchDataset.from_file(dataset_file)
    print("done")
