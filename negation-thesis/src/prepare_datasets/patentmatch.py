import asyncio
import random
from typing import Dict, List, Union
from uuid import uuid4
import pandas as pd
from pathlib import Path
import json
import logging
import openai
import os
import orjson
from pydash import get
from pydantic import BaseModel
from .utils import (
    get_cache_dir,
    load_cache_map,
    save_cache_map,
    prepare_openai_batch_request,
)

logger = logging.getLogger(__name__)

openai_async_client = openai.AsyncClient(
    api_key=os.getenv("THESIS_OPENAI_API_KEY"),
)


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
            "id": str(uuid4()),
            "data": {
                "q1": record["text"],
                "doc1": record["text_b"],
                "metadata": record,
            },
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
            standardized_records = [orjson.loads(line) for line in f.readlines()]
        logger.info(f"Loaded {len(standardized_records)} records from the cache")
        return standardized_records

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
            model="gpt-4o-mini",
        )
        openai_records.append(openai_record)
    return openai_records


def prepare_batch_splits(records: List[Dict], batch_size: int) -> List[List[Dict]]:
    """
    Split the records into batches of the specified size.
    """
    logger.info(
        f"Preparing batch splits for {len(records)} records with batch size {batch_size}"
    )
    batches = []
    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        batches.append(batch_records)
    logger.info(f"Prepared {len(batches)} batches")
    return batches


async def prepare_openai_batch_files(jsonl_files: List[Union[Path, str]]) -> List[str]:
    """
    Prepare the OpenAI batch files for processing.
    """
    logger.info(f"Preparing OpenAI batch files for {len(jsonl_files)} files")
    futures = []

    for jsonl_file in jsonl_files:
        futures.append(
            openai_async_client.files.create(
                file=open(jsonl_file, "rb"),
                purpose="batch",
            )
        )

    openai_files = await asyncio.gather(*futures)
    logger.info(f"Prepared {len(openai_files)} OpenAI batch files")
    return openai_files


async def prepare_openai_batch_requests(openai_files) -> List[str]:
    """
    Prepare the OpenAI batch requests for processing.
    """
    logger.info(f"Preparing OpenAI batch requests for {len(openai_files)} files")
    futures = []

    for openai_file in openai_files:
        futures.append(
            openai_async_client.batches.create(
                input_file_id=openai_file["id"],
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"type": "patent_negation"},
            )
        )

    openai_batch_requests = await asyncio.gather(*futures)
    logger.info(f"Prepared {len(openai_batch_requests)} OpenAI batch requests")
    return openai_batch_requests


class PatentMatchAnnotator:
    """
    This class is used to handle the batch annotation by chatgpt
    It should be used to check the batch status, retry failed batches, and save the results.
    """

    def __init__(
        self,
        openai_records: List[Dict],
        batch_size: int = 1000,
        annotation_id: str = "annotation",
    ):
        self.openai_records = openai_records
        self.batch_size = batch_size
        self.annotation_id = annotation_id
        self.openai_files = []
        self.openai_batch_requests = {}

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
        batch_dir = get_cache_dir("patentmatch") / self.annotation_id / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        jsonl_dir = batch_dir / "jsonl_files"
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        batch_openai_files_dir = batch_dir / "openai_files"
        batch_openai_files_dir.mkdir(parents=True, exist_ok=True)
        batch_openai_requests_dir = batch_dir / "openai_requests"
        batch_openai_requests_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Batch directory: {batch_dir}")
        # Split the records into batches and write each one to each it's own jsonl file
        batch_jsonl_files = list(jsonl_dir.glob("*.jsonl"))
        if not batch_jsonl_files:
            logger.info(
                f"No batch jsonl files found in {jsonl_dir}. Creating new ones."
            )
            batches = prepare_batch_splits(self.openai_records, self.batch_size)
            for i, batch in enumerate(batches):
                batch_file = jsonl_dir / f"batch_{i}.jsonl"
                with open(batch_file, "w") as f:
                    for record in batch:
                        f.write(json.dumps(record) + "\n")
                logger.debug(f"Created batch file: {batch_file}")
        else:
            logger.info(
                f"Found {len(batch_jsonl_files)} batch jsonl files in {jsonl_dir}."
            )

        # Load the first jsonl file and count the number of input tokens
        logger.debug("Counting the number of input tokens in the first jsonl file")
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        token_sum = 0
        with open(batch_jsonl_files[0], "r") as f:
            lines = f.readlines()
            for line in lines:
                tokens = enc.encode(line)
                token_sum += len(tokens)

        model_batch_price_per_1m = 0.075
        estimated_cost = (
            (token_sum * len(batch_jsonl_files)) / 1e6
        ) * model_batch_price_per_1m
        logger.info(f"Estimated cost for batch processing: ${estimated_cost:.2f}")

        logger.info(f"Number of tokens in first jsonl file: {token_sum}")
        logger.info(
            f"Created {len(batch_jsonl_files)} batch jsonl files\n"
            f"Creating OpenAI batch requests"
        )
        
        self.openai_files = []
        cached_openai_files = list(batch_openai_files_dir.glob("*.json"))
        for cached_path in cached_openai_files:
            with open(cached_path, "r") as f:
                file_response = json.load(f)
                self.openai_files.append(file_response)
        if not self.openai_files:
            logger.info(
                f"No OpenAI files found in {batch_openai_files_dir}. Creating new ones."
            )

        if not self.openai_files:
            logger.info(
                f"No OpenAI files found in {batch_openai_files_dir}. Creating new ones."
            )
            # Create OpenAI batch file request for each jsonl file (async)
            self.openai_files = await prepare_openai_batch_files(batch_jsonl_files)
            self.openai_files = [r.to_dict() for r in self.openai_files]
            logger.info(f"Created {len(self.openai_files)} OpenAI files")
            # Save all file requests to cache dir
            for response in self.openai_files:
                # Save the file response to the cache dir
                file_path = batch_openai_files_dir / f"{response['id']}.json"
                with open(file_path, "w") as f:
                    # Save the dumped json to the file
                    json.dump(response, f, indent=4)
                                                         
                logger.debug(f"Saved OpenAI file response to {file_path}")
        else:
            logger.info(
                f"Found {len(self.openai_files)} OpenAI files in {batch_openai_files_dir}."
            )

        # Create OpenAI batch file request for each jsonl file (async)
        batch_requests = [
            json.load(f) for f in batch_openai_requests_dir.glob("*.json")
        ]
        self.openai_batch_requests = {
            x["id"]: {
                "batch_file": x
            }
            for x in self.openai_files
        }

        if not batch_requests:
            logger.info(
                f"No OpenAI batch requests found in {batch_openai_requests_dir}. Creating new ones."
            )
            # Create OpenAI batch request for each file response (async)
            batch_requests = await prepare_openai_batch_requests(self.openai_files)
            batch_requests = [r.to_dict() for r in batch_requests]
            logger.info(f"Created {len(batch_requests)} OpenAI batch requests")
            # Save all file requests to cache dir
            for response in batch_requests:
                # Save the file response to the cache dir
                file_path = batch_openai_requests_dir / f"{response['id']}.json"
                with open(file_path, "w") as f:
                    json.dump(response, f, indent=4)
                logger.debug(f"Saved OpenAI batch request to {file_path}")
        else:
            logger.info(
                f"Found {len(batch_requests)} OpenAI batch requests in {batch_openai_requests_dir}."
            )
        
        # Update the batch requests
        for batch_request in batch_requests:
            input_file_id = batch_request["input_file_id"]
            self.openai_batch_requests[input_file_id]["batch_request"] = batch_request

        logger.info(f"Created {len(batch_requests)} OpenAI batch requests")

    async def long_poll_results(self):
        """
        Long poll the OpenAI API for results.
        The status of a given Batch object can be any of the following:

        Status	Description
        validating	the input file is being validated before the batch can begin
        failed	the input file has failed the validation process
        in_progress	the input file was successfully validated and the batch is currently being run
        finalizing	the batch has completed and the results are being prepared
        completed	the batch has been completed and the results are ready
        expired	the batch was not able to be completed within the 24-hour time window
        cancelling	the batch is being cancelled (may take up to 10 minutes)
        cancelled	the batch was cancelled
        """
        logger.info(f"Long polling OpenAI API for results")
        batch_dir = get_cache_dir("patentmatch") / self.annotation_id / "batches"
        batch_openai_files_dir = batch_dir / "openai_files"
        batch_openai_requests_dir = batch_dir / "openai_requests"

        # In case the status requires recreation, we will recreate the batch request and remove the old one
        recreate_statuses = ["expired", "failed", "cancelled", "cancelling"]

        while True:
            logger.info(f"Long polling OpenAI API for results")
            # Load all openai batch requests from cache dir
            completed_count = len(
                [
                    x
                    for x in self.openai_batch_requests.values()
                    if x["status"] == "completed"
                ]
            )
            logger.debug(
                f"Completed batch requests: {completed_count}/{len(self.openai_batch_requests)}"
            )
            if completed_count == len(self.openai_batch_requests):
                logger.info(f"All batch requests completed")
                break

            for openai_batch in self.openai_batch_requests.values():
                # Check if the batch request is complete
                response = await openai_async_client.batches.retrieve(
                    openai_batch["batch_id"]
                )
                status = response["status"]
                self.openai_batch_requests[openai_batch["batch_id"]]["status"] = status
                logger.debug(
                    f"Batch request {openai_batch['batch_id']} status: {status}"
                )
                # If the batch request is complete, save the results to the cache dir
                if status in recreate_statuses:
                    # Recreate the batch request
                    logger.info(f"Recreating batch request {openai_batch['batch_id']}")
                    batch_response = await openai_async_client.batches.create(
                        input_file_id=openai_batch["openai_file_id"],
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                        metadata={"type": "patent_negation"},
                    )
                    self.openai_batch_requests[openai_batch["batch_id"]]["status"] = (
                        batch_response["status"]
                    )

                if status == "completed":
                    # Save the results to the cache dir
                    logger.info(f"Batch request {openai_batch['batch_id']} succeeded")
                    batch_response = await openai_async_client.batches.retrieve(
                        openai_batch["batch_id"]
                    )
                    file_path = (
                        batch_openai_requests_dir / f"{batch_response['id']}.json"
                    )
                    with open(file_path, "w") as f:
                        json.dump(batch_response, f, indent=4)
                    logger.debug(f"Saved OpenAI batch request to {file_path}")


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

    def get_annotator(
        self,
        batch_size: int = 1000,
        sample_size: int = 0,
        annotation_id: str = "annotation",
    ) -> PatentMatchAnnotator:
        """
        Get the annotator for the dataset.

        Args:
            batch_size (int): The size of each batch for annotation.
            random_sample_size (int): The size of the random sample to annotate.
            if 0, all records will be annotated.
        """
        logger.info(
            f"Getting annotator for the dataset with batch size {batch_size} and random sample size {sample_size}"
        )

        # Sample records if a sample size is provided
        if sample_size > 0:
            logger.info(
                f"Sampling {sample_size} records from {len(self.standarized_records)} total records"
            )
            sampled_records = random.sample(self.standarized_records, sample_size)
            return PatentMatchAnnotator.from_records(
                sampled_records, batch_size=batch_size, annotation_id=annotation_id
            )

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


async def main():
    # Set up logging configuration
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Example usage
    dataset_file = (
        r"C:\Users\orgrd\workspace\data\patentmatch_test\patentmatch_test.tsv"
    )
    dataset_obj = PatentMatchDataset.from_file(dataset_file)
    annotator = dataset_obj.get_annotator(
        batch_size=5000, sample_size=5000, annotation_id="patentmatch_test"
    )

    await annotator.create_batch_requests()
    await annotator.long_poll_results()

    print("done")


if __name__ == "__main__":
    asyncio.run(main())
