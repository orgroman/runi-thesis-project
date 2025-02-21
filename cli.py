import argparse
import logging
import os
import pandas as pd
import textwrap
from pathlib import Path
from runi_thesis_project.config_loader import load_configs
from runi_thesis_project.clients.aiopika_df import AioPikaDataFrameClient
from pydash import get
import tqdm
from pydantic import BaseModel
from typing import List, Optional
import json
from time import time

logger = logging.getLogger(__name__)

# configure stdout logging to show INFO level messages
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
def remove_dependent():
    pass


def main(args):
    config_dir = os.getenv("CONFIG_DIR", str(Path.cwd() / "configs"))
    config = load_configs(config_dir)
    logger.info(f"Loading configs from {config_dir}")

    if args.sub_command == "train_model":
        logger.info(
            f"Running `train_model` with:\n"
            f"Train input: {args.train_input}\n"
            f"Train output directory: {args.train_output_dir}"
        )
    elif args.sub_command == "remove_dependent":
        logger.info(
            f"Running `remove_dependent` with:\n"
            f"Input TSV: {args.input_tsv}\n"
            f"Output TSV: {args.output_tsv}\n"
            f"Column: {args.column}"
        )
    elif args.sub_command == "prepare_jsonl_column":
        # Load the input csv, split to batches of 25,000 rows max each and prepare for openai
        logger.info(f"Reading input file from: {args.input_tsv}")
        file_type = Path(args.input_tsv).suffix
        sep = "\t" if file_type == ".tsv" else ","
        df = pd.read_csv(args.input_tsv, sep=sep)
        column = args.column
        output_dir = Path(args.input_tsv).parent / f"output_jsonl_{column}"
        output_dir.mkdir(exist_ok=True)
        batch_size = 5000
        num_batches = len(df) // batch_size + 1

        class NegationResponse(BaseModel):
            negation_present: bool
            negation_types: Optional[List[str]]
            short_explanation: str

        from runi_thesis_project.models.negation_detection.prompts import NEGATION_PROMPT_DETAILED

        def jsonl_openai_line(row):
            text = row[column]
            messages = [
                {
                    "role": "system",
                    "content": NEGATION_PROMPT_DETAILED
                },
                {
                    "role": "user",
                    "content": f"Analyze the following text: {text}"
                }
            ]
            body = {
                "model": "gpt-4o-mini",
                "messages": messages,
                # This is the Beta Chat Completion parse approach:
                # We instruct it to parse the output into our Pydantic model
                "response_format": {
                    "type": "json_schema",
                    "schema": NegationResponse.model_json_schema()  # The Pydantic model's JSON schema
                },
                "max_tokens": 500
            }
            patent_application_id = row["patent_application_id"]
            row_index = row["index"]
            custom_id = f'request_{column}_{patent_application_id}_{row_index}'
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }
            return line

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_df = df.iloc[start_idx:end_idx]
            logger.info(f"Preparing batch {i + 1}/{num_batches} with {len(batch_df)} rows")
            lines = batch_df.apply(jsonl_openai_line, axis=1)

            with open(output_dir / f"batch_{i}.jsonl", "w", encoding='utf-8') as f:
                for line in lines:
                    # Use json.dumps() to ensure valid JSON is written
                    f.write(json.dumps(line, ensure_ascii=False))
                    f.write("\n")
                
                    
        
        
    elif args.sub_command == "run_negation_detection_model":
        start_time = time()
        logger.info("Starting negation detection process...")
        
        input_path = Path(r"G:\My Drive\Colab Notebooks\research_data\thesis\patentmatch\patentmatch_test\patentmatch_test_no_claims.csv")
        logger.info(f"Checking input file at: {input_path}")
        if not input_path.exists():
            logger.error(f"Input file not found at: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        logger.info(f"Reading input file from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Successfully loaded {len(df)} rows from input file")
        logger.info(f"Memory usage of dataframe: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
        
        from runi_thesis_project.models.negation_detection.prompts import NEGATION_PROMPT_DETAILED
        from openai import AsyncOpenAI
        import asyncio
        from tqdm.asyncio import tqdm as atqdm
        import time
        from datetime import datetime
        
        api_key = os.getenv("THESIS_ALON_OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment variables")
            raise ValueError("THESIS_ALON_OPENAI_API_KEY environment variable not set")
            
        logger.info("Initializing OpenAI client...")
        async_client = AsyncOpenAI(api_key=api_key)
        
        async def process_text(text, index):
            try:
                start = time.time()
                logger.debug(f"Processing text {index}: {text[:100]}...")
                response = await async_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": NEGATION_PROMPT_DETAILED},
                        {"role": "user", "content": f"Analyze the following text: {text}"}
                    ]
                )
                duration = time.time() - start
                logger.debug(f"Text {index} processed in {duration:.2f}s")
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error processing text {index}: {str(e)}")
                logger.exception("Full traceback:")
                return f"ERROR: {str(e)}"

        async def run_model():
            logger.info("Starting model processing with concurrency limit of 5")
            semaphore = asyncio.Semaphore(5)
            processed = 0
            errors = 0
            
            async def process_with_semaphore(text, index):
                nonlocal processed, errors
                async with semaphore:
                    result = await process_text(text, index)
                    processed += 1
                    if result.startswith("ERROR"):
                        errors += 1
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed}/{len(df)} texts. Errors: {errors}")
                    return result
            
            tasks = [process_with_semaphore(text, i) for i, text in enumerate(df[args.column])]
            logger.info(f"Created {len(tasks)} tasks for processing")
            results = await atqdm.gather(*tasks)
            return results

        logger.info(f"Starting processing of {len(df)} rows at {datetime.now()}")
        df["negation_detected"] = asyncio.run(run_model())
        
        # Calculate statistics
        error_count = sum(1 for x in df["negation_detected"] if str(x).startswith("ERROR"))
        success_rate = ((len(df) - error_count) / len(df)) * 100
        
        # Save results
        output_path = Path(args.output_tsv)
        logger.info(f"Saving results to {output_path}")
        logger.info(f"Processing statistics:")
        logger.info(f"Total processed: {len(df)}")
        logger.info(f"Successful: {len(df) - error_count}")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        
        df.to_csv(output_path, sep=sep, index=False)
        total_time = time.time() - start_time
        logger.info(f"Process completed in {total_time:.2f} seconds")
        logger.info(f"Average time per record: {total_time/len(df):.2f} seconds")
        logger.info("Done!")
    
        
#         logger.info(f"Read {len(df)} rows from {args.input_tsv}")

#         # Run the negation detection model
#         async def run_model():
#             futures = []
#             for text in tqdm.tqdm(df[args.column]):
#                 future = negation_model.acomplete_chat(
#                     messages=[
#                         {
#                             "role": "system",
#                             "content": textwrap.dedent("""
# You will receive text passages taken from patent claims or related legal documents. Your task is to determine whether **linguistic negation** is present in the text. 
# ### What to Look For
# - **Explicit negation words** (e.g., "not," "never," "cannot," "no," "none," "nor," "without").  
# - **Phrases or constructions** that semantically negate or exclude something (e.g., “excluding,” “disclaims,” “fails to,” “lacks,” “prohibits,” etc.).  
# - **Subtle or context-dependent negation** (e.g., "it is impossible," "it does not apply to," "not limited to").  

# ### Required Output
# 1. **Presence of Negation**: State whether negation is found.
# 2. **Evidence**: Highlight or specify the negation indicators (words or phrases) that led to your decision.
# ANSWER ONLY YES OR NO, NOTHING ELSE!

# **Example:**

# **Input**:  
# > “A method of manufacturing a device **without** using solvent X, wherein the device **does not** require a safety lock.”

# **Output**: YES
#                             """),
#                         },
#                         {
#                             "role": "user",
#                             "content": text,
#                         },
#                     ]
#                 )
#                 futures.append(future)
            
#             logger.info(f"Running {len(futures)} futures")
#             results = []
#             semaphore = asyncio.Semaphore(5)
#             # from tqdm.asyncio import tqdm as atqdm
#             async for future in futures:
#                 # limit concurrent requests to 5 with semaphore
#                 result = await future
#                 print(result)
#                 results.append(result)

#                 # async with semaphore:
#                 #     result = await future
#                 #     print(result)
#                 #     results.append(result)

#             #results = await tqdm.gather(*futures)
#             return results

#         import asyncio

#         df["negation_detected"] = asyncio.run(run_model())

#         # Write the results to the output TSV



    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI for the research thesis different pipelines and experiments"
    )

    # Add subparsers to handle multiple commands
    subparsers = parser.add_subparsers(dest="sub_command")

    parser.add_argument(
        "--command",
        help="Command to run",
        required=False,
        type=str,
        choices=["remove_dependent"],
    )
    train_model_p = subparsers.add_parser(
        "train_model",
        help="Train a model on the input dataset",
    )
    train_model_p.add_argument(
        "--train_input",
        help="Input dataset to train the model on",
        required=False,
        type=str,
    )
    train_model_p.add_argument(
        "--train_output_dir",
        help="Output directory to save the trained model",
        default="output",
        type=str,
    )

    # Define the `remove_dependent` subcommand
    remove_dependent_claims_p = subparsers.add_parser(
        "remove_dependent",
        help="Detect and remove dependent patent claims from the input dataset",
    )
    remove_dependent_claims_p.add_argument(
        "--input_tsv",
        help="Input dataset to remove dependent claims from",
        required=True,
        type=str,
    )
    remove_dependent_claims_p.add_argument(
        "--output_tsv",
        help="Output dataset to save the results to",
        default="output.tsv",
        type=str,
    )
    remove_dependent_claims_p.add_argument(
        "--column",
        help="Column to remove dependent claims from",
        required=True,
        type=str,
    )

    run_negation_detection_model_p = subparsers.add_parser(
        "run_negation_detection_model",
        help="Run the negation detection model on the input dataset",
    )
    run_negation_detection_model_p.add_argument(
        "--input_tsv",
        help="Input dataset to run the negation detection model on",
        required=True,
        type=str,
    )
    run_negation_detection_model_p.add_argument(
        "--output_tsv",
        help="Output dataset to save the results to",
        default="output.tsv",
        type=str,
    )
    run_negation_detection_model_p.add_argument(
        "--column",
        help="Column to run the negation detection model on",
        required=True,
        type=str,
    )
    
    create_jsonl_p = subparsers.add_parser(
        "prepare_jsonl_column",
        help="Prepare the input TSV file for OpenAI API by splitting it into JSONL files"
    )
    create_jsonl_p.add_argument(
        "--input_tsv",
        help="Input TSV file to prepare for OpenAI API",
        required=True,
        type=str
    )
    create_jsonl_p.add_argument(
        "--column",
        help="Column to prepare for OpenAI API",
        required=True,
        type=str
    )

    # Parse arguments
    args = parser.parse_args()
    main(args)
