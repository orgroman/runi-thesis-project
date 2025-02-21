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
        from runi_thesis_project.models.negation_detection.client import create_model
        from runi_thesis_project.models.negation_detection.prompts import NEGATION_PROMPT_DETAILED
        import asyncio
        
        api_key = os.getenv("THESIS_ALON_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("THESIS_ALON_OPENAI_API_KEY environment variable not set")

        # Create the client
        client = create_model(
            provider_type="openai",
            api_key=api_key,
            model=args.model,
            batch_size=args.batch_size,
            max_concurrent_requests=args.max_concurrent
        )

        # Determine which stages to run
        stages = args.stages.lower().split(',') if args.stages != 'all' else ['prepare_jsonl', 'process_api', 'save_results']
        
        async def run_pipeline():
            # Initialize paths
            input_path = Path(r"G:\My Drive\Colab Notebooks\research_data\thesis\patentmatch\patentmatch_test\patentmatch_test_no_claims.csv")
            output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "negation_detection_output"
            output_dir.mkdir(exist_ok=True)
            
            # Load input data
            logger.info(f"Loading input file: {input_path}")
            df = pd.read_csv(input_path)
            
            # Stage 1: Prepare JSONL files
            if 'prepare_jsonl' in stages:
                logger.info("Stage 1: Preparing JSONL files")
                batch_files = await client.prepare_batches(
                    df=df,
                    text_column=args.column,
                    output_dir=output_dir / "jsonl_batches",
                    prompt_template=NEGATION_PROMPT_DETAILED
                )
                logger.info(f"Created {len(batch_files)} batch files")
            
            # Stage 2: Process through API
            if 'process_api' in stages:
                logger.info("Stage 2: Processing through API")
                df = await client.process_dataframe(
                    df=df,
                    text_column=args.column,
                    prompt_template=NEGATION_PROMPT_DETAILED
                )
            
            # Stage 3: Save results
            if 'save_results' in stages:
                logger.info("Stage 3: Saving results")
                output_file = output_dir / "negation_results.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")
            
            return df

        # Run the pipeline
        df = asyncio.run(run_pipeline())
        logger.info("Pipeline completed successfully")

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

    # Add new stage control for negation detection
    run_negation_detection_model_p.add_argument(
        "--stages",
        help="Stages to run (prepare_jsonl,process_api,save_results)",
        default="all",
        type=str
    )
    run_negation_detection_model_p.add_argument(
        "--model",
        help="Model to use for negation detection",
        default="gpt-4",
        type=str
    )
    run_negation_detection_model_p.add_argument(
        "--batch_size",
        help="Batch size for processing",
        default=5000,
        type=int
    )
    run_negation_detection_model_p.add_argument(
        "--max_concurrent",
        help="Maximum concurrent API requests",
        default=5,
        type=int
    )
    run_negation_detection_model_p.add_argument(
        "--output_dir",
        help="Directory for intermediate JSONL files",
        default=None,
        type=str
    )

    # Parse arguments
    args = parser.parse_args()
    main(args)
