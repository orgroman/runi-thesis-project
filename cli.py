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
    elif args.sub_command == "run_negation_detection_model":
        from runi_thesis_project.models.negation_detection.llamafile import (
            LLamaFileModel,
        )

        negation_model = LLamaFileModel.create_model(
            **get(config, "models.negation_detection")
        )
        logger.info(
            f"Running `run_negation_detection_model` with:\n"
            f"Input TSV: {args.input_tsv}\n"
            f"Output TSV: {args.output_tsv}\n"
            f"Column: {args.column}"
        )

        # Read the input TSV
        logger.info("Reading data from the tsv file")
        df = pd.read_csv(args.input_tsv, sep="\t")
        
        async def queue_transform_df():
            client = AioPikaDataFrameClient(
            amqp_url="amqp://admin:password@127.0.0.1/",
            input_queue_name="df_queue",
            results_queue_name="df_results_queue"
            )
        
            # 1. Connect and publish the original DataFrame
            await client.connect()
            
            async with asyncio.TaskGroup() as tg:
                print("Publishing DataFrame rows to input queue...")
                #tg.create_task(client.publish_dataframe(df))
                print("Done publishing.")

                # 2. Consume from input queue, transform, and re-publish to results queue
                #    We'll consume exactly len(original_df) messages
                print("Starting consumer to transform and re-publish...")
                async def transform_example(row: dict) -> dict:
                    negation_result = await negation_model.acomplete_chat(
                            messages=[
                                {
                                    "role": "system",
                                    "content": textwrap.dedent("""

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. The user will provide one or more sentences from various sources (including academic, technical, or legal domains). Your task is to determine whether **linguistic negation** is present in the text.

**Important Requirements**:
1. You must only output “YES” or “NO” (a single word) to indicate presence or absence of any form of negation.  
2. Provide no additional information or context beyond “YES” or “NO.”

**Background**:
Negation can appear in multiple forms:
- **Sentence-level negation**: “not,” “no,” “never,” “cannot,” “nor,” etc.  
- **Constituent negation**: negating only part of the sentence.  
- **Morphological negation**: negative prefixes/affixes (e.g., “un-,” “dis-,” “in-,” “non-,” “asynchronous” for “not synchronous”).  
- **Lexical negation**: words inherently carrying a negative meaning (e.g., “fail,” “deny,” “lack,” “impossible,” “refuse,” “reject,” etc.).  
- **Negative Polarity Items** (NPIs): words like “any,” “ever,” “at all,” used in contexts requiring negation.

Your procedure:
1. Check if any negation cue exists (explicit negation markers, morphological negation, inherently negative verbs/adjectives, NPIs in a negative context, etc.).
2. Output only “YES” if negation is found.
3. Output only “NO” if no negation is found.

Here are some examples:


<|begin_example|> (Morphological Negation) Input: “The process is asynchronous, preventing synchronous updates.”  Output: YES <|end_of_example|>
<|begin_example|> (Lexical Negation) Input: “The system fails to meet the required standard.” Output: YES <|end_of_example|>
<|begin_example|> (Standard Sentential Negation) Input: “They did not complete the review by the deadline.”  Output: YES <|end_of_example|>

<|begin_example|> (Constituent Negation) Input: “The final output is incorrect.”  Output: YES <|end_of_example|>
<|begin_example|> (No Negation) Input: “We tested the prototype in multiple languages to ensure consistency.”  Output: NO <|end_of_example|>

Reminder:
- Output only “YES” or “NO”, with no further text.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is the text to analyze:
[User’s sentence goes here]

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                    """),
                                },
                                {
                                    "role": "user",
                                    "content": row[args.column],
                                },
                            ]
                        )
                    choices_0_answer = get(negation_result, "choices.0.message.content", "NO")
                    row["negation_detected"] = choices_0_answer
                    print(f"Negation result: {choices_0_answer}")
                    return row
                
                tg.create_task(client.consume_and_transform(
                    transform_callback=transform_example,
                    expected_count=len(df)
                ))
            print("Done consuming and transforming to results queue.")

            # 3. Finally, consume from results queue to build the final DataFrame
            print("Consuming transformed rows from results queue...")
            transformed_df = await client.consume_results(
                expected_count=len(df)
            )
            print("Done consuming from results queue.\nTransformed DataFrame:")
            print(transformed_df)
            transformed_df.to_csv(args.output_tsv, sep="\t", index=False)

            await client.close()
        
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(queue_transform_df())
    
        
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

    # Parse arguments
    args = parser.parse_args()
    main(args)
