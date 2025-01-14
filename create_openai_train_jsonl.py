from functools import partial
import json
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from runi_thesis_project.models.negation_detection.prompts import NEGATION_PROMPT_DETAILED

logger = logging.getLogger(__name__)

class NegationResponse(BaseModel):
    negation_present: bool
    negation_types: Optional[List[str]]
    short_explanation: str

def jsonl_openai_line(row, column):
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
            "json_schema": {
                "name": "negation_response",
                "schema": {
                    "type": "object",
                    "strict": True,
                    **NegationResponse.model_json_schema()  # The Pydantic model's JSON schema                    
                }                
                }
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

def main():
    tqdm.pandas()
    # Load the input csv, split to batches of 25,000 rows max each and prepare for openai
    column = 'text'
    input_file = Path(r'C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_train\patentmatch_train_no_claims.csv')
    logger.info(f"Reading input file from: {input_file}")    
    file_type = input_file.suffix
    sep = "\t" if file_type == ".tsv" else ","
    df = pd.read_csv(input_file, sep=sep)
    column = 'text_b'
    output_dir = input_file.parent / f"jsonl_{column}_{input_file.stem}"
    output_dir.mkdir(exist_ok=True)
    batch_size = 5000
    num_batches = len(df) // batch_size + 1

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_df = df.iloc[start_idx:end_idx]
        logger.info(f"Preparing batch {i + 1}/{num_batches} with {len(batch_df)} rows")
        lines = batch_df.progress_apply(jsonl_openai_line, axis=1, args=(column,))

        with open(output_dir / f"batch_{i}.jsonl", "w", encoding='utf-8') as f:
            for line in lines:
                # Use json.dumps() to ensure valid JSON is written
                f.write(json.dumps(line, ensure_ascii=False))
                f.write("\n")
                
                    
if __name__ == "__main__":    
    main()