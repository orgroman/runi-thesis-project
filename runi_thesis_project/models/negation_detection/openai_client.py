import asyncio
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from openai import AsyncOpenAI
from datetime import datetime
from tqdm.asyncio import tqdm as atqdm
import time

logger = logging.getLogger(__name__)

class NegationResponse(BaseModel):
    negation_present: bool
    negation_types: Optional[List[str]]
    short_explanation: str

class OpenAINegationClient:
    def __init__(
        self, 
        api_key: str,
        model: str = "gpt-4",
        batch_size: int = 5000,
        max_concurrent_requests: int = 5,
        max_retries: int = 3,
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        self.max_retries = max_retries
        self.client = AsyncOpenAI(api_key=api_key)
        
    async def prepare_batches(
        self, 
        df: pd.DataFrame, 
        text_column: str,
        output_dir: Path,
        prompt_template: str
    ) -> List[Path]:
        """Prepare JSONL batch files for processing"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        batch_files = []
        
        num_batches = len(df) // self.batch_size + 1
        logger.info(f"Preparing {num_batches} batches of size {self.batch_size}")
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_file = output_dir / f"batch_{i}.jsonl"
            batch_files.append(batch_file)
            
            lines = []
            for _, row in batch_df.iterrows():
                text = row[text_column]
                request = {
                    "custom_id": f"request_{text_column}_{row.name}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": prompt_template},
                            {"role": "user", "content": f"Analyze the following text: {text}"}
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "schema": NegationResponse.model_json_schema()
                        },
                        "max_tokens": 500
                    }
                }
                lines.append(json.dumps(request, ensure_ascii=False))
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
                
            logger.info(f"Created batch file {batch_file} with {len(lines)} requests")
            
        return batch_files

    async def process_text(self, text: str, prompt_template: str, index: int) -> Dict[str, Any]:
        """Process a single text with retries"""
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt_template},
                        {"role": "user", "content": f"Analyze the following text: {text}"}
                    ],
                    response_format={
                        "type": "json_schema",
                        "schema": NegationResponse.model_json_schema()
                    }
                )
                duration = time.time() - start
                logger.debug(f"Text {index} processed in {duration:.2f}s")
                
                result = json.loads(response.choices[0].message.content)
                return NegationResponse(**result).model_dump()
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for text {index}: {str(e)}")
                if attempt == self.max_retries - 1:
                    return {"error": str(e), "text_index": index}
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def process_batch(
        self,
        texts: List[str],
        prompt_template: str,
        start_index: int = 0
    ) -> List[Dict[str, Any]]:
        """Process a batch of texts concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def process_with_semaphore(text: str, idx: int) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_text(text, prompt_template, idx)
        
        tasks = [
            process_with_semaphore(text, i + start_index) 
            for i, text in enumerate(texts)
        ]
        
        return await atqdm.gather(*tasks)

    async def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        prompt_template: str,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Process entire dataframe with batching and progress tracking"""
        start_time = time.time()
        logger.info(f"Starting processing of {len(df)} texts at {datetime.now()}")
        
        all_results = []
        errors = 0
        processed = 0
        
        for i in range(0, len(df), self.batch_size):
            batch_df = df.iloc[i:i + self.batch_size]
            batch_results = await self.process_batch(
                batch_df[text_column].tolist(),
                prompt_template,
                start_index=i
            )
            
            all_results.extend(batch_results)
            processed += len(batch_results)
            errors += sum(1 for r in batch_results if "error" in r)
            
            logger.info(f"Processed {processed}/{len(df)} texts. Errors: {errors}")
        
        # Add results to dataframe
        df = df.copy()
        df["negation_detection"] = all_results
        
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        total_time = time.time() - start_time
        success_rate = ((len(df) - errors) / len(df)) * 100
        
        logger.info(f"Processing completed in {total_time:.2f} seconds")
        logger.info(f"Average time per record: {total_time/len(df):.2f} seconds")
        logger.info(f"Success rate: {success_rate:.2f}%")
        
        return df

    @classmethod
    def create_client(cls, **kwargs):
        """Factory method to create a client instance"""
        return cls(**kwargs)
