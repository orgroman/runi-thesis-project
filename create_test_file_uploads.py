import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI
from pydash.utilities import retry


@retry(attempts=3, delay=0.5, max_delay=150.0, scale=2.0, jitter=0)
async def create_file(async_client, file_path: Path):
    """Creates an OpenAI file upload with retry logic."""
    with open(file_path, "rb") as f:
        return await async_client.files.create(
            file=f,
            purpose="fine-tune"
        )


async def create_file_with_sem(async_client, file_path: Path, sem: asyncio.Semaphore):
    """Creates a file using a semaphore to limit concurrency."""
    async with sem:
        return await create_file(async_client, file_path)


async def main():
    async_client = AsyncOpenAI(api_key=os.getenv("THESIS_ALON_OPENAI_API_KEY"))

    # Base directory for test data
    test_data_dir = Path(
        r"C:\Users\orgrd\workspace\repos\runi-thesis-project\hidrive\patentmatch_test"
    )
    
    # Define paths for text and text_b directories
    text_dir = test_data_dir / "text"
    text_b_dir = test_data_dir / "text_b"

    # Get all .jsonl files from both directories
    text_files = list(text_dir.glob("*.jsonl"))
    text_b_files = list(text_b_dir.glob("*.jsonl"))

    if not text_files and not text_b_files:
        print("No .jsonl files found in the directories. Exiting.")
        return

    # Prepare for file uploads
    files_to_upload = text_files + text_b_files
    print(f"Will upload {len(files_to_upload)} files total:")
    for file_path in files_to_upload:
        print(f" - {file_path}")

    # Concurrency control
    MAX_CONCURRENT_UPLOADS = 2
    sem = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

    # Prepare tasks
    tasks = []
    for file_path in files_to_upload:
        tasks.append(
            asyncio.create_task(create_file_with_sem(async_client, file_path, sem))
        )

    print("Uploading files...")
    upload_results = await asyncio.gather(*tasks)
    print("Files uploaded!")

    # Create mapping dictionary
    mapping = {
        "text": {},
        "text_b": {}
    }

    # Populate mapping
    for file_path, upload_result in zip(files_to_upload, upload_results):
        file_id = upload_result.id
        str_path = str(file_path)
        
        if "text_b" in str_path:
            mapping["text_b"][str_path] = file_id
        else:
            mapping["text"][str_path] = file_id

    # Save mapping to JSON
    mapping_file = test_data_dir / "openai_files_gpt4o_mapping_test.json"
    print(f"Saving mapping to {mapping_file}")
    
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print("All done!")


if __name__ == "__main__":
    asyncio.run(main())
