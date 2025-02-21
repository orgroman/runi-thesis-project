import os
import json
import asyncio
from pathlib import Path
from typing import List

from openai import AsyncOpenAI
from pydash.utilities import retry


@retry(attempts=3, delay=0.5, max_delay=150.0, scale=2.0, jitter=0)
async def create_batch(async_client, file_id: str, metadata: dict):
    """Creates an OpenAI batch with retry logic."""
    return await async_client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata,
    )


async def create_batch_with_sem(async_client, file_id: str, metadata: dict, sem: asyncio.Semaphore):
    """Creates a batch using a semaphore to limit concurrency."""
    async with sem:
        return await create_batch(async_client, file_id, metadata)


async def main():
    async_client = AsyncOpenAI(api_key=os.getenv("THESIS_ALON_OPENAI_API_KEY"))

    # Path to the JSON file that contains info about uploaded files
    file_id_json = Path(
        r"C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_train\openai_files_gpt4o_mapping_new.json"
    )

    # Load JSON data
    with open(file_id_json, 'r', encoding='utf-8') as f:
        file_id_data = json.load(f)

    # file_id_data['text_b'] and file_id_data['text'] are dictionaries, e.g.:
    #
    # "text_b": {
    #   "C:\\path\\to\\file1.jsonl": "file-xyz",
    #   "C:\\path\\to\\file2.jsonl": "file-abc",
    #   ...
    # }
    #
    # Convert the first 2 items to a list of dicts: [{"file_path":..., "id":...}, ...]
    text_b_dict = file_id_data.get("text_b", {})
    text_dict = file_id_data.get("text", {})

    text_b_files = [
        {"file_path": path_str, "id": file_id}
        for path_str, file_id in list(text_b_dict.items())[:2]
    ]

    text_files = [
        {"file_path": path_str, "id": file_id}
        for path_str, file_id in list(text_dict.items())[:2]
    ]

    # Combine them for batch creation
    files_to_batch = text_b_files + text_files

    if not files_to_batch:
        print("No valid files found in the JSON. Exiting.")
        return

    print(f"Will create batches for {len(files_to_batch)} files total:")
    for file_info in files_to_batch:
        print(f" - {file_info['file_path']} ({file_info['id']})")

    # Concurrency control
    MAX_CONCURRENT_CREATIONS = 2
    sem = asyncio.Semaphore(MAX_CONCURRENT_CREATIONS)

    # Prepare tasks
    tasks = []
    for file_info in files_to_batch:
        file_path = file_info["file_path"]
        file_id = file_info["id"]

        metadata = {"input_file": file_path}
        tasks.append(
            asyncio.create_task(create_batch_with_sem(async_client, file_id, metadata, sem))
        )

    print("Creating batches...")
    batch_results = await asyncio.gather(*tasks)
    print("Batches created!")

    # Where we want to store the batch responses
    root_path = file_id_json.parent
    batch_response_dir = root_path / "batch_responses"
    batch_response_dir.mkdir(exist_ok=True)

    print("Writing batch responses to disk...")
    for file_info, batch_response in zip(files_to_batch, batch_results):
        # The batch response object from openai-python typically has an '.id'
        # Adjust if your library version differs
        batch_id = batch_response.id
        batch_response_path = batch_response_dir / f"{batch_id}.json"

        output_data = {
            "file_path": file_info["file_path"],
            "file_id": file_info["id"],
            "batch_response": batch_response.model_dump(),  
        }

        print(f"  -> {batch_response_path}")
        with open(batch_response_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

    print("All done!")


if __name__ == "__main__":
    asyncio.run(main())
