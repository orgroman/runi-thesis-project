import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from pydash.utilities import retry
from tqdm import tqdm


# 1) A retry-decorated function that attempts to upload a file to OpenAI
@retry(attempts=3, delay=0.5, max_delay=150.0, scale=2.0, jitter=0)
async def upload_file(async_client, file_path):
    # You could also wrap `open(...)` in a 'with' statement if you prefer:
    # with open(file_path, "rb") as fp:
    #     return await async_client.files.create(file=fp, purpose="batch")
    return await async_client.files.create(file=open(file_path, "rb"), purpose="batch")


async def upload_file_with_sem(async_client, file_path, sem):
    # 2) Use a Semaphore to limit concurrency
    async with sem:
        return await upload_file(async_client, file_path)


async def main():
    async_client = AsyncOpenAI(api_key=os.getenv("THESIS_ALON_OPENAI_API_KEY"))
    root_path = Path(
        r"C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_train"
    )

    jsonl_files = list(root_path.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} *.jsonl files to upload.")

    # This controls how many parallel uploads are allowed
    MAX_CONCURRENT_UPLOADS = 10
    sem = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

    print("Uploading files to OpenAI...")
    tasks = []
    for jsonl_file in jsonl_files:
        tasks.append(asyncio.create_task(upload_file_with_sem(async_client, jsonl_file, sem)))

    # We'll gather results but process them as they complete, so we can keep the progress bar updated.
    openai_files = []
    with tqdm(total=len(tasks), desc="Uploading") as pbar:
        for coro in asyncio.as_completed(tasks):
            # As soon as one upload finishes, we collect its result
            openai_file = await coro
            openai_files.append(openai_file)
            pbar.update(1)

    print("All files uploaded!")

    # Create a mapping of jsonl file path to OpenAI file id
    print("Creating mapping of jsonl file path to OpenAI file id...")
    openai_files_with_ids = [
        (str(jsonl_file), openai_file.id)
        for jsonl_file, openai_file in zip(jsonl_files, openai_files)
    ]
    openai_files_mapping = dict(openai_files_with_ids)
    print("Mapping created!")

    # Save the mapping to disk
    print("Saving mapping to disk...")
    openai_files_mapping_path = root_path / "openai_files_gpt4o_mapping.json"
    with open(openai_files_mapping_path, "w", encoding="utf-8") as f:
        json.dump(openai_files_mapping, f, indent=2)

    print(f"Mapping saved to {openai_files_mapping_path}!")


if __name__ == "__main__":
    asyncio.run(main())
