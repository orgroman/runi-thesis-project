from openai import AsyncOpenAI
import os
import asyncio
from pathlib import Path
import json
from typing import List


async def main():
    async_client = AsyncOpenAI(api_key=os.getenv("THESIS_ALON_OPENAI_API_KEY"))
    root_path = Path(
        r"C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_train"
    )
    jsonl_files = list(root_path.rglob("*.jsonl"))
    jsonl_files = [jsonl_files[0]]

    print("Creating OpenAI files...")
    openai_files = []
    for jsonl_f in jsonl_files:
        print(f"Creating OpenAI file for {jsonl_f}...")
        openai_files.append(
            await async_client.files.create(file=open(jsonl_f, "rb"), purpose="batch")
        )

    # gather all the futures
    # openai_files = await asyncio.gather(*openai_files_futures)
    openai_files_with_ids = [
        (jsonl_f, openai_file)
        for jsonl_f, openai_file in zip(jsonl_files, openai_files)
    ]
    print("OpenAI files created!")
    
    # write the file ids to disk
    print("Writing OpenAI file ids to disk...")
    openai_files_disk_dump = []
    for jsonl_f, openai_file in openai_files_with_ids:
        openai_files_disk_dump.append({
            "json_path": str(jsonl_f),
            "openai_file_id": openai_file.id
        })
    
    # write the disk dump to disk
    openai_files_disk_dump_path = root_path / "openai_files_disk_dump.json"
    with open(openai_files_disk_dump_path, "w") as f:
        f.write(json.dumps(openai_files_disk_dump))

    print("Creating batches...")
    create_batch_futures = []
    for jsonl_f, openai_file in openai_files_with_ids:
        create_batch_futures.append(
            async_client.batches.create(
                input_file_id=openai_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"input_file": str(jsonl_f)},
            )
        )

    # gather all the futures
    batches = await asyncio.gather(*create_batch_futures)
    print("Batches created!")
    
    # write all json responses from batches to disk
    print("Writing batch responses to disk...")
    batch_response_dir = root_path / "batch_responses"
    batch_response_dir.mkdir(exist_ok=True)
    for jsonl_f, batch_response in zip(jsonl_files, batches):
        batch_response_path = batch_response_dir / f"{batch_response.id}.json"
        print(f"Writing batch response to {batch_response_path}... for file {jsonl_f}")
        with open(batch_response_path, "w") as f:
            f.write(json.dumps({
                "json_path": str(jsonl_f),
                "batch_response": batch_response.model_dump()
            }) )
    
    print("All done!")


if __name__ == "__main__":
    asyncio.run(main())
