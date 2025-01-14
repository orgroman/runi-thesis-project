import os
import json
import asyncio
from pathlib import Path

# IMPORTANT: Make sure you have installed the relevant openai (or custom) library
# that provides `AsyncOpenAI`.
from openai import AsyncOpenAI


async def main():
    """
    1) Gathers all batch JSON files in `batch_responses/`.
    2) For each, retrieves the updated batch info from OpenAI using `batches.retrieve(...)`.
    3) Checks if the batch is 'completed'.
    4) If so, retrieves the content of the batch's output file and saves it to `completed_files/`.
    """

    # Initialize the AsyncOpenAI client with your environment variable
    client = AsyncOpenAI(api_key=os.getenv("THESIS_ALON_OPENAI_API_KEY"))

    # Directory containing the old JSON response files from your original script
    batch_response_dir = Path(
        r"C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_train\batch_responses"
    )

    # Directory where we'll store completed batches' JSONL outputs
    completed_dir = Path(
        r"C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_train\completed_files"
    )
    completed_dir.mkdir(exist_ok=True)

    # Collect all *.json files from `batch_responses/`
    batch_json_files = list(batch_response_dir.rglob('*.json'))
    if not batch_json_files:
        print(f"No JSON files found in {batch_response_dir}. Exiting.")
        return

    # We'll retrieve updated batch info concurrently
    batch_ids = []
    retrieve_tasks = []
    for batch_json_path in batch_json_files:
        with open(batch_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        batch_response = data.get("batch_response", {})
        batch_id = batch_response.get("id")
        if not batch_id:
            print(f"Skipping {batch_json_path.name}: no 'batch_response.id' found.")
            continue

        batch_ids.append(batch_id)
        retrieve_tasks.append(client.batches.retrieve(batch_id))

    # Run all retrieval requests simultaneously
    print(f"Retrieving {len(retrieve_tasks)} batches from OpenAI...")
    retrieved_batches = await asyncio.gather(*retrieve_tasks, return_exceptions=True)

    # Now we pair each retrieved object with its original batch_id
    id_to_batch_data = {}
    for batch_id, batch_info_or_exc in zip(batch_ids, retrieved_batches):
        # Check if the result is an exception
        if isinstance(batch_info_or_exc, Exception):
            print(f"Error retrieving batch {batch_id}: {batch_info_or_exc}")
            continue
        id_to_batch_data[batch_id] = batch_info_or_exc

    # Filter for completed batches
    completed_batches = {
        bid: batch_obj
        for (bid, batch_obj) in id_to_batch_data.items()
        if getattr(batch_obj, "status", None) == "completed"
    }

    if not completed_batches:
        print("No batches with status='completed'. Exiting.")
        return

    # Retrieve the content for each completed batch (output_file_id)
    content_tasks = []
    for batch_id, batch_obj in completed_batches.items():
        output_file_id = getattr(batch_obj, "output_file_id", None)
        if not output_file_id:
            print(f"Batch {batch_id} is completed but has no 'output_file_id'. Skipping.")
            continue
        content_tasks.append((batch_id, client.files.content(output_file_id)))

    if not content_tasks:
        print("No completed batches have an output_file_id. Exiting.")
        return

    # Run all content file retrievals concurrently
    print(f"Retrieving file content for {len(content_tasks)} completed batches...")
    retrieved_contents = await asyncio.gather(*[t[1] for t in content_tasks], return_exceptions=True)

    # Save each completed batch's output into `completed_files/`
    for (batch_id, _), file_content_or_exc in zip(content_tasks, retrieved_contents):
        if isinstance(file_content_or_exc, Exception):
            print(f"Error retrieving file content for batch {batch_id}: {file_content_or_exc}")
            continue

        # file_content_or_exc should be an object with .text or .content (depending on the library)
        jsonl_text = file_content_or_exc.text
        # Parse the JSON lines
        jsonl_lines = [json.loads(line) for line in jsonl_text.split('\n') if line.strip()]

        # Save the lines as JSONL to `completed_files/batch_{batch_id}.jsonl`
        out_path = completed_dir / f"batch_{batch_id}.jsonl"
        with open(out_path, 'w', encoding='utf-8') as out_f:
            for line_dict in jsonl_lines:
                out_f.write(json.dumps(line_dict, ensure_ascii=False) + '\n')

        print(f"Saved content for batch {batch_id} to: {out_path}")

    print("All done retrieving and saving completed batches!")


if __name__ == "__main__":
    asyncio.run(main())
