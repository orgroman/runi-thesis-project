import asyncio
import json
import os
from pathlib import Path
import tempfile
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from openai import AsyncOpenAI
from pydash.utilities import retry
from tqdm import tqdm
from motor.motor_asyncio import AsyncIOMotorClient



def get_openai_key():
    """Retrieve OpenAI API key from Azure Key Vault"""
    try:
        # Initialize the Azure credentials
        credential = DefaultAzureCredential()
        
        # Create a secret client
        vault_url = f"https://kvrunithesis.vault.azure.net/"
        secret_client = SecretClient(vault_url=vault_url, credential=credential)
        
        # Get the secret
        secret = secret_client.get_secret("alon-thesis-openai-key")
        
        # Set as environment variable
        os.environ["OPENAI_API_KEY"] = secret.value
        os.environ["THESIS_ALON_OPENAI_API_KEY"] = secret.value
        
        print("Successfully retrieved OpenAI API key from Azure Key Vault")
    except Exception as e:
        print(f"Error retrieving secret from Key Vault: {str(e)}")
        raise

# Retrieve and set the OpenAI API key
get_openai_key()
mongodb_client = AsyncIOMotorClient("mongodb://user:pass@localhost:27017")
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JSONL_B_COLLECTION = "jsonl_b_batches"
OPENAI_FILES_B_COLLECTION = "openai_files_b"
BATCH_REQUESTS_B_COLLECTION = "batch_requests_b"

jsonl_b_coll = mongodb_client["patent_negation"][JSONL_B_COLLECTION]
openai_files_b_coll = mongodb_client["patent_negation"][OPENAI_FILES_B_COLLECTION]
batch_requests_b_coll = mongodb_client["patent_negation"][BATCH_REQUESTS_B_COLLECTION]

# 1) A retry-decorated function that attempts to upload a file to OpenAI
@retry(attempts=3, delay=0.5, max_delay=150.0, scale=2.0, jitter=0)
async def upload_file(async_client, file_path):
    # You could also wrap `open(...)` in a 'with' statement if you prefer:
    # with open(file_path, "rb") as fp:
    #     return await async_client.files.create(file=fp, purpose="batch")
    jsonl_content = file_path["content"]
    # convert the string to file pointer
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(jsonl_content)
        temp_file_path = temp_file.name        

    file_response = await async_client.files.create(file=open(temp_file_path, "rb"), purpose="batch")
    file_id = file_response.id
    # upload the file response to the openai_files_b collection
    await openai_files_b_coll.insert_one({"openai_file_id": file_response.id})
    # create a batch request
    batch_request = await async_client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"type": "patent_negation"}
    )
    # upload the batch request to the batch_requests_b collection
    await batch_requests_b_coll.insert_one({
        "batch_id": batch_request.id,
        "openai_file_id": file_response.id
    })
    return file_response



async def upload_file_with_sem(async_client, file_path, sem):
    # 2) Use a Semaphore to limit concurrency
    async with sem:
        return await upload_file(async_client, file_path)


async def main():
    # root_path = Path(
    #     r"C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_train"
    # )

    # jsonl_files = list(root_path.rglob("*.jsonl"))
    # print(f"Found {len(jsonl_files)} *.jsonl files to upload.")

    # jsonl_b_docs = await jsonl_b_coll.find()

    # This controls how many parallel uploads are allowed
    MAX_CONCURRENT_UPLOADS = 10
    sem = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

    print("Uploading files to OpenAI...")
    tasks = []
    async for jsonl_doc in jsonl_b_coll.find():
        tasks.append(asyncio.create_task(upload_file_with_sem(async_client, jsonl_doc, sem)))

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
