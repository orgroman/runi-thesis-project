from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os
import json

def get_openai_key():
    """Retrieve OpenAI API key from Azure Key Vault"""
    try:
        credential = DefaultAzureCredential()
        vault_url = "https://kvrunithesis.vault.azure.net/"
        secret_client = SecretClient(vault_url=vault_url, credential=credential)
        secret = secret_client.get_secret("alon-thesis-openai-key")
        os.environ["OPENAI_API_KEY"] = secret.value
        os.environ["THESIS_ALON_OPENAI_API_KEY"] = secret.value
        print("Successfully retrieved OpenAI API key from Azure Key Vault")
    except Exception as e:
        print(f"Error retrieving secret from Key Vault: {str(e)}")
        raise

get_openai_key()

async def main():
    motor_client = AsyncIOMotorClient("mongodb://user:pass@localhost:27017")
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    db_name = "patent_negation"
    batch_requests_collection_name = "batch_requests"
    file_collection_name = "openai_files"

    # Go over all batch requests attempt retrieval of results and if successful, insert into completed_batches
    batch_requests = motor_client[db_name][batch_requests_collection_name]
    completed_batches = motor_client[db_name]["completed_batches"]

    async for batch_request in batch_requests.find():
        # Check if the batch request is complete
        # retrieve the results        
        batch_results = await openai_client.batches.retrieve(batch_request["batch_id"])
        if batch_results.status == "completed":
            # Insert the batch request into completed_batches
            # Get the file results
            openai_file_id = batch_request["openai_file_id"]
            file_response = await openai_client.files.content(openai_file_id)

            raw_print_text = file_response.text
            response_list = [json.loads(x) for x in raw_print_text.split('\n')[:-1]]
            batch_result_doc = {
                "openai_file_id": openai_file_id,
                "batch_id": batch_request["batch_id"],
                "results": response_list
            }
            # get the print() output of the raw_print_text

            await completed_batches.insert_one(batch_result_doc)
        else:
            print(f"Batch request {batch_request['_id']} is not yet complete")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
