import asyncio
from openai import AsyncOpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

import os
import logging
import json
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COMPLETED_BATCHES_COLLECTION = "completed_batches"
ANNOTATED_SAMPLES_COLLECTION = "annotated_samples"

async def construct_annotated_dataset():
    completed_coll = mongodb_client["patent_negation"][COMPLETED_BATCHES_COLLECTION]    
    annotated_coll = mongodb_client["patent_negation"][ANNOTATED_SAMPLES_COLLECTION]
    futures = []
    async def process_completed_batch(doc):
        try:
            if 'output_version' in doc:
                input_batch = await openai_client.files.content(doc["openai_file_id"])
                input_samples = [json.loads(line) for line in input_batch.text.splitlines()]
                openai_results = doc['results']
                final_doc = {
                    "input_samples": input_samples,
                    "openai_results": openai_results
                }
                await annotated_coll.insert_one(final_doc)
        except Exception as e:
            logger.error(f"Error processing batch {doc['_id']}: {str(e)}")


    async for doc in completed_coll.find():
        # Process each completed batch
        futures.append(process_completed_batch(doc))
    
    await asyncio.gather(*futures)








async def main():
    await construct_annotated_dataset()

if __name__ == "__main__":
    print("Running main function")
    asyncio.run(main())