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

ANNOTATED_SAMPLES_COLLECTION = "annotated_samples"
ANNOTATED_SAMPLES_B_COLLECTION = "annotated_samples_b"
ANNOTATED_PAIRS_COLLECTION = "annotated_pairs"

async def construct_annotated_dataset():
    completed_coll = mongodb_client["patent_negation"][ANNOTATED_SAMPLES_COLLECTION]
    completed_coll_b = mongodb_client["patent_negation"][ANNOTATED_SAMPLES_B_COLLECTION]
    
    annotated_samples = {}
    async for doc in completed_coll.find():
        # Process each completed batch
        input_samples = doc["input_samples"]
        openai_results = doc["openai_results"]
        for i, sample in enumerate(input_samples):
            sample_id = sample["custom_id"]
            result_id = openai_results[i]["custom_id"]
            assert sample_id == result_id

            id_parts = sample_id.split("_")
            part_1 = id_parts.pop(-1)
            part_2 = id_parts.pop(-1)
            dict_key = (part_1, part_2)
            if dict_key not in annotated_samples:
                annotated_samples[dict_key] = {}
            annotated_samples[dict_key].update({
                f"text_a": {
                    "sample": sample,
                    "result": openai_results[i]
                }
            })

        
    async for doc in completed_coll_b.find():
        # Process each completed batch
        input_samples = doc["input_samples"]
        openai_results = doc["openai_results"]
        for i, sample in enumerate(input_samples):
            sample_id = sample["custom_id"]
            result_id = openai_results[i]["custom_id"]
            assert sample_id == result_id

            id_parts = sample_id.split("_")
            part_1 = id_parts.pop(-1)
            part_2 = id_parts.pop(-1)
            dict_key = (part_1, part_2)
            if dict_key not in annotated_samples:
                annotated_samples[dict_key] = {}
            annotated_samples[dict_key].update({
                f"text_b": {
                    "sample": sample,
                    "result": openai_results[i]
                }
            })
    
    # Filter out samples that don't have both text_a and text_b
    annotated_samples = {k: v for k, v in annotated_samples.items() if "text_a" in v and "text_b" in v}

    # Save the annotated samples to the database
    annotated_pairs_coll = mongodb_client["patent_negation"][ANNOTATED_PAIRS_COLLECTION]
    docs = []
    for key, value in annotated_samples.items():
        docs.append({
            "annotated_sample": value
        })
    
    await annotated_pairs_coll.insert_many(docs)
    
    print('done')



async def main():
    await construct_annotated_dataset()

if __name__ == "__main__":
    print("Running main function")
    asyncio.run(main())