from io import BytesIO
import aiofiles
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os
import json
import logging
from datetime import datetime
from bson import ObjectId
import asyncio
import signal
import sys
from pydash import get
from mongodb_async import check_collection_exists
from aiolimiter import AsyncLimiter  # Rate limit concurrency

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB collection names
JSONL_BATCHES_COLLECTION = "jsonl_b_batches"
OPENAI_FILES_COLLECTION = "openai_files_b"
BATCH_REQUESTS_COLLECTION = "batch_requests_b"
COMPLETED_BATCHES_COLLECTION = "completed_batches_b"
ANNOTATED_SAMPLES_COLLECTION = "annotated_samples_b"
DB_NAME = "patent_negation"

# Polling configuration
POLLING_INTERVAL_SECONDS = 300  # 5 minutes
MAX_POLLING_ATTEMPTS = None  # Set to an integer for limited polls, None for unlimited
SHUTDOWN_FLAG = False  # Flag for graceful shutdown

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

async def check_file_expired(openai_client, file_id):
    """Check if an OpenAI file has expired"""
    try:
        file_info = await openai_client.files.retrieve(file_id)
        return False  # If no error, file is not expired
    except Exception as e:
        if "No such file" in str(e) or "expired" in str(e).lower():
            logger.info(f"File {file_id} is expired or not found")
            return True
        else:
            logger.error(f"Error checking file {file_id} status: {str(e)}")
            raise

async def reupload_jsonl_file(openai_client: AsyncOpenAI, mongodb_client, db_name, jsonl_batch_id):
    """Reupload a JSONL file from MongoDB to OpenAI"""
    try:
        # Get the JSONL batch from MongoDB
        jsonl_collection = mongodb_client[db_name][JSONL_BATCHES_COLLECTION]
        jsonl_batch = await jsonl_collection.find_one({"_id": ObjectId(jsonl_batch_id)})
        
        if not jsonl_batch:
            logger.error(f"JSONL batch {jsonl_batch_id} not found")
            return None
        
        # Create a temporary file to upload
        temp_file_path = f"/tmp/batch_{jsonl_batch_id}.jsonl"
        with open(temp_file_path, 'w') as f:
            f.write(jsonl_batch["content"])
        
        # Upload the file to OpenAI
        with open(temp_file_path, 'rb') as f:
            response = await openai_client.files.create(
                file=f,
                purpose="batch"
            )
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        logger.info(f"Reuploaded file for batch {jsonl_batch_id}, new file ID: {response.id}")
        return response.id
    
    except Exception as e:
        logger.error(f"Error reuploading JSONL file: {str(e)}")
        return None

async def create_batch_request(openai_client: AsyncOpenAI, mongodb_client, db_name, file_id):
    """Create a new batch request for a file"""
    try:
        # Create a batch request
        batch_request = await openai_client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"type": "patent_negation"}
        )
        
        # Save the batch request to MongoDB
        batch_requests = mongodb_client[db_name][BATCH_REQUESTS_COLLECTION]
        batch_doc = {
            "batch_id": batch_request.id,
            "openai_file_id": file_id, 
            "status": "pending",
            "created_at": datetime.now()
        }
        
        await batch_requests.insert_one(batch_doc)
        logger.info(f"Created new batch request {batch_request.id} for file {file_id}")
        return batch_request.id
    
    except Exception as e:
        logger.error(f"Error creating batch request: {str(e)}")
        return None

async def handle_file_management(openai_client: AsyncOpenAI, mongodb_client, db_name):
    """Manage OpenAI files and batch requests"""
    try:
        # Check if required collections exist
        collections_to_check = [
            OPENAI_FILES_COLLECTION,
            BATCH_REQUESTS_COLLECTION,
            COMPLETED_BATCHES_COLLECTION,
            JSONL_BATCHES_COLLECTION
        ]
        
        for coll_name in collections_to_check:
            exists = await check_collection_exists(db_name, coll_name, mongodb_client)
            if not exists:
                logger.warning(f"Collection {coll_name} does not exist in database {db_name}. It will be created as needed.")
        
        files_collection = mongodb_client[db_name][OPENAI_FILES_COLLECTION]
        batch_requests = mongodb_client[db_name][BATCH_REQUESTS_COLLECTION]
        completed_batches = mongodb_client[db_name][COMPLETED_BATCHES_COLLECTION]
        jsonl_batches_collection = mongodb_client[db_name][JSONL_BATCHES_COLLECTION]
        annotated_samples_collection = mongodb_client[db_name][ANNOTATED_SAMPLES_COLLECTION]

        # check if the files_collection is empty if so, then we need to upload all the jsonl files from the collection jsonl_batches
        # we need to limit the concurrent uploads to 5

        async def process_file(file_doc, openai_client, mongodb_client, db_name, semaphore):
            """Process a single file and upload directly from memory."""
            async with semaphore:
                try:
                    jsonl_batch_id = file_doc["_id"]
                    jsonl_content = file_doc["content"]

                    # Convert content to a file-like object (BytesIO)
                    file_obj = BytesIO(jsonl_content.encode("utf-8"))

                    logger.info(f"Uploading batch {jsonl_batch_id} to OpenAI")

                    # Upload to OpenAI
                    result = await openai_client.files.create(file=file_obj, purpose="batch")

                    # Insert metadata into MongoDB
                    await mongodb_client[db_name]["openai_files"].insert_one({
                        "openai_file_id": result.id,
                        "uploaded_at": datetime.utcnow(),
                        "jsonl_batch_id": jsonl_batch_id
                    })

                    logger.info(f"Uploaded batch {jsonl_batch_id} successfully")

                except Exception as e:
                    logger.error(f"Error processing batch {jsonl_batch_id}: {e}")

        # Semaphore for concurrency control
        # semaphore = asyncio.Semaphore(5)
        # tasks = []

        # logger.info("Starting to upload jsonl files to OpenAI")

        # async for file_doc in jsonl_batches_collection.find():
        #     tasks.append(process_file(file_doc, openai_client, mongodb_client, db_name, semaphore))

        # await asyncio.gather(*tasks)

        # logger.info("Finished uploading jsonl files to OpenAI")

        CONCURRENCY_LIMIT = 5  # Set concurrency limit
        limiter = AsyncLimiter(CONCURRENCY_LIMIT)

        async def process_file_doc(file_doc, openai_client, mongodb_client, db_name, files_collection, batch_requests, completed_batches):
            async with limiter:
                try:
                    if not file_doc.get("openai_file_id"):
                        logger.info(f"File {file_doc['_id']} has no OpenAI file ID, skipping")
                        return

                    openai_file_id = file_doc["openai_file_id"]
                    is_expired = await check_file_expired(openai_client, openai_file_id)
                    
                    if is_expired:
                        logger.info(f"File {openai_file_id} is expired, reuploading")
                        new_file_id = await reupload_jsonl_file(openai_client, mongodb_client, db_name, file_doc["jsonl_batch_id"])
                        
                        if new_file_id:
                            await files_collection.update_one(
                                {"_id": file_doc["_id"]},
                                {"$set": {"openai_file_id": new_file_id, "uploaded_at": datetime.now()}}
                            )
                            await create_batch_request(openai_client, mongodb_client, db_name, new_file_id)
                        return
                    
                    batch_request = await batch_requests.find_one({"openai_file_id": openai_file_id})
                    completed_batch = await completed_batches.find_one({"openai_file_id": openai_file_id})
                    
                    if completed_batch:
                        if batch_request:
                            await batch_requests.delete_one({"_id": batch_request["_id"]})
                            logger.info(f"Deleted batch request {batch_request['_id']} as it's already completed")
                        return
                    
                    if batch_request:
                        try:
                            batch_results = await openai_client.batches.retrieve(batch_request["batch_id"])
                            
                            if batch_results.status == "completed":
                                try:
                                    file_response = await openai_client.files.content(batch_results.output_file_id)
                                    raw_print_text = file_response.text
                                    response_list = [json.loads(x) for x in raw_print_text.split('\n') if x.strip()]
                                    
                                    batch_result_doc = {
                                        "openai_file_id": openai_file_id,
                                        "batch_id": batch_request["batch_id"],
                                        "results": response_list,
                                        "output_version": "v2",
                                        "completed_at": datetime.now()
                                    }

                                    input_batch = await openai_client.files.content(openai_file_id)
                                    input_samples = [json.loads(line) for line in input_batch.text.splitlines()]
                                    final_doc = {
                                        "input_samples": input_samples,
                                        "openai_results": response_list
                                    }
                                    
                                    await annotated_samples_collection.insert_one(final_doc)
                                    await batch_requests.delete_one({"_id": batch_request["_id"]})
                                    logger.info(f"Batch request {batch_request['batch_id']} completed and saved")
                                except Exception as e:
                                    logger.error(f"Error retrieving batch results: {str(e)}")
                            
                            elif batch_results.status == "failed":
                                # error_message = batch_results.get("errors", {}).get("data", [{}])[0].get("message", "Unknown error")
                                error_message = get(batch_results, "errors.data.0.message", "Unknown error")
                                logger.error(f"Batch {batch_request['batch_id']} failed: {error_message}")
                                
                                if "token limit" in error_message.lower():
                                    await batch_requests.delete_one({"_id": batch_request["_id"]})
                                    logger.info(f"Deleted rate-limited batch request {batch_request['batch_id']}")
                                    #await asyncio.sleep(5)
                                    await create_batch_request(openai_client, mongodb_client, db_name, openai_file_id)
                                
                                elif "expired" in error_message.lower():
                                    await batch_requests.delete_one({"_id": batch_request["_id"]})
                                    new_file_id = await reupload_jsonl_file(openai_client, mongodb_client, db_name, file_doc["jsonl_batch_id"])
                                    
                                    if new_file_id:
                                        await files_collection.update_one(
                                            {"_id": file_doc["_id"]},
                                            {"$set": {"openai_file_id": new_file_id, "uploaded_at": datetime.now()}}
                                        )
                                        await create_batch_request(openai_client, mongodb_client, db_name, new_file_id)
                            else:
                                logger.info(f"Batch request {batch_request['batch_id']} is in status: {batch_results.status}")
                        except Exception as e:
                            logger.error(f"Error processing batch request {batch_request['batch_id']}: {str(e)}")
                    else:
                        await create_batch_request(openai_client, mongodb_client, db_name, openai_file_id)
                except Exception as e:
                    logger.error(f"Error processing file {file_doc['_id']}: {str(e)}")

        tasks = []
        async for file_doc in files_collection.find():
            tasks.append(
                process_file_doc(file_doc, openai_client, mongodb_client, db_name, files_collection, batch_requests, completed_batches)
            )
        
        await asyncio.gather(*tasks)

        
        # async for file_doc in files_collection.find():
        #     if not file_doc.get("openai_file_id"):
        #         logger.info(f"File {file_doc['_id']} has no OpenAI file ID, skipping")
        #         continue
                
        #     openai_file_id = file_doc["openai_file_id"]
        #     is_expired = await check_file_expired(openai_client, openai_file_id)
            
        #     if is_expired:
        #         logger.info(f"File {openai_file_id} is expired, reuploading")
                
        #         # Reupload the file
        #         new_file_id = await reupload_jsonl_file(
        #             openai_client, 
        #             mongodb_client, 
        #             db_name, 
        #             file_doc["jsonl_batch_id"]
        #         )
                
        #         if new_file_id:
        #             # Update the file document with the new file ID
        #             await files_collection.update_one(
        #                 {"_id": file_doc["_id"]},
        #                 {"$set": {"openai_file_id": new_file_id, "uploaded_at": datetime.now()}}
        #             )
                    
        #             # Create a new batch request for the new file
        #             await create_batch_request(openai_client, mongodb_client, db_name, new_file_id)
                
        #         continue
            
        #     # Check for existing batch requests for this file
        #     batch_request = await batch_requests.find_one({"openai_file_id": openai_file_id})
        #     completed_batch = await completed_batches.find_one({"openai_file_id": openai_file_id})
            
        #     if completed_batch:
        #         # If completed batch exists and a batch request also exists, delete the batch request
        #         if batch_request:
        #             await batch_requests.delete_one({"_id": batch_request["_id"]})
        #             logger.info(f"Deleted batch request {batch_request['_id']} as it's already completed")
        #         continue
            
        #     if batch_request:
        #         # Check the status of the existing batch request
        #         try:
        #             batch_results = await openai_client.batches.retrieve(batch_request["batch_id"])
                    
        #             if batch_results.status == "completed":
        #                 # Get the file results
        #                 try:
        #                     file_response = await openai_client.files.content(batch_results.result_files[0].id)
                            
        #                     raw_print_text = file_response.text
        #                     response_list = [json.loads(x) for x in raw_print_text.split('\n') if x.strip()]
                            
        #                     batch_result_doc = {
        #                         "openai_file_id": openai_file_id,
        #                         "batch_id": batch_request["batch_id"],
        #                         "results": response_list,
        #                         "completed_at": datetime.now()
        #                     }
                            
        #                     await completed_batches.insert_one(batch_result_doc)
        #                     await batch_requests.delete_one({"_id": batch_request["_id"]})
        #                     logger.info(f"Batch request {batch_request['batch_id']} completed and saved")
                            
        #                 except Exception as e:
        #                     logger.error(f"Error retrieving batch results: {str(e)}")
                    
        #             elif batch_results.status == "failed":
        #                 # Handle failed batch
        #                 error_message = get(batch_results, "errors.data.0.message", "Unknown error")
        #                 logger.error(f"Batch {batch_request['batch_id']} failed: {error_message}")
                        
        #                 if "token limit" in error_message.lower():
        #                     # Delete the failed batch request and create a new one
        #                     await batch_requests.delete_one({"_id": batch_request["_id"]})
        #                     logger.info(f"Deleted rate-limited batch request {batch_request['batch_id']}")
                            
        #                     # Wait a bit before resubmitting
        #                     await asyncio.sleep(5)
        #                     await create_batch_request(openai_client, mongodb_client, db_name, openai_file_id)
                        
        #                 elif "expired" in error_message.lower():
        #                     # File expired during batch processing
        #                     await batch_requests.delete_one({"_id": batch_request["_id"]})
                            
        #                     # Reupload the file
        #                     new_file_id = await reupload_jsonl_file(
        #                         openai_client, 
        #                         mongodb_client, 
        #                         db_name, 
        #                         file_doc["jsonl_batch_id"]
        #                     )
                            
        #                     if new_file_id:
        #                         await files_collection.update_one(
        #                             {"_id": file_doc["_id"]},
        #                             {"$set": {"openai_file_id": new_file_id, "uploaded_at": datetime.now()}}
        #                         )
        #                         await create_batch_request(openai_client, mongodb_client, db_name, new_file_id)
                    
        #             else:
        #                 logger.info(f"Batch request {batch_request['batch_id']} is in status: {batch_results.status}")
                
        #         except Exception as e:
        #             logger.error(f"Error processing batch request {batch_request['batch_id']}: {str(e)}")
            
        #     else:
        #         # No existing batch request, create a new one
        #         await create_batch_request(openai_client, mongodb_client, db_name, openai_file_id)
    
    except Exception as e:
        logger.error(f"Error in file management: {str(e)}")

async def poll_and_process():
    """Continuously poll and process batch requests at regular intervals"""
    mongodb_uri = "mongodb://user:pass@localhost:27017"
    motor_client = AsyncIOMotorClient(mongodb_uri)
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    attempts = 0
    
    try:
        logger.info(f"Starting polling cycle every {POLLING_INTERVAL_SECONDS} seconds")
        
        while not SHUTDOWN_FLAG:
            if MAX_POLLING_ATTEMPTS and attempts >= MAX_POLLING_ATTEMPTS:
                logger.info(f"Reached maximum polling attempts ({MAX_POLLING_ATTEMPTS}), stopping")
                break
                
            logger.info(f"Polling cycle {attempts + 1} started")
            start_time = datetime.now()
            
            try:
                await handle_file_management(openai_client, motor_client, DB_NAME)
                logger.info("Completed file and batch management cycle")
            except Exception as e:
                logger.error(f"Error in polling cycle: {str(e)}")
            
            attempts += 1
            
            # Calculate time to sleep (ensure we don't have negative sleep time)
            elapsed = (datetime.now() - start_time).total_seconds()
            sleep_time = max(0, POLLING_INTERVAL_SECONDS - elapsed)
            
            if sleep_time > 0 and not SHUTDOWN_FLAG:
                logger.info(f"Sleeping for {sleep_time:.2f} seconds until next polling cycle")
                await asyncio.sleep(sleep_time)
            
    except asyncio.CancelledError:
        logger.info("Polling task cancelled")
    finally:
        logger.info("Polling cycle ended")

def handle_shutdown(sig=None, frame=None):
    """Handle shutdown signals gracefully"""
    global SHUTDOWN_FLAG
    if sig:
        logger.info(f"Received signal {sig}, initiating shutdown")
    SHUTDOWN_FLAG = True
    logger.info("Shutdown flag set, will exit after current cycle completes")

async def main():
    """Main entry point with signal handling for graceful shutdown"""
    get_openai_key()
    
    # Register signal handlers for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_shutdown)
    
    try:
        logger.info("Starting OpenAI file and batch management service")
        await poll_and_process()
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
    finally:
        logger.info("Service shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
