import logging
import asyncio
from datetime import timedelta
from typing import List, Dict, Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

# Import activities module outside the workflow sandbox
# Activities will be accessible through execute_activity calls only
logger = logging.getLogger(__name__)

@workflow.defn(sandboxed=False)  # Disable sandbox to avoid import restrictions
class PatentNegationAnalysisOptimizedWorkflow:
    """
    Optimized workflow that avoids unnecessary database operations by checking
    if files are already registered before processing.
    """
    
    @workflow.run
    async def run(self, csv_path: str, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Execute the patent negation analysis workflow with optimized MongoDB operations.
        
        Args:
            csv_path: Path to CSV file with patent data
            batch_size: Number of records per batch
            
        Returns:
            Dict with workflow execution results
        """
        workflow.logger.info(f"Starting optimized workflow for {csv_path}")
        
        # Import activities inside the method to avoid sandbox issues
        import activities
        
        # Common retry policies
        standard_retry = RetryPolicy(
            maximum_attempts=3,
            initial_interval=timedelta(seconds=5),
            maximum_interval=timedelta(minutes=10)
        )
        
        # Track statistics
        stats = {
            "status": "in_progress",
            "csv_path": csv_path,
            "batch_size": batch_size
        }
        
        # 1. Authenticate with Azure and get OpenAI API Key
        try:
            api_key = await workflow.execute_activity(
                activities.authenticate_with_azure,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=standard_retry
            )
            workflow.logger.info("Successfully authenticated with Azure Key Vault")
        except Exception as e:
            workflow.logger.error(f"Authentication failed: {str(e)}")
            stats["status"] = "failed"
            stats["error"] = f"Authentication failed: {str(e)}"
            return stats

        # 2. Load and validate patent CSV data
        try:
            df_info = await workflow.execute_activity(
                activities.load_patent_data,
                csv_path,
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=standard_retry
            )
            workflow.logger.info(f"Loaded {df_info['row_count']} records from CSV")
            stats["row_count"] = df_info["row_count"]
        except Exception as e:
            workflow.logger.error(f"Failed to load CSV data: {str(e)}")
            stats["status"] = "failed"
            stats["error"] = f"CSV loading failed: {str(e)}"
            return stats

        # 3. Check if all files exist for this dataframe
        try:
            all_exist, file_ids = await workflow.execute_activity(
                activities.check_files_exist_for_dataframe,
                df_info["dataframe_pickle"],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=standard_retry
            )
            
            if all_exist:
                workflow.logger.info(f"All {len(file_ids)} files already exist for this dataframe")
                
                # Load all file metadata
                file_metadatas = []
                for file_id in file_ids:
                    try:
                        file_metadata = await workflow.execute_activity(
                            activities.get_file_metadata_by_id,
                            file_id,
                            start_to_close_timeout=timedelta(minutes=1),
                            retry_policy=standard_retry
                        )
                        file_metadatas.append(file_metadata)
                    except Exception as e:
                        workflow.logger.error(f"Failed to load file metadata: {str(e)}")
                
                workflow.logger.info(f"Loaded {len(file_metadatas)} file metadata objects")
                
            else:
                # Continue with JSONL preparation and file registration
                workflow.logger.info("Not all files exist, continuing with JSONL preparation")
                
                # Prepare JSONL files
                jsonl_batch_ids = await workflow.execute_activity(
                    activities.prepare_jsonl_files,
                    args=[df_info['dataframe_pickle'], batch_size],
                    start_to_close_timeout=timedelta(minutes=30),
                    retry_policy=standard_retry
                )
                workflow.logger.info(f"Created/found {len(jsonl_batch_ids)} JSONL batches")
                
                # Register files
                file_metadatas = []
                concurrency_limit = 5
                
                for i in range(0, len(jsonl_batch_ids), concurrency_limit):
                    batch = jsonl_batch_ids[i:i+concurrency_limit]
                    workflow.logger.info(f"Registering batch {i//concurrency_limit + 1}/{len(jsonl_batch_ids)//concurrency_limit + 1}")
                    
                    register_tasks = []
                    for jsonl_batch_id in batch:
                        task = workflow.execute_activity(
                            activities.register_file_in_mongodb,
                            jsonl_batch_id,
                            start_to_close_timeout=timedelta(minutes=5),
                            retry_policy=standard_retry
                        )
                        register_tasks.append(task)
                    
                    batch_results = await asyncio.gather(*register_tasks, return_exceptions=True)
                    
                    for result in batch_results:
                        if isinstance(result, Exception):
                            workflow.logger.error(f"Registration failed: {str(result)}")
                        else:
                            file_metadatas.append(result)
                
                workflow.logger.info(f"Registered {len(file_metadatas)} files")
            
            # Upload files
            uploaded_files = []
            concurrency_limit = 5
            
            for i in range(0, len(file_metadatas), concurrency_limit):
                batch = file_metadatas[i:i+concurrency_limit]
                workflow.logger.info(f"Uploading batch {i//concurrency_limit + 1}/{len(file_metadatas)//concurrency_limit + 1}")
                
                upload_tasks = []
                for file_metadata in batch:
                    task = workflow.execute_activity(
                        activities.upload_file_to_openai,
                        args=[file_metadata, api_key],
                        start_to_close_timeout=timedelta(minutes=10),
                        retry_policy=standard_retry
                    )
                    upload_tasks.append(task)
                
                batch_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        workflow.logger.error(f"Upload failed: {str(result)}")
                    else:
                        uploaded_files.append(result)
            
            workflow.logger.info(f"Uploaded {len(uploaded_files)}/{len(file_metadatas)} files")
            stats["files_uploaded"] = len(uploaded_files)
            
        except Exception as e:
            workflow.logger.error(f"Error in workflow: {str(e)}")
            stats["status"] = "failed"
            stats["error"] = str(e)
            return stats
        
        stats["status"] = "completed"
        return stats
