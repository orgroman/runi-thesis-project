import logging
import asyncio
from datetime import timedelta, datetime
from typing import List, Dict, Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

logger = logging.getLogger(__name__)

@workflow.defn(sandboxed=False)
class PatentNegationAnalysisEfficientWorkflow:
    """
    Efficient workflow that avoids redundant work by checking existing data in MongoDB.
    """
    
    @workflow.run
    async def run(self, csv_path: str, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Execute the patent negation analysis workflow with optimization for existing data.
        
        Args:
            csv_path: Path to CSV file with patent data
            batch_size: Number of records per batch
            
        Returns:
            Dict with workflow execution results
        """
        workflow.logger.info(f"Starting efficient workflow for {csv_path}")
        
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
            "batch_size": batch_size,
            "start_time": datetime.now().isoformat()
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

        # 3. Prepare JSONL batches if they don't already exist
        try:
            jsonl_batch_ids = await workflow.execute_activity(
                activities.prepare_jsonl_files,
                args=[df_info['dataframe_pickle'], batch_size],
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=standard_retry
            )
            workflow.logger.info(f"Found/created {len(jsonl_batch_ids)} JSONL batches")
            stats["batch_count"] = len(jsonl_batch_ids)
        except Exception as e:
            workflow.logger.error(f"Failed to prepare JSONL batches: {str(e)}")
            stats["status"] = "failed"
            stats["error"] = f"JSONL preparation failed: {str(e)}"
            return stats
            
        # 4. Check if files already exist in MongoDB and are uploaded to OpenAI
        try:
            all_uploaded, file_metadatas = await workflow.execute_activity(
                activities.check_existing_openai_files,
                jsonl_batch_ids,
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=standard_retry
            )
            
            if all_uploaded and file_metadatas:
                workflow.logger.info(f"All {len(file_metadatas)} files are already registered and uploaded to OpenAI")
            else:
                # If files aren't all registered or uploaded, do it now
                # 4a. Register files in MongoDB if needed
                if len(file_metadatas) < len(jsonl_batch_ids):
                    workflow.logger.info("Some files need to be registered in MongoDB")
                    
                    # Get IDs of batches that need registration
                    existing_batch_ids = [fm.jsonl_batch_id for fm in file_metadatas]
                    batches_to_register = [bid for bid in jsonl_batch_ids if bid not in existing_batch_ids]
                    
                    # Register files in batches with controlled concurrency
                    concurrency_limit = 5
                    for i in range(0, len(batches_to_register), concurrency_limit):
                        batch = batches_to_register[i:i+concurrency_limit]
                        register_tasks = []
                        
                        for batch_id in batch:
                            task = workflow.execute_activity(
                                activities.register_file_in_mongodb,
                                batch_id,
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
                
                # 4b. Upload files to OpenAI if needed
                files_to_upload = [fm for fm in file_metadatas if not fm.openai_file_id or fm.status != "uploaded"]
                
                if files_to_upload:
                    workflow.logger.info(f"{len(files_to_upload)} files need to be uploaded to OpenAI")
                    
                    # Upload files in batches
                    concurrency_limit = 5
                    uploaded_files = []
                    
                    for i in range(0, len(files_to_upload), concurrency_limit):
                        batch = files_to_upload[i:i+concurrency_limit]
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
                    
                    # Update file_metadatas with newly uploaded files
                    workflow.logger.info(f"Uploaded {len(uploaded_files)} files to OpenAI")
                    
                    # Replace file metadata entries with updated versions that have OpenAI file IDs
                    for uploaded_file in uploaded_files:
                        for i, fm in enumerate(file_metadatas):
                            if fm.mongodb_id == uploaded_file.mongodb_id:
                                file_metadatas[i] = uploaded_file
                
        except Exception as e:
            workflow.logger.error(f"Failed to check or process existing files: {str(e)}")
            stats["status"] = "failed"
            stats["error"] = f"File processing failed: {str(e)}"
            return stats
        
        # 5. Submit batch requests to OpenAI (only for files that don't already have batch requests)
        try:
            # TODO: Check if batch requests already exist for these files
            
            # For now, submit batch requests for all files
            batch_requests = []
            concurrency_limit = 5
            
            for i in range(0, len(file_metadatas), concurrency_limit):
                batch = file_metadatas[i:i+concurrency_limit]
                submit_tasks = []
                
                for file_metadata in batch:
                    task = workflow.execute_activity(
                        activities.submit_batch_request,
                        args=[file_metadata, api_key],
                        start_to_close_timeout=timedelta(minutes=10),
                        retry_policy=standard_retry
                    )
                    submit_tasks.append(task)
                
                batch_results = await asyncio.gather(*submit_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        workflow.logger.error(f"Batch submission failed: {str(result)}")
                    else:
                        batch_requests.append(result)
            
            workflow.logger.info(f"Submitted {len(batch_requests)} batch requests")
            stats["batches_submitted"] = len(batch_requests)
            
        except Exception as e:
            workflow.logger.error(f"Failed to submit batch requests: {str(e)}")
            stats["status"] = "failed"
            stats["error"] = f"Batch submission failed: {str(e)}"
            return stats
        
        # Continue with monitoring and result processing...
        
        stats["status"] = "completed"
        stats["end_time"] = datetime.now().isoformat()
        return stats
