import logging
import asyncio
from datetime import timedelta, datetime
from typing import List, Dict, Any, Optional, Tuple

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

# Import activities module
import activities
from models import FileMetadata, BatchRequest, ProcessingResult

logger = logging.getLogger(__name__)

@workflow.defn
class PatentNegationAnalysisConcurrentWorkflow:
    """
    Enhanced workflow for patent negation analysis using native Temporal concurrency.
    This workflow uses Temporal's native concurrency to parallelize operations for better
    visibility, reliability, and performance.
    """
    
    @workflow.run
    async def run(self, csv_path: str, batch_size: int = 1000, concurrency_limit: int = 5) -> Dict[str, Any]:
        """
        Execute the end-to-end patent negation analysis workflow with native concurrency.
        
        Args:
            csv_path: Path to input CSV file
            batch_size: Number of records per batch
            concurrency_limit: Maximum number of concurrent operations (rate limiting)
            
        Returns:
            Dict with workflow execution results
        """
        workflow.logger.info(f"Starting concurrent patent negation analysis workflow for {csv_path}")
        
        # Common retry policies
        standard_retry = RetryPolicy(
            maximum_attempts=3,
            initial_interval=timedelta(seconds=5),
            maximum_interval=timedelta(minutes=10)
        )
        
        long_retry = RetryPolicy(
            maximum_attempts=10,
            initial_interval=timedelta(seconds=30),
            maximum_interval=timedelta(minutes=30)
        )
        
        # Track statistics
        stats = {
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
            "csv_path": csv_path,
            "batch_size": batch_size,
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
            raise ApplicationError(f"Authentication failed: {str(e)}")

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

        # 3. Prepare JSONL files for OpenAI batch processing
        try:
            jsonl_batch_ids = await workflow.execute_activity(
                activities.prepare_jsonl_files,
                args=[df_info['dataframe_pickle'], batch_size],
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=standard_retry
            )
            workflow.logger.info(f"Created {len(jsonl_batch_ids)} JSONL batches in MongoDB")
            stats["batch_count"] = len(jsonl_batch_ids)
        except Exception as e:
            workflow.logger.error(f"Failed to prepare JSONL batches: {str(e)}")
            stats["status"] = "failed"
            stats["error"] = f"JSONL preparation failed: {str(e)}"
            return stats
            
        # 4. Register batches in MongoDB as files (CONCURRENTLY)
        file_metadatas = []
        
        # Register all batches concurrently with rate limiting
        for i in range(0, len(jsonl_batch_ids), concurrency_limit):
            batch = jsonl_batch_ids[i:i+concurrency_limit]
            workflow.logger.info(f"Registering batch {i//concurrency_limit + 1} of {len(jsonl_batch_ids)//concurrency_limit + 1}")
            
            # Create tasks for concurrent registration
            register_tasks = []
            for jsonl_batch_id in batch:
                task = workflow.execute_activity(
                    activities.register_file_in_mongodb,
                    jsonl_batch_id,
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=standard_retry
                )
                register_tasks.append(task)
            
            # Wait for all registrations in this batch to complete
            batch_results = await asyncio.gather(*register_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    workflow.logger.error(f"Registration failed: {str(result)}")
                else:
                    file_metadatas.append(result)
        
        workflow.logger.info(f"Registered {len(file_metadatas)} of {len(jsonl_batch_ids)} JSONL batches")
        
        # 5. Upload files to OpenAI (CONCURRENTLY)
        uploaded_files = []
        
        # Upload files concurrently with rate limiting
        for i in range(0, len(file_metadatas), concurrency_limit):
            batch = file_metadatas[i:i+concurrency_limit]
            workflow.logger.info(f"Uploading batch {i//concurrency_limit + 1} of {len(file_metadatas)//concurrency_limit + 1}")
            
            # Create tasks for concurrent uploads
            upload_tasks = []
            for file_metadata in batch:
                task = workflow.execute_activity(
                    activities.upload_file_to_openai,
                    args=[file_metadata, api_key],
                    start_to_close_timeout=timedelta(minutes=20),
                    retry_policy=standard_retry
                )
                upload_tasks.append(task)
            
            # Wait for all uploads in this batch to complete
            batch_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    workflow.logger.error(f"Upload failed: {str(result)}")
                else:
                    uploaded_files.append(result)
        
        workflow.logger.info(f"Uploaded {len(uploaded_files)} of {len(file_metadatas)} files")
        stats["files_uploaded"] = len(uploaded_files)
        
        # 6. Submit batch requests to OpenAI (CONCURRENTLY)
        batch_requests = []
        
        # Submit batch requests concurrently with rate limiting
        for i in range(0, len(uploaded_files), concurrency_limit):
            batch = uploaded_files[i:i+concurrency_limit]
            workflow.logger.info(f"Submitting batch {i//concurrency_limit + 1} of {len(uploaded_files)//concurrency_limit + 1}")
            
            # Create tasks for concurrent batch submission
            submit_tasks = []
            for file in batch:
                task = workflow.execute_activity(
                    activities.submit_batch_request,
                    args=[file, api_key],
                    start_to_close_timeout=timedelta(minutes=10),
                    retry_policy=standard_retry
                )
                submit_tasks.append(task)
            
            # Wait for all submissions in this batch to complete
            batch_results = await asyncio.gather(*submit_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    workflow.logger.error(f"Batch submission failed: {str(result)}")
                else:
                    batch_requests.append(result)
        
        workflow.logger.info(f"Submitted {len(batch_requests)} batch requests")
        stats["batches_submitted"] = len(batch_requests)
        
        # 7. Monitor batch requests until completion (CONCURRENTLY)
        completed_batches = []
        
        # Monitor batches concurrently
        monitor_tasks = []
        for batch_request in batch_requests:
            task = workflow.execute_activity(
                activities.monitor_batch_status,
                args=[batch_request, api_key],
                start_to_close_timeout=timedelta(hours=24),  # Long timeout for monitoring
                heartbeat_timeout=timedelta(minutes=10),
                retry_policy=long_retry
            )
            monitor_tasks.append(task)
        
        # Wait for all monitoring tasks to complete
        monitoring_results = await asyncio.gather(*monitor_tasks, return_exceptions=True)
        
        # Process results
        for result in monitoring_results:
            if isinstance(result, Exception):
                workflow.logger.error(f"Batch monitoring failed: {str(result)}")
            elif result.status == "completed":
                completed_batches.append(result)
            else:
                workflow.logger.warning(f"Batch {result.batch_id} ended with status: {result.status}")
        
        workflow.logger.info(f"Completed {len(completed_batches)} of {len(batch_requests)} batches")
        stats["batches_completed"] = len(completed_batches)
        
        # 8. Process batch results (CONCURRENTLY)
        processing_results = []
        
        # Process results concurrently with rate limiting
        for i in range(0, len(completed_batches), concurrency_limit):
            batch = completed_batches[i:i+concurrency_limit]
            workflow.logger.info(f"Processing result batch {i//concurrency_limit + 1} of {len(completed_batches)//concurrency_limit + 1}")
            
            # Create tasks for concurrent result processing
            process_tasks = []
            for completed_batch in batch:
                task = workflow.execute_activity(
                    activities.process_batch_results,
                    args=[completed_batch, api_key],
                    start_to_close_timeout=timedelta(hours=2),
                    retry_policy=standard_retry
                )
                process_tasks.append(task)
            
            # Wait for all processing tasks in this batch to complete
            batch_results = await asyncio.gather(*process_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    workflow.logger.error(f"Result processing failed: {str(result)}")
                else:
                    processing_results.append(result)
        
        # Update final statistics
        successful_results = sum(result.success_count for result in processing_results)
        error_results = sum(result.error_count for result in processing_results)
        
        workflow.logger.info(f"Processed {successful_results} successful results and {error_results} error results")
        
        stats.update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "successful_results": successful_results,
            "error_results": error_results,
            "total_processed": successful_results + error_results,
        })
        
        return stats
