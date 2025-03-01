import logging
from datetime import timedelta
from typing import List, Dict, Any, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

# Import activities module but don't use activities directly in the workflow
# This prevents sandbox issues
import activities
from models import FileMetadata, BatchRequest, ProcessingResult

logger = logging.getLogger(__name__)

@workflow.defn(sandboxed=False)  # Disable sandboxing for simplicity while developing
class PatentNegationAnalysisWorkflow:
    """Main workflow for patent negation analysis using OpenAI batch processing."""
    
    @workflow.run
    async def run(self, csv_path: str, batch_size: int = 1000) -> Dict[str, Any]:
        """Execute the end-to-end patent negation analysis workflow."""
        workflow.logger.info(f"Starting patent negation analysis workflow for {csv_path}")
        
        # Activity retry policies
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
        except Exception as e:
            workflow.logger.error(f"Failed to load CSV data: {str(e)}")
            raise ApplicationError(f"CSV loading failed: {str(e)}")

        # 3. Prepare JSONL files for OpenAI batch processing
        try:
            jsonl_batch_ids = await workflow.execute_activity(
                activities.prepare_jsonl_files,
                args=[df_info['dataframe_pickle'], batch_size],  # Pass args as a list
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=standard_retry
            )
            workflow.logger.info(f"Created {len(jsonl_batch_ids)} JSONL batches in MongoDB")
        except Exception as e:
            workflow.logger.error(f"Failed to prepare JSONL batches: {str(e)}")
            raise ApplicationError(f"JSONL preparation failed: {str(e)}")
            
        # 4. Register batches in MongoDB and upload to OpenAI
        file_metadatas = []
        
        # Register all batches in MongoDB as files
        for jsonl_batch_id in jsonl_batch_ids:
            try:
                file_metadata = await workflow.execute_activity(
                    activities.register_file_in_mongodb,
                    jsonl_batch_id,
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=RetryPolicy(maximum_attempts=3),
                )
                file_metadatas.append(file_metadata)
            except Exception as e:
                workflow.logger.error(f"Failed to process batch {jsonl_batch_id}: {str(e)}")
        
        # Upload files to OpenAI concurrently
        try:
            uploaded_files = await workflow.execute_activity(
                activities.upload_files_to_openai,  # New concurrent upload activity
                file_metadatas,
                api_key,
                5,  # Max 5 concurrent uploads
                start_to_close_timeout=timedelta(hours=2),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            
            workflow.logger.info(f"Successfully uploaded {len(uploaded_files)} files to OpenAI")
        except Exception as e:
            workflow.logger.error(f"Failed to upload files: {str(e)}")
            return {"status": "failed", "error": str(e)}
        
        # Continue with the rest of your workflow...
        
        return {
            "status": "completed",
            "csv_path": csv_path,
            "batch_size": batch_size,
            "row_count": df_info["row_count"],
            "batch_count": len(jsonl_batch_ids),
            "processed_count": len(uploaded_files),
        }
