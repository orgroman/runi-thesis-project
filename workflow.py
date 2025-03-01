import logging
import asyncio
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
            
        # 4. OPTIMIZATION: Check if files already exist in MongoDB and are uploaded to OpenAI
        try:
            all_uploaded, file_metadatas = await workflow.execute_activity(
                activities.check_existing_openai_files,
                jsonl_batch_ids,
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=standard_retry
            )
            
            if all_uploaded and file_metadatas:
                workflow.logger.info(f"All {len(file_metadatas)} files are already registered and uploaded to OpenAI")
                # Skip registration and upload steps
            else:
                # If not all files exist, register them
                file_metadatas = []
                for i, jsonl_batch_id in enumerate(jsonl_batch_ids):
                    try:
                        workflow.logger.info(f"Registering file {i+1}/{len(jsonl_batch_ids)}")
                        file_metadata = await workflow.execute_activity(
                            activities.register_file_in_mongodb,
                            jsonl_batch_id,
                            start_to_close_timeout=timedelta(minutes=5),
                            retry_policy=standard_retry
                        )
                        file_metadatas.append(file_metadata)
                    except Exception as e:
                        workflow.logger.error(f"Failed to register file: {str(e)}")
                
                workflow.logger.info(f"Registered {len(file_metadatas)} files")
                
                # Upload files to OpenAI
                uploaded_files = await workflow.execute_activity(
                    activities.upload_files_to_openai,
                    file_metadatas,
                    api_key,
                    start_to_close_timeout=timedelta(minutes=60),
                    retry_policy=standard_retry
                )
                
                workflow.logger.info(f"Uploaded {len(uploaded_files)} files to OpenAI")
                
                # Update file_metadatas with OpenAI file IDs
                file_metadatas = uploaded_files
        
        except Exception as e:
            workflow.logger.error(f"Failed to check or process files: {str(e)}")
            workflow.logger.info("Falling back to standard registration and upload")
            
            # Continue with default file registration and upload
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