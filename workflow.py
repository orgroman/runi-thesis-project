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
            
            # Start with initial steps only to verify everything is working
            return {
                "status": "workflow_started",
                "message": "Authentication successful",
                "api_key": "[REDACTED]" # Never log the actual key
            }
            
        except Exception as e:
            workflow.logger.error(f"Authentication failed: {str(e)}")
            raise ApplicationError("Authentication failed", details={"error": str(e)})
