import asyncio
import logging
import argparse
from datetime import timedelta

from temporalio.client import Client
from temporalio.common import RetryPolicy

from workflow import PatentNegationAnalysisWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("patent_workflow_starter.log"),
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Start the Patent Negation Analysis workflow."""
    
    # Parse command line arguments
    csv_path = r'C:\Users\orgrd\workspace\data\patentmatch_test\patentmatch_test_no_claims.csv'
    parser = argparse.ArgumentParser(description="Start the Patent Negation Analysis workflow")
    #parser.add_argument("--csv", required=True, help="Path to patent CSV file")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of records per batch")
    args = parser.parse_args()
    
    # Connect to Temporal server
    client = await Client.connect("localhost:7233")
    logger.info(f"Connected to Temporal server")
    
    # Start the workflow
    logger.info(f"Starting workflow for CSV: {csv_path} with batch size: {args.batch_size}")
    
    # Create a proper RetryPolicy object
    retry_policy = RetryPolicy(
        maximum_attempts=3,
        initial_interval=timedelta(seconds=1),
        maximum_interval=timedelta(minutes=1),
    )
    
    handle = await client.start_workflow(
        PatentNegationAnalysisWorkflow.run,
        args=[csv_path, args.batch_size],
        id="patent-negation-analysis",
        task_queue="patent-negation-task-queue",
        execution_timeout=timedelta(hours=72),  # 3 days maximum execution time
        retry_policy=retry_policy
    )
    
    logger.info(f"Workflow started with ID: {handle.id}")
    logger.info("Waiting for workflow to complete...")
    
    # Wait for workflow completion
    result = await handle.result()
    logger.info(f"Workflow completed with result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
