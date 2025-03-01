import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from temporalio.client import Client
from workflow_concurrent import PatentNegationAnalysisConcurrentWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("concurrent_workflow.log"),
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Run the concurrent patent negation analysis workflow."""
    if len(sys.argv) < 2:
        print("Usage: python main_concurrent.py <csv_path> [batch_size] [concurrency_limit]")
        return
    
    # Parse command line arguments
    csv_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    concurrency_limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    # Connect to Temporal server
    client = await Client.connect("localhost:7233")
    logger.info(f"Connected to Temporal server")
    
    # Create a workflow ID based on the CSV filename and timestamp
    csv_filename = Path(csv_path).stem
    workflow_id = f"patent-negation-concurrent-{csv_filename}-{int(datetime.now().timestamp())}"
    
    # Start the workflow
    logger.info(f"Starting workflow {workflow_id} for {csv_path} with batch size {batch_size}")
    handle = await client.start_workflow(
        PatentNegationAnalysisConcurrentWorkflow.run,
        args=[csv_path, batch_size, concurrency_limit],
        id=workflow_id,
        task_queue="patent-negation-task-queue"
    )
    
    logger.info(f"Workflow started with ID: {workflow_id}")
    logger.info(f"Monitor progress at http://localhost:8080/namespaces/default/workflows/{workflow_id}")
    
    # Wait for workflow to complete
    result = await handle.result()
    logger.info(f"Workflow completed with status: {result['status']}")
    
    if result['status'] == 'completed':
        logger.info(f"Processed {result['total_processed']} records with {result['successful_results']} successes and {result['error_results']} errors")
    else:
        logger.error(f"Workflow failed: {result.get('error', 'unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())
