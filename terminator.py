"""
Utility script to terminate a running workflow.
Use this when a workflow gets stuck or you need to restart it.
"""
import asyncio
import logging
from temporalio.client import Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

async def terminate_workflow(workflow_id: str, reason: str = "Manual termination"):
    """Terminate a running workflow execution."""
    try:
        # Connect to Temporal server
        client = await Client.connect("localhost:7233")
        logger.info(f"Connected to Temporal server")
        
        # Get workflow handle
        handle = client.get_workflow_handle(workflow_id)
        
        # Terminate workflow
        await handle.terminate(reason)
        logger.info(f"Successfully terminated workflow {workflow_id}: {reason}")
        
    except Exception as e:
        logger.error(f"Failed to terminate workflow: {str(e)}")

async def main():
    """Terminate the patent negation analysis workflow."""
    await terminate_workflow(
        "patent-negation-analysis", 
        "Restarting workflow after fixing code issues"
    )

if __name__ == "__main__":
    asyncio.run(main())
