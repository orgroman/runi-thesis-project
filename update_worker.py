import asyncio
import logging
import subprocess
import sys

from temporalio.client import Client
from temporalio.worker import Worker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

async def update_worker():
    """Update and restart the worker with all workflows."""
    try:
        logger.info("Updating imports...")
        
        # Import all activities, updated file operation modules, and workflows
        import activities
        
        # Dynamically import our workflows
        logger.info("Importing workflows...")
        from workflow import PatentNegationAnalysisWorkflow
        
        # Import optimized workflow
        logger.info("Importing optimized workflow...")
        try:
            from workflow_optimized import PatentNegationAnalysisOptimizedWorkflow
            logger.info("Successfully imported optimized workflow.")
        except Exception as e:
            logger.error(f"Failed to import optimized workflow: {str(e)}")
            return

        # Connect to Temporal
        logger.info("Connecting to Temporal...")
        client = await Client.connect("localhost:7233")

        # Create worker with all workflows
        logger.info("Starting worker with all workflows...")
        worker = Worker(
            client,
            task_queue="patent-negation-task-queue",
            workflows=[
                PatentNegationAnalysisWorkflow,         # Original workflow
                PatentNegationAnalysisOptimizedWorkflow # Optimized workflow
            ],
            activities=activities
        )
        
        # Print available activities for verification
        logger.info("Available activities:")
        for name in dir(activities):
            if not name.startswith("_") and callable(getattr(activities, name)):
                logger.info(f"  - {name}")
        
        # Print registered workflows
        logger.info("Registered workflows:")
        logger.info("  - PatentNegationAnalysisWorkflow")
        logger.info("  - PatentNegationAnalysisOptimizedWorkflow")
        
        # Run worker
        logger.info("Starting worker...")
        await worker.run()
        
    except Exception as e:
        logger.error(f"Error updating worker: {str(e)}")

def launch_optimized_workflow():
    """Launch the optimized workflow in a new terminal."""
    try:
        logger.info("Launching optimized workflow...")
        
        if len(sys.argv) < 2:
            csv_path = input("Enter CSV path: ")
        else:
            csv_path = sys.argv[1]
            
        # Use the platform-specific command to open a new terminal
        if sys.platform == "win32":
            cmd = [
                "start", "cmd", "/k",
                f"python run_optimized.py \"{csv_path}\""
            ]
            subprocess.Popen(" ".join(cmd), shell=True)
        else:
            # Linux/Mac
            cmd = [
                "gnome-terminal", "--", "bash", "-c",
                f"python run_optimized.py \"{csv_path}\"; read -p 'Press enter to close'"
            ]
            subprocess.Popen(cmd)
            
        logger.info("Optimized workflow launched in a new terminal.")
            
    except Exception as e:
        logger.error(f"Error launching optimized workflow: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "launch":
        # Launch the optimized workflow
        launch_optimized_workflow()
    else:
        # Update and run the worker
        asyncio.run(update_worker())
