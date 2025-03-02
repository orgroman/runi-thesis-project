"""
Service script for running the OpenAI batch management process continuously.
This can be deployed as a background service or container.
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI

import config
from retr_batch_openai import get_openai_key, handle_file_management

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
SHUTDOWN_FLAG = False

def handle_shutdown(sig=None, frame=None):
    """Handle shutdown signals gracefully"""
    global SHUTDOWN_FLAG
    if sig:
        logger.info(f"Received signal {sig}, initiating shutdown")
    SHUTDOWN_FLAG = True
    logger.info("Shutdown flag set, will exit after current cycle completes")

async def poll_and_process():
    """Continuously poll and process batch requests at regular intervals"""
    mongodb_client = AsyncIOMotorClient(config.MONGODB_URI)
    openai_client = AsyncOpenAI(api_key=get_openai_key())
    
    attempts = 0
    
    try:
        logger.info(f"Starting polling cycle every {config.POLLING_INTERVAL_SECONDS} seconds")
        
        while not SHUTDOWN_FLAG:
            if config.MAX_POLLING_ATTEMPTS and attempts >= config.MAX_POLLING_ATTEMPTS:
                logger.info(f"Reached maximum polling attempts ({config.MAX_POLLING_ATTEMPTS}), stopping")
                break
                
            logger.info(f"Polling cycle {attempts + 1} started")
            start_time = datetime.now()
            
            try:
                await handle_file_management(openai_client, mongodb_client, config.DB_NAME)
                logger.info("Completed file and batch management cycle")
            except Exception as e:
                logger.error(f"Error in polling cycle: {str(e)}")
            
            attempts += 1
            
            # Calculate time to sleep (ensure we don't have negative sleep time)
            elapsed = (datetime.now() - start_time).total_seconds()
            sleep_time = max(0, config.POLLING_INTERVAL_SECONDS - elapsed)
            
            if sleep_time > 0 and not SHUTDOWN_FLAG:
                logger.info(f"Sleeping for {sleep_time:.2f} seconds until next polling cycle")
                await asyncio.sleep(sleep_time)
            
    except asyncio.CancelledError:
        logger.info("Polling task cancelled")
    finally:
        logger.info("Polling cycle ended")

async def main():
    """Main entry point with signal handling for graceful shutdown"""
    # Register signal handlers for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_shutdown)
    
    try:
        logger.info("Starting OpenAI file and batch management service")
        await poll_and_process()
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        return 1
    finally:
        logger.info("Service shutdown complete")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
