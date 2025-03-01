import asyncio
import logging
import os
import sys
import signal
import subprocess
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("worker_restart.log"),
    ]
)

logger = logging.getLogger(__name__)

def restart_worker():
    """Force stop any running workers and start a fresh one."""
    logger.info("Restarting worker process...")
    
    # Kill any existing worker processes
    if os.name == 'nt':  # Windows
        try:
            # Find python processes running worker scripts
            result = subprocess.run(
                'tasklist /fi "imagename eq python.exe" /fo csv /nh',
                capture_output=True,
                text=True,
                shell=True
            )
            
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if '.venv\\Scripts\\python.exe' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        pid = parts[1].strip('"')
                        try:
                            pid = int(pid)
                            logger.info(f"Killing existing Python worker process with PID {pid}")
                            os.kill(pid, signal.SIGTERM)
                        except (ValueError, ProcessLookupError) as e:
                            logger.error(f"Error killing process: {str(e)}")
        except Exception as e:
            logger.error(f"Error finding/killing existing processes: {str(e)}")
    else:  # Unix/Linux/Mac
        try:
            os.system("pkill -f 'python.*worker'")
        except Exception as e:
            logger.error(f"Error killing existing processes: {str(e)}")
    
    # Give processes time to terminate
    time.sleep(2)
    
    # Start new worker in a new process
    logger.info("Starting new worker process...")
    
    python_executable = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
    worker_script = os.path.join(os.getcwd(), "restart_worker.py")
    
    if not os.path.exists(python_executable):
        logger.warning(f"Python executable not found at {python_executable}, using system python")
        python_executable = "python"
    
    # Start the worker in a new process, detached from this script
    if os.name == 'nt':  # Windows
        subprocess.Popen(
            [python_executable, worker_script],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:  # Unix/Linux/Mac
        subprocess.Popen(
            [python_executable, worker_script],
            start_new_session=True
        )
    
    logger.info("New worker process started")
    logger.info("Waiting 5 seconds for worker to initialize...")
    time.sleep(5)
    logger.info("Worker restart completed")

if __name__ == "__main__":
    restart_worker()
    print("Worker restarted. You can now run your workflow.")
