import argparse
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def deploy_dags(source_dir, target_dir=None, backup=True):
    """
    Deploy DAG files to Airflow dags directory.
    
    Args:
        source_dir: Directory containing DAG files
        target_dir: Airflow dags directory (default: /c:/Users/orgrd/workspace/airflow-docker/dags)
        backup: Whether to backup existing files before overwriting
    """
    # Default target directory if not specified
    if target_dir is None:
        target_dir = Path(r"C:\Users\orgrd\workspace\airflow_dags")
    else:
        target_dir = Path(target_dir)
        
    # source_dir = Path(source_dir)
    source_dir = Path(__file__).parent.parent / "dags"
    
    # Ensure directories exist
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    
    if not target_dir.exists():
        logger.info(f"Creating target directory: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create backup directory if needed
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = target_dir.parent / f"dags_backup_{timestamp}"
        logger.info(f"Creating backup of current DAGs in {backup_dir}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy existing files to backup
        for file in target_dir.glob("*.py"):
            shutil.copy2(file, backup_dir / file.name)
    
    # Copy DAG files to target directory
    count = 0
    for file in source_dir.glob("*.py"):
        if "dag" in file.name.lower():  # Only copy files that likely contain DAGs
            logger.info(f"Copying {file.name} to {target_dir}")
            shutil.copy2(file, target_dir / file.name)
            count += 1
    
    logger.info(f"Deployed {count} DAG files to {target_dir}")
    
    # Optionally touch the __init__.py file to ensure the directory is a package
    init_file = target_dir / "__init__.py"
    if not init_file.exists():
        logger.info(f"Creating {init_file}")
        init_file.touch()
    
    return count

def main():
    parser = argparse.ArgumentParser(description="Deploy DAG files to Airflow")
    parser.add_argument("--source-dir", default="/c:/Users/orgrd/workspace/repos/runi-thesis-project/dags",
                        help="Directory containing DAG files")
    parser.add_argument("--target-dir", default=r"C:\Users\orgrd\workspace\airflow_dags",
                        help="Airflow dags directory")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip backup of existing DAG files")
    
    args = parser.parse_args()
    
    try:
        count = deploy_dags(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            backup=not args.no_backup
        )
        
        print(f"\n✓ Successfully deployed {count} DAG files")
        print(f"  From: {args.source_dir}")
        print(f"  To:   {args.target_dir}")
        print("\nNext steps:")
        print("1. Refresh the Airflow UI to see your new DAGs")
        print("2. If DAGs are not showing up, check logs for syntax errors")
        print("3. Use 'check_dag_syntax.py' to validate your DAGs")
        
    except Exception as e:
        logger.error(f"Error deploying DAGs: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
