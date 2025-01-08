import argparse
import logging
import os
from pathlib import Path
from runi_thesis_project.config_loader import load_configs

logger = logging.getLogger(__name__)
def remove_dependent():
    pass
    
def main(args):
    config_dir = os.getenv("CONFIG_DIR",  str(Path.cwd() / "configs"))
    config = load_configs(config_dir)
    logger.info(f"Loading configs from {config_dir}")
    
    if args.sub_command == "train_model":
        logger.info(f"Running `train_model` with:\n"
                    f"Train input: {args.train_input}\n"
                    f"Train output directory: {args.train_output_dir}")        
    elif args.sub_command == "remove_dependent":
        logger.info(f"Running `remove_dependent` with:\n"
                    f"Input TSV: {args.input_tsv}\n"
                    f"Output TSV: {args.output_tsv}\n"
                    f"Column: {args.column}")
    elif args.sub_command == "run_negation_detection_model":
        from runi_thesis_project.models.negation_detection.client import create_model
        negation_model = create_model(**config["models.negation_detection"])
        logger.info(f"Running `run_negation_detection_model` with:\n"
                    f"Input TSV: {args.input_tsv}\n"
                    f"Output TSV: {args.output_tsv}\n"
                    f"Column: {args.column}")            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI for the research thesis different pipelines and experiments"
    )
    
    # Add subparsers to handle multiple commands
    subparsers = parser.add_subparsers(dest="sub_command")
    
    parser.add_argument(
        "--command", help="Command to run", required=False, type=str, choices=["remove_dependent"]
    )
    train_model_p = subparsers.add_parser(
        "train_model",
        help="Train a model on the input dataset",
    )
    train_model_p.add_argument(
        "--train_input", help="Input dataset to train the model on", required=False, type=str
    )
    train_model_p.add_argument(
        "--train_output_dir", help="Output directory to save the trained model", default="output", type=str
    )
    
    # Define the `remove_dependent` subcommand
    remove_dependent_claims_p = subparsers.add_parser(
        "remove_dependent",
        help="Detect and remove dependent patent claims from the input dataset",
    )
    remove_dependent_claims_p.add_argument(
        "--input_tsv", help="Input dataset to remove dependent claims from", required=True, type=str
    )
    remove_dependent_claims_p.add_argument(
        "--output_tsv", help="Output dataset to save the results to", default="output.tsv", type=str
    )
    remove_dependent_claims_p.add_argument(
        "--column", help="Column to remove dependent claims from", required=True, type=str
    )
    
    run_negation_detection_model_p = subparsers.add_parser(
        "run_negation_detection_model",
        help="Run the negation detection model on the input dataset",
    )
    run_negation_detection_model_p.add_argument(
        "--input_tsv", help="Input dataset to run the negation detection model on", required=True, type=str
    )
    run_negation_detection_model_p.add_argument(
        "--output_tsv", help="Output dataset to save the results to", default="output.tsv", type=str
    )
    run_negation_detection_model_p.add_argument(
        "--column", help="Column to run the negation detection model on", required=True, type=str
    )
    
    # Parse arguments
    args = parser.parse_args()
    main(args)
