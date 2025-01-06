import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI for the research thesis different pipelines and experiments"
    )
    
    # Add subparsers to handle multiple commands
    subparsers = parser.add_subparsers(dest="command", required=True)
    
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
    
    # Parse arguments
    args = parser.parse_args()

    # Example: Handle commands
    if args.command == "remove_dependent":
        print(f"Running `remove_dependent` with:")
        print(f"Input TSV: {args.input_tsv}")
        print(f"Output TSV: {args.output_tsv}")
        print(f"Column: {args.column}")
