import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for the research thesis different pipelines and experiments")
    subparsers = parser.add_subparsers()
    
    parser.add_argument("train_model", help="Command to run")
    remove_dependent_claims_p = subparsers.add_parser("remove_dependent_claims", help="Detect and remove and dependent patent claim from the input dataset")
    remove_dependent_claims_p.add_argument("input_tsv", help="Input dataset to remove dependent claims from")
    remove_dependent_claims_p.add_argument("output_tsv", help="Output dataset to save the results to")
    remove_dependent_claims_p.add_argument("column", help="Column to remove dependent claims from")
                                           
    args = parser.parse_args()
    print(args.command)
