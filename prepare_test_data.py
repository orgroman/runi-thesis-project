import pandas as pd
import json
from pathlib import Path


def create_jsonl_entry(row, field_name):
    """Creates a JSONL entry in the format expected by OpenAI."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful patent analysis assistant."},
            {"role": "user", "content": f"Here is the patent text:\n\n{row[field_name]}"}
        ]
    }


def main():
    # Input CSV file
    input_csv = Path(r"G:\My Drive\Colab Notebooks\research_data\thesis\patentmatch\patentmatch_test\patentmatch_test_no_claims.csv")
    
    # Output directories
    output_base = Path(__file__).parent / "hidrive" / "patentmatch_test"
    text_dir = output_base / "text"
    text_b_dir = output_base / "text_b"

    # Create directories if they don't exist
    text_dir.mkdir(parents=True, exist_ok=True)
    text_b_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    # Process in chunks of 1000 rows
    chunk_size = 1000
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        chunk_num = i // chunk_size

        # Create JSONL files for both text and text_b
        text_file = text_dir / f"chunk_{chunk_num}.jsonl"
        text_b_file = text_b_dir / f"chunk_{chunk_num}.jsonl"

        # Write text entries
        with open(text_file, 'w', encoding='utf-8') as f:
            for _, row in chunk.iterrows():
                entry = create_jsonl_entry(row, 'text')
                f.write(json.dumps(entry) + '\n')

        # Write text_b entries
        with open(text_b_file, 'w', encoding='utf-8') as f:
            for _, row in chunk.iterrows():
                entry = create_jsonl_entry(row, 'text_b')
                f.write(json.dumps(entry) + '\n')

        print(f"Processed chunk {chunk_num}: {len(chunk)} rows")
        print(f"  -> {text_file}")
        print(f"  -> {text_b_file}")

    print("All done!")


if __name__ == "__main__":
    main()
