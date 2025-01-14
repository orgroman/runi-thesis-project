from tqdm import tqdm
from pathlib import Path


def main():
    jsonl_root_dir = Path(r'C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_train')
    jsonl_files = list(jsonl_root_dir.rglob("*.jsonl"))
    
    # Iterate over all jsonl files and for each file replace any match of the string gpt-4o-mini with gpt-4o
    # Print the number of replacements made
    replacements = 0
    for jsonl_file in tqdm(jsonl_files):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            content = f.read()
        new_content = content.replace('gpt-4o-mini', 'gpt-4o')
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        replacements += content.count('gpt-4o-mini')

    
if __name__ == "__main__":
    main()