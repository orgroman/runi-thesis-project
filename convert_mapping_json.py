import json
from pathlib import Path


def main():
    orig_json = r'C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_train\openai_files_gpt4o_mapping.json'
    
    # load the json
    with open(orig_json, 'r') as f:
        data = json.load(f)
    
    column_mapping = {
        "jsonl_text_b_patentmatch_train_no_claims": "text_b",
        "jsonl_text_patentmatch_train_no_claims": "text",
    }
    

    text_b_data = {}
    text_data = {}
    new_data = {
        'text_b': text_b_data,
        'text': text_data
    }
        
    for k,v in data.items():
        new_k = Path(k)
        batch_part = new_k.stem
        column_part = new_k.parent.stem
        
        column_type = column_mapping[column_part]
        if column_type == 'text':
            new_data['text'][batch_part] = v
        else:
            new_data['text_b'][batch_part] = v
                    
    
    new_json = r'C:\workspace_or_private\repos\runi-thesis-project\hidrive\patentmatch_train\openai_files_gpt4o_mapping_new.json'
    with open(new_json, 'w') as f:
        f.write(json.dumps(new_data))
        
    
        
        
        
        
        



if __name__ == "__main__":
    main()