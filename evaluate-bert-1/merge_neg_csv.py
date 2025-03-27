from typing import List, Optional
import pandas as pd
import ast
import json
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class NegationResponse(BaseModel):
    negation_present: bool
    negation_types: List[str]
    short_explanation: Optional[str]
    
def main():
    df_1 = pd.read_csv(r"C:\Users\orgrd\workspace\data\patentmatch_test\patentmatch_test_no_claims.csv")
    df_2 = pd.read_csv(r"C:\Users\orgrd\workspace\data\patentmatch_test\patentmatch_test_no_claims_negannotated.csv")
    df_2['index'] = df_2['part_1']
    
    merged_df = pd.merge(df_1, df_2, how='inner', on=['index'])
    
    df = merged_df.copy()
    parser = JsonOutputParser(pydantic_object=NegationResponse)
    
    # Function to parse JSON safely
    def parse_json(val):
        try:
            # Handle single quotes and escape sequences properly
            parsed_val = parser.parse(val)
            return parsed_val

            # cleaned_val = val.replace("\'", "")  # Fix incorrectly escaped single quotes
            # cleaned_val = cleaned_val.replace("'", '"')  # Convert single quotes to double quotes
            # cleaned_val = cleaned_val.replace('\n', '')  # Remove newline characters
            # brackets_char = cleaned_val[-1]
            # if brackets_char != '}':
            #     cleaned_val = cleaned_val[:-1] + '"}'

            # return json.loads(cleaned_val) if isinstance(val, str) else {}
        except json.JSONDecodeError as e:
            return {}
    
    # Flatten text_a_response JSON
    df['text_a_response'] = df['text_a_response'].apply(parse_json)
    text_a_df = pd.json_normalize(df['text_a_response'])
    text_a_df.columns = [f"text_a_response_{col}" for col in text_a_df.columns]
    df = pd.concat([df, text_a_df], axis=1)
    
    # Flatten text_b_response JSON
    df['text_b_response'] = df['text_b_response'].apply(parse_json)
    text_b_df = pd.json_normalize(df['text_b_response'])
    text_b_df.columns = [f"text_b_response_{col}" for col in text_b_df.columns]
    df = pd.concat([df, text_b_df], axis=1)
    
    # Dropping the original dictionary columns
    df.drop(columns=['text_a_response', 'text_b_response'], inplace=True)
    
    # Save processed DataFrame
    reduced_df = df[['index', 'claim_id','patent_application_id','cited_document_id','text','text_b','label','text_a_response_negation_present','text_a_response_short_explanation','text_a_response_negation_types','text_b_response_negation_present','text_b_response_short_explanation','text_b_response_negation_types']]
    reduced_df.to_csv(r"C:\Users\orgrd\workspace\data\patentmatch_test\test_no_claims_with_neg_final.csv", index=False)
    print('Flattened data saved to flattened_output.csv')
    
if __name__ == "__main__":
    print("Running main function")
    main()
