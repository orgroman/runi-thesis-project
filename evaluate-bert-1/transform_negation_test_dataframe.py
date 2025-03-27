import asyncio
import json
from pydash import get
from motor.motor_asyncio import AsyncIOMotorClient
from tqdm import tqdm
import pandas as pd
mongodb_client = AsyncIOMotorClient("mongodb://user:pass@localhost:27017")


ANNOTATED_PAIRS_COLLECTION = "annotated_pairs"

async def main():
    # Load all documents from the collection
    test_df = pd.read_csv(r"C:\Users\orgrd\workspace\data\patentmatch_test\patentmatch_test_no_claims.csv")

    coll = mongodb_client["patent_negation"][ANNOTATED_PAIRS_COLLECTION]
    annotated_samples = []
    async for doc in coll.find():
        annotated_samples.append(doc)
        
    flattened_samples = []
    for x in tqdm(annotated_samples):
        custom_id = get(x, 'annotated_sample.text_a.sample.custom_id')
        request_parts = custom_id.split('_')
        part_1 = request_parts.pop(-1)
        part_2 = request_parts.pop(-1)
        text_a_response = get(x, 'annotated_sample.text_a.result.response.body.choices.0.message.content')
        #formatted_a_response = json.loads(text_a_response)
        text_b_response = get(x, 'annotated_sample.text_b.result.response.body.choices.0.message.content')
        #formatted_b_response = json.loads(text_b_response)
        text_a_input = get(x,'annotated_sample.text_a.sample.body.messages.1.content').removeprefix('Analyze the following text: ')
        text_b_input = get(x,'annotated_sample.text_b.sample.body.messages.1.content').removeprefix('Analyze the following text: ')
        flattened_samples.append({
            "part_1": part_1,
            "part_2": part_2,
            "text_a_input": text_a_input,
            "text_b_input": text_b_input,
            "text_a_response": text_a_response,
            "text_b_response": text_b_response
        })

    new_df = pd.DataFrame(flattened_samples)
    # Write the dataframe to a CSV file
    new_df.to_csv(r"C:\Users\orgrd\workspace\data\patentmatch_test\patentmatch_test_no_claims_negannotated.csv", index=False)

    print("done")

if __name__ == "__main__":
    asyncio.run(main())