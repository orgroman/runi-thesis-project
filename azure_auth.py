from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from pathlib import Path
import pandas as pd

# authenticate
credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="8592e500-3312-4991-9d2a-2b97e43b1810",
    resource_group_name="rgrunithesis",
    workspace_name="runi_ml_thesis_workspace",
)

tsv_direcotry = Path(r'C:\workspace_or_private\repos\runi-thesis-project\hidrive')
tsv_files = list(tsv_direcotry.rglob('*.tsv'))
for tsv_file in tsv_files:
    # convert the tsv file to csv and upload it to the workspace
    print(f'Uploading {tsv_file.name}')
    csv_file = tsv_file.with_suffix('.csv')
    if not csv_file.exists():
        print(f'Converting {tsv_file.name} to csv')
        df = pd.read_csv(tsv_file, sep='\t')    
        # write the csv file
        df.to_csv(csv_file, index=False)
        print(f'Converted {tsv_file.name} to csv')
    
    ds_name = f'{tsv_file.stem}_data'
    my_data = Data(
        name=ds_name,
        description=f'PatentMatch data from {tsv_file.name}',
        type=AssetTypes.URI_FILE,
        version='initial',
        path=str(csv_file)
    )
    ## create data asset if it doesn't already exist:
    try:
        print(f"Checking if data asset {my_data.name} exists")
        data_asset = ml_client.data.get(name=ds_name, version='initial')
        print(
            f"Data asset already exists. Name: {my_data.name}, version: {my_data.version}"
        )
    except:
        print(f"Data asset {my_data.name} doesn't exist. Creating it.")
        ml_client.data.create_or_update(my_data)
        print(f"Data asset created. Name: {my_data.name}, version: {my_data.version}")
            
    print(f'Uploaded {tsv_file.name}')
print('done')