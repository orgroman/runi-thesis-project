import asyncio
import datetime
import os
from openai import AsyncOpenAI

async def main():
    async_client = AsyncOpenAI(api_key=os.getenv("THESIS_ALON_OPENAI_API_KEY"))
    
    # list all the files
    files = await async_client.files.list()
    for f in files.data:        
        creation_date: int = f.created_at
        # print formatted date
        formatted_date = datetime.datetime.fromtimestamp(creation_date).strftime('%Y-%m-%d %H:%M:%S')
        print(f"File: {f.id} created at: {formatted_date}")


if __name__ == "__main__":
    asyncio.run(main())