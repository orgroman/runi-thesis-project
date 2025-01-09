import asyncio
from aiohttp import ClientSession
import openai

class LLamaFileModel:
    def __init__(self, endpoint_url="http://localhost:8080/v1/chat/completions", **kwargs):
        # Default endpoint points to the OpenAI-compatible chat completions endpoint
        self.url = endpoint_url
        self._client = openai.AsyncOpenAI(base_url="http://localhost:8080/v1", api_key = "sk-no-key-required")

    @classmethod
    def create_model(cls, **kwargs):
        return cls(**kwargs)

    async def acomplete_chat(self, messages):
        """
        Send the messages to the llamafile chat completions endpoint and return the JSON response.
        """
        # payload = {
        #     "model": "LLaMA_CPP",
        #     "messages": messages
        # }
        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": "Bearer no-key"
        # }
        
        result = await self._client.chat.completions.create(
            model="LLaMA_CPP",
            messages=messages,
            max_completion_tokens=5,
            max_tokens=5
        )
        return result
        
        # async with ClientSession() as session:
        #     async with session.post(self.url, json=payload, headers=headers) as response:
        #         # Return the full JSON response
        #         result = await response.json()
        #         return result

# Example usage:
async def main():
    # Create our model pointing to the local llamafile endpoint
    model = LLamaFileModel.create_model(
        endpoint_url="http://localhost:8080/v1/chat/completions"
    )

    # Example messages that match the curl sample
    messages = [
        {
            "role": "system",
            "content": "You are LLAMAfile, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
        },
        {
            "role": "user",
            "content": "Write a limerick about python exceptions"
        }
    ]

    # Send messages to the llamafile server
    response = await model.acomplete_chat(messages)

    # Print the response in a formatted way
    import json
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
