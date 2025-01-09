import asyncio
from aiohttp import ClientSession
import openai
import textwrap

class LLamaFileModel:
    def __init__(
        self, endpoint_url="http://localhost:8080/v1/chat/completions", **kwargs
    ):
        # Default endpoint points to the OpenAI-compatible chat completions endpoint
        self.url = endpoint_url
        self._client = openai.AsyncOpenAI(
            base_url="http://localhost:8080/v1", api_key="sk-no-key-required"
        )

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
            model="LLaMA_CPP", messages=messages, max_completion_tokens=5, max_tokens=5
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
    messages = [
        {
            "role": "system",
            "content": textwrap.dedent("""

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. The user will provide one or more sentences from various sources (including academic, technical, or legal domains). Your task is to determine whether **linguistic negation** is present in the text.

**Important Requirements**:
1. You must only output “YES” or “NO” (a single word) to indicate presence or absence of any form of negation.  
2. Provide no additional information or context beyond “YES” or “NO.”

**Background**:
Negation can appear in multiple forms:
- **Sentence-level negation**: “not,” “no,” “never,” “cannot,” “nor,” etc.  
- **Constituent negation**: negating only part of the sentence.  
- **Morphological negation**: negative prefixes/affixes (e.g., “un-,” “dis-,” “in-,” “non-,” “asynchronous” for “not synchronous”).  
- **Lexical negation**: words inherently carrying a negative meaning (e.g., “fail,” “deny,” “lack,” “impossible,” “refuse,” “reject,” etc.).  
- **Negative Polarity Items** (NPIs): words like “any,” “ever,” “at all,” used in contexts requiring negation.

Your procedure:
1. Check if any negation cue exists (explicit negation markers, morphological negation, inherently negative verbs/adjectives, NPIs in a negative context, etc.).
2. Output only “YES” if negation is found.
3. Output only “NO” if no negation is found.

Here are some examples:


<|begin_example|> (Morphological Negation) Input: “The process is asynchronous, preventing synchronous updates.”  Output: YES <|end_of_example|>
<|begin_example|> (Lexical Negation) Input: “The system fails to meet the required standard.” Output: YES <|end_of_example|>
<|begin_example|> (Standard Sentential Negation) Input: “They did not complete the review by the deadline.”  Output: YES <|end_of_example|>

<|begin_example|> (Constituent Negation) Input: “The final output is incorrect.”  Output: YES <|end_of_example|>
<|begin_example|> (No Negation) Input: “We tested the prototype in multiple languages to ensure consistency.”  Output: NO <|end_of_example|>

Reminder:
- Output only “YES” or “NO”, with no further text.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is the text to analyze:
[User’s sentence goes here]

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                                    """),
        },
        {
            "role": "user",
            "content": row[args.column],
        },
    ]

    # Send messages to the llamafile server
    response = await model.acomplete_chat(messages)

    # Print the response in a formatted way
    import json

    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
