from typing import Optional

from pydantic import BaseModel
from typing import List
import json
import pandas as pd

system_prompt = """
You are a linguistic analysis assistant specializing in analyzing patent claims and cited technical descriptions.

Your task is to identify and analyze negation in patent-related texts, focusing on both explicit and implicit negation. You should:

1. Determine whether **any negation** is present in the text.
2. For each instance of negation, extract the following:
   - `cue`: The word or phrase that signals the negation (e.g., “not”, “no”, “fail”, “without”, etc.)
   - `explicit`: Whether the cue is **explicit** (e.g., “not”, “no”, “never”) or **implicit** (e.g., “fail”, “limited”, “lack”)
   - `type`: The category of negation, one of:
     - `"standard"` – basic syntactic negation using “not” or equivalent
     - `"lexical"` – negation expressed via a lexical item (e.g., “fail”, “lack”, “omit”)
     - `"NPI-induced"` – negation inferred from Negative Polarity Items (e.g., “ever”, “any”, “yet”)
     - `"constituent"` – negation of a phrase or clause using structural means
   - `scope`: The phrase or clause that is negated semantically
   - `focus`: The most directly affected word or concept within the scope
   - `event`: The verb or proposition that is being negated

3. Provide a short summary explanation of your findings.

4. Return your response as JSON that matches the following structure:
```json
{
  "negation_present": boolean,
  "instances": [
    {
      "cue": string,
      "explicit": boolean,
      "type": "standard" | "lexical" | "NPI-induced" | "constituent",
      "scope": string,
      "focus": string,
      "event": string
    }
  ],
  "negation_types_summary": [ "standard", "lexical", "NPI-induced", "constituent" ],
  "short_explanation": string
}

---

### 🔹 **Example 1: Explicit Standard Negation**
**Text:**  
> "No shrinkage was observed in this carpet."

```json
{
  "negation_present": true,
  "instances": [
    {
      "cue": "No shrinkage",
      "explicit": true,
      "type": "standard",
      "scope": "No shrinkage was observed in this carpet",
      "focus": "shrinkage",
      "event": "was observed"
    }
  ],
  "negation_types_summary": ["standard"],
  "short_explanation": "The phrase 'No shrinkage' explicitly negates the occurrence of shrinkage."
}
```

---

### 🔹 **Example 2: Lexical Negation (Implicit)**
**Text:**  
> "This level of centrifugal pumping has limited negative impact on disk temperature."

```json
{
  "negation_present": true,
  "instances": [
    {
      "cue": "limited negative impact",
      "explicit": false,
      "type": "lexical",
      "scope": "limited negative impact on disk temperature",
      "focus": "negative impact",
      "event": "has"
    }
  ],
  "negation_types_summary": ["lexical"],
  "short_explanation": "The phrase implies absence of significant adverse effect, representing lexical negation."
}
```

---

### 🔹 **Example 3: NPI-Induced + Standard Negation**
**Text:**  
> "The system hasn't ever operated at high voltage."

```json
{
  "negation_present": true,
  "instances": [
    {
      "cue": "hasn't",
      "explicit": true,
      "type": "standard",
      "scope": "hasn't ever operated at high voltage",
      "focus": "ever operated",
      "event": "operated"
    },
    {
      "cue": "ever",
      "explicit": false,
      "type": "NPI-induced",
      "scope": "ever operated at high voltage",
      "focus": "ever",
      "event": "operated"
    }
  ],
  "negation_types_summary": ["standard", "NPI-induced"],
  "short_explanation": "Standard negation via 'hasn't' is reinforced by the NPI 'ever'."
}
```

---

### 🔹 **Example 4: Constituent Negation**
**Text:**  
> "The validation logic checks either the MAC address or the serial number, but not both."

```json
{
  "negation_present": true,
  "instances": [
    {
      "cue": "not both",
      "explicit": true,
      "type": "constituent",
      "scope": "but not both",
      "focus": "MAC address and serial number",
      "event": "checks"
    }
  ],
  "negation_types_summary": ["constituent"],
  "short_explanation": "Constituent negation arises from the exclusion of simultaneous truth for both elements."
}
```

---

### 🔹 **Example 5: No Negation (Negative Case)**
**Text:**  
> "The rotor assembly includes a heat shield that spans between the turbine and compressor modules."

```json
{
  "negation_present": false,
  "instances": [],
  "negation_types_summary": [],
  "short_explanation": "The sentence is fully affirmative and contains no form of negation."
}
```
"""

class NegationInstance(BaseModel):
    cue: str
    explicit: bool
    type: str  # One of: 'standard', 'constituent', 'lexical', 'NPI-induced'
    scope: Optional[str]
    focus: Optional[str]
    event: Optional[str]

class NegationAnalysisResponse(BaseModel):
    negation_present: bool
    instances: List[NegationInstance]
    negation_types_summary: Optional[List[str]]
    short_explanation: str




if __name__ == "__main__":
    patentmatch_test_file = r'C:\Users\orgrd\workspace\data\patentmatch_test\patentmatch_test_no_claims.csv'
    df = pd.read_csv(patentmatch_test_file)

    # Prepare the jsonl files
    text_a = df['text']
    text_b = df['text_b']

    # Split to 1000 samples batches
    batch_size = 1000
    batch_count = len(df) // batch_size
    if len(df) % batch_size != 0:
        batch_count += 1

    for i in range(batch_count):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        text_a_batch = text_a[start_idx:end_idx]
        text_b_batch = text_b[start_idx:end_idx]

        # Prepare the JSONL request
        jsonl_request = []
        for idx, (text_a, text_b) in enumerate(zip(text_a_batch, text_b_batch)):
            request = {
                "custom_id": f"request_text_b_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt  # ← Full prompt with ICL examples and JSON schema instructions
                        },
                        {
                            "role": "user",
                            "content": f"Text: {text_a}\n\nReturn the negation analysis for this text as valid JSON."
                        }
                    ],
                    "max_tokens": 500,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "negation_analysis_response",
                            "schema": NegationAnalysisResponse.model_json_schema(),
                            "strict": True
                        } 
                    }
                }
            }
            jsonl_request.append(json.dumps(request))

        # Save the JSONL request to a file
        with open(f'requests_batch_{i}.jsonl', 'w') as f:
            f.write('\n'.join(jsonl_request))

    print(f"Prepared {batch_count} JSONL request files with {batch_size} samples each.")
    print("You can now upload these files to OpenAI for processing.")
    print("Remember to use the following system prompt for the OpenAI API:")






    pass