import re

# Regular expression pattern to match claim or claims numbers
pattern = re.compile(r'\bclaims?(?: number)? \d+\b', re.IGNORECASE)

def predict_text(text):
    return bool(pattern.search(text))

