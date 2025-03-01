import json
from datetime import datetime
from typing import Any, Dict

class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects by converting them to ISO format strings."""
    
    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

def datetime_serializable(obj: Any) -> Any:
    """Convert an object to be JSON serializable, handling datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: datetime_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [datetime_serializable(i) for i in obj]
    else:
        return obj
