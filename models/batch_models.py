from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class FileMetadata(BaseModel):
    """Data model for OpenAI file metadata."""
    file_name: str
    file_size: int
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "ready"  # ready, uploaded, batch_submitted, completed, failed
    attempts: int = 0
    jsonl_batch_id: Optional[str] = None
    mongodb_id: Optional[str] = None
    openai_file_id: Optional[str] = None
    uploaded_at: Optional[datetime] = None
    batch_id: Optional[str] = None
    error: Optional[str] = None

class BatchRequest(BaseModel):
    """Data model for OpenAI batch request."""
    batch_id: str
    file_id: str
    jsonl_batch_id: Optional[str] = None
    status: str  # validating, in_progress, finalizing, completed, failed, expired, cancelled
    submitted_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_checked: Optional[datetime] = None
    output_file_id: Optional[str] = None
    error: Optional[str] = None
    mongodb_id: Optional[str] = None
    
class JsonlBatch(BaseModel):
    """Data model for JSONL batch stored in MongoDB."""
    batch_number: int
    content: str
    record_count: int
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = "created"  # created, registered, processed
    source_dataframe: str
    file_id: Optional[str] = None
    mongodb_id: Optional[str] = None

class BatchResult(BaseModel):
    """Data model for processed batch results."""
    custom_id: str
    batch_id: str
    file_id: Optional[str] = None
    jsonl_batch_id: Optional[str] = None
    negation_present: bool
    negation_types: List[str] = []
    explanation: str
    processed_at: datetime = Field(default_factory=datetime.now)
    raw_response: Dict[str, Any]
    mongodb_id: Optional[str] = None
