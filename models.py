from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class NegationResponse(BaseModel):
    """Schema for OpenAI response about negation analysis."""
    negation_present: bool = Field(description="Whether negation is present in the text")
    negation_types: Optional[List[str]] = Field(
        default=None, 
        description="Types of negation found (e.g., 'explicit', 'implicit', 'syntactic', etc.)"
    )
    short_explanation: str = Field(description="Brief explanation of negation findings")


class FileMetadata(BaseModel):
    """Metadata for tracking JSONL files for batch processing."""
    mongodb_id: Optional[str] = None
    file_path: str
    file_name: str
    file_size: int
    status: str = "ready"  # ready, uploaded, processing, processed, error
    created_at: datetime
    uploaded_at: Optional[datetime] = None
    openai_file_id: Optional[str] = None
    batch_id: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0


class BatchRequest(BaseModel):
    """Information about an OpenAI batch processing request."""
    mongodb_id: Optional[str] = None
    file_id: str  # Reference to FileMetadata in MongoDB
    openai_file_id: str
    batch_id: str
    status: str  # in_progress, completed, failed, expired, cancelled, processed
    created_at: datetime
    expires_at: datetime
    last_checked: datetime
    output_file_id: Optional[str] = None
    error: Optional[str] = None
    processed_at: Optional[datetime] = None
    success_count: Optional[int] = None
    error_count: Optional[int] = None


class ProcessingResult(BaseModel):
    """Results from processing a batch of patent data."""
    batch_id: str
    file_id: str
    raw_result_id: str  # Reference to raw results in MongoDB
    success_count: int
    error_count: int
    processed_at: Optional[datetime] = None


class BatchError(BaseModel):
    """Information about batch processing errors."""
    batch_id: str
    file_id: Optional[str] = None
    error_type: str
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int = 0
    retry_limit: int = 3
    resolved: bool = False
