from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_serializer

class JsonlBatch(BaseModel):
    """MongoDB model for JSONL batch storage"""
    batch_number: int
    content: str  # JSONL content as string
    record_count: int
    created_at: datetime = datetime.now()
    status: str = "created"
    source_dataframe: str  # Path to source DataFrame pickle
    mongodb_id: Optional[str] = None

    @model_serializer
    def ser_model(self) -> Dict[str, Any]:
        # Manually build dictionary to avoid recursion with model_dump()
        data = {
            "batch_number": self.batch_number,
            "content": self.content,
            "record_count": self.record_count,
            "status": self.status,
            "source_dataframe": self.source_dataframe
        }
        
        # Handle datetime separately
        if hasattr(self, 'created_at') and self.created_at is not None:
            data["created_at"] = self.created_at.isoformat()
            
        # Add optional fields if present
        if self.mongodb_id is not None:
            data["mongodb_id"] = self.mongodb_id
            
        return data

class NegationResponse(BaseModel):
    """Schema for OpenAI response about negation analysis."""
    negation_present: bool = Field(description="Whether negation is present in the text")
    negation_types: Optional[List[str]] = Field(
        default=None, 
        description="Types of negation found (e.g., 'explicit', 'implicit', 'syntactic', etc.)"
    )
    short_explanation: str = Field(description="Brief explanation of negation findings")

class FileMetadata(BaseModel):
    """File metadata for tracking uploads."""
    file_path: Optional[str] = None
    file_name: str
    file_size: int
    created_at: datetime
    uploaded_at: Optional[datetime] = None
    status: str  # ready, uploaded, processing, completed, error
    openai_file_id: Optional[str] = None
    batch_id: Optional[str] = None
    attempts: int = 0
    mongodb_id: Optional[str] = None
    jsonl_batch_id: Optional[str] = None  # Reference to JsonlBatch

    @model_serializer
    def ser_model(self) -> Dict[str, Any]:
        # Convert all datetime objects to ISO format strings
        data = self.model_dump()
        for field in ['created_at', 'uploaded_at']:
            if field in data and isinstance(data[field], datetime):
                data[field] = data[field].isoformat()
        return data

class BatchRequest(BaseModel):
    """OpenAI batch request metadata."""
    file_id: str  # MongoDB ID of the file
    openai_file_id: str
    jsonl_batch_id: Optional[str] = None  # Reference to JsonlBatch
    batch_id: str  # OpenAI batch ID
    status: str  # in_progress, completed, failed, expired
    created_at: datetime
    expires_at: datetime
    last_checked: datetime
    output_file_id: Optional[str] = None
    error: Optional[str] = None
    mongodb_id: Optional[str] = None

    @model_serializer
    def ser_model(self) -> Dict[str, Any]:
        data = self.model_dump()
        for field in ['created_at', 'last_checked']:
            if field in data and isinstance(data[field], datetime):
                data[field] = data[field].isoformat()
        return data

class ProcessingResult(BaseModel):
    """Results of batch processing."""
    batch_id: str
    file_id: str
    success_count: int
    error_count: int
    raw_result_id: str  # MongoDB ID of raw results

    @model_serializer
    def ser_model(self) -> Dict[str, Any]:
        return self.model_dump()

class BatchError(BaseModel):
    """Error information for failed batches."""
    batch_id: str
    file_id: str
    error_type: str  # rate_limit, processing_error, etc.
    error_message: str
    timestamp: datetime

    @model_serializer
    def ser_model(self) -> Dict[str, Any]:
        data = self.model_dump()
        if 'timestamp' in data and isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data
