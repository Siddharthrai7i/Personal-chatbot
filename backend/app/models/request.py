from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Request model for querying the chatbot"""
    query: str = Field(..., min_length=1, description="User's question")
    top_k: Optional[int] = Field(5, ge=1, le=10, description="Number of chunks to retrieve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are your hobbies?",
                "top_k": 5
            }
        }


class DocumentMetadata(BaseModel):
    """Optional metadata for uploaded documents"""
    category: Optional[str] = None
    tags: Optional[list] = None
    description: Optional[str] = None