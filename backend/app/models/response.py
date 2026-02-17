from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class SourceInfo(BaseModel):
    """Information about a source chunk"""
    chunk_text: str
    source_file: str
    chunk_index: int
    similarity_score: float


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    success: bool
    answer: Optional[str] = None
    sources: List[SourceInfo] = []
    chunks_used: int = 0
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Response model for upload endpoint"""
    success: bool
    message: str
    filename: str
    chunks_created: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    database_connected: bool
    total_documents: int
    total_chunks: int
    sources: List[str] = []


class ErrorResponse(BaseModel):
    """Generic error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None