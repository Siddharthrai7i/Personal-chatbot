from fastapi import APIRouter
from app.models.response import HealthResponse

router = APIRouter()

# This will be set from main.py
vector_store = None

def set_vector_store(store):
    """Set the vector store instance"""
    global vector_store
    vector_store = store


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check system health and return database statistics
    """
    try:
        if not vector_store:
            return HealthResponse(
                status="error",
                database_connected=False,
                total_documents=0,
                total_chunks=0,
                sources=[]
            )
        
        stats = vector_store.get_stats()
        
        return HealthResponse(
            status="healthy",
            database_connected=True,
            total_documents=stats.get('total_sources', 0),
            total_chunks=stats.get('total_chunks', 0),
            sources=stats.get('sources', [])
        )
    
    except Exception as e:
        return HealthResponse(
            status="error",
            database_connected=False,
            total_documents=0,
            total_chunks=0,
            sources=[]
        )