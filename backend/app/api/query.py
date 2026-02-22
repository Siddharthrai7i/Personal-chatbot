from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.request import QueryRequest
from app.models.response import QueryResponse, SourceInfo

router = APIRouter()

# Global variable - will be set from main.py
retriever = None


def set_retriever(ret):
    """Set the retriever instance"""
    global retriever
    retriever = ret


@router.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """
    Query the chatbot with a question
    """
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    try:
        result = retriever.retrieve_and_generate(
            query=request.query,
            top_k=request.top_k
        )
        
        sources = [SourceInfo(**source) for source in result.get('sources', [])]
        
        return QueryResponse(
            success=result['success'],
            answer=result.get('answer'),
            sources=sources,
            chunks_used=result.get('chunks_used', 0),
            error=result.get('error')
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_chatbot_stream(request: QueryRequest):
    """
    Query the chatbot with streaming response (Server-Sent Events).
    Tokens are sent as 'data: <token>\n\n' events.
    Sources are sent as a JSON event at the end.
    Final event is 'data: [DONE]\n\n'.
    """
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    return StreamingResponse(
        retriever.retrieve_and_generate_stream(
            query=request.query,
            top_k=request.top_k
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
