from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Import configuration
from app.config import (
    GOOGLE_API_KEY, CHROMA_DB_PATH, COLLECTION_NAME,
    UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
    EMBEDDING_MODEL, LLM_MODEL
)

# Import core components
from app.core.vector_store import VectorStore
from app.core.embeddings import EmbeddingGenerator
from app.core.llm_handler import LLMHandler
from app.core.retriever import Retriever
from app.core.document_processor import DocumentProcessor
from app.utils.text_splitter import TextSplitter

# Import API routers
from app.api import health, query, upload

# Global components (initialized at startup)
vector_store = None
retriever = None
document_processor = None
text_splitter = None
embedding_generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events
    """
    # Startup
    print("\n" + "="*60)
    print("🚀 STARTING FASTAPI SERVER")
    print("="*60)
    
    # Load environment variables
    load_dotenv(override=True)
    
    # Initialize components
    global vector_store, retriever, document_processor, text_splitter, embedding_generator
    
    print("\n🔧 Initializing components...")
    
    # Initialize vector store
    vector_store = VectorStore(
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        api_key=GOOGLE_API_KEY,
        model_name=EMBEDDING_MODEL
    )
    
    # Initialize LLM handler
    llm_handler = LLMHandler(
        api_key=GOOGLE_API_KEY,
        model_name=LLM_MODEL
    )
    
    # Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        llm_handler=llm_handler,
        top_k=TOP_K_RESULTS
    )
    
    # Initialize document processor
    document_processor = DocumentProcessor()
    
    # Initialize text splitter
    text_splitter = TextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Set dependencies in routers
    health.set_vector_store(vector_store)
    query.set_retriever(retriever)
    upload.set_dependencies(
        document_processor,
        text_splitter,
        embedding_generator,
        vector_store,
        UPLOAD_DIR
    )
    
    print("\n✅ All components initialized successfully!")
    print("="*60 + "\n")
    
    yield
    
    # Shutdown
    print("\n" + "="*60)
    print("👋 SHUTTING DOWN SERVER")
    print("="*60 + "\n")


# Create FastAPI app
app = FastAPI(
    title="Personal RAG Chatbot API",
    description="RAG-based chatbot API for personal knowledge base",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(query.router, tags=["Query"])
app.include_router(upload.router, tags=["Upload"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Personal RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )