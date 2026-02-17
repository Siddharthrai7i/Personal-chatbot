from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.response import UploadResponse
import os
from pathlib import Path
import shutil

router = APIRouter()

# These will be set from main.py
document_processor = None
text_splitter = None
embedding_generator = None
vector_store = None
upload_dir = None

def set_dependencies(processor, splitter, embedder, store, upload_path):
    """Set all required dependencies"""
    global document_processor, text_splitter, embedding_generator, vector_store, upload_dir
    document_processor = processor
    text_splitter = splitter
    embedding_generator = embedder
    vector_store = store
    upload_dir = upload_path
@router.post("/reset-database")
async def reset_database():
    """
    WARNING: This deletes ALL data in the database!
    Use only for testing or when changing embedding models.
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        success = vector_store.reset()
        
        if success:
            return {
                "success": True,
                "message": "Database reset successfully. All data deleted."
            }
        else:
            return {
                "success": False,
                "message": "Failed to reset database"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document
    
    This endpoint:
    1. Receives a PDF or TXT file
    2. Saves it to disk
    3. Extracts text
    4. Splits into chunks
    5. Generates embeddings
    6. Stores in ChromaDB
    """
    if not all([document_processor, text_splitter, embedding_generator, vector_store, upload_dir]):
        raise HTTPException(status_code=500, detail="Server not properly initialized")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.txt'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file
    file_path = Path(upload_dir) / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📁 Saved file: {file_path}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        # Step 1: Extract text
        print(f"📄 Extracting text from {file.filename}...")
        text = document_processor.process_file(str(file_path))
        
        if not text:
            return UploadResponse(
                success=False,
                message="Failed to extract text from document",
                filename=file.filename,
                error="Could not extract text"
            )
        
        # Step 2: Split into chunks
        print(f"✂️  Splitting text into chunks...")
        chunks = text_splitter.split_text(
            text=text,
            source_file=file.filename
        )
        
        if not chunks:
            return UploadResponse(
                success=False,
                message="Failed to create chunks",
                filename=file.filename,
                error="Could not split text into chunks"
            )
        
        # Step 3: Generate embeddings
        print(f"🧮 Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_generator.generate_embeddings_batch(texts, show_progress=True)
        
        # Filter out failed embeddings
        valid_chunks = []
        valid_embeddings = []
        
        for chunk, embedding in zip(chunks, embeddings):
            if embedding is not None:
                valid_chunks.append(chunk)
                valid_embeddings.append(embedding)
        
        if not valid_embeddings:
            return UploadResponse(
                success=False,
                message="Failed to generate embeddings",
                filename=file.filename,
                error="Could not generate embeddings"
            )
        
        # Step 4: Store in ChromaDB
        print(f"💾 Storing {len(valid_chunks)} chunks in database...")
        success = vector_store.add_documents(valid_chunks, valid_embeddings)
        
        if not success:
            return UploadResponse(
                success=False,
                message="Failed to store in database",
                filename=file.filename,
                error="Could not store embeddings"
            )
        
        return UploadResponse(
            success=True,
            message=f"Successfully processed {file.filename}",
            filename=file.filename,
            chunks_created=len(valid_chunks)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/document/{filename}")
async def delete_document(filename: str):
    """
    Delete a document and all its chunks from the database
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        success = vector_store.delete_by_source(filename)
        
        if success:
            return {"success": True, "message": f"Deleted {filename}"}
        else:
            return {"success": False, "message": f"Document {filename} not found"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    """
    Get list of all documents in the database
    """
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        sources = vector_store.list_sources()
        stats = vector_store.get_stats()
        
        return {
            "success": True,
            "documents": sources,
            "total_documents": len(sources),
            "total_chunks": stats.get('total_chunks', 0)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))