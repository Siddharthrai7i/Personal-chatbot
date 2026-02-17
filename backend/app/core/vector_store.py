import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

class VectorStore:
    """
    Manages ChromaDB operations for storing and retrieving document embeddings
    """
    
    def __init__(self, db_path: str, collection_name: str = "personal_knowledge"):
        """
        Initialize ChromaDB client and collection
        
        Args:
            db_path: Path to store ChromaDB data
            collection_name: Name of the collection
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Personal knowledge base for RAG chatbot"}
        )
        
        print(f"✅ ChromaDB initialized at: {db_path}")
        print(f"📚 Collection: {collection_name}")
        print(f"📊 Current documents: {self.collection.count()}")
    
    def add_documents(
        self, 
        chunks: List[Dict],
        embeddings: List[List[float]]
    ) -> bool:
        """
        Add document chunks with their embeddings to ChromaDB
        
        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: List of embedding vectors
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks or not embeddings:
            print("⚠️  No chunks or embeddings provided")
            return False
        
        if len(chunks) != len(embeddings):
            print(f"❌ Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
            return False
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                # Generate unique ID for each chunk
                chunk_id = str(uuid.uuid4())
                ids.append(chunk_id)
                
                # Add the text
                documents.append(chunk['text'])
                
                # Add metadata (ChromaDB will store this with each embedding)
                metadata = {
                    'source_file': chunk.get('source_file', 'unknown'),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'chunk_size': chunk.get('chunk_size', 0),
                    'created_at': chunk.get('created_at', datetime.now().isoformat())
                }
                metadatas.append(metadata)
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"✅ Added {len(chunks)} chunks to ChromaDB")
            return True
            
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            return False
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> Dict:
        """
        Search for similar documents using query embedding
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = {
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'count': len(results['documents'][0]) if results['documents'] else 0
            }
            
            print(f"🔍 Found {formatted_results['count']} relevant chunks")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching: {str(e)}")
            return {'documents': [], 'metadatas': [], 'distances': [], 'count': 0}
    
    def delete_by_source(self, source_file: str) -> bool:
        """
        Delete all chunks from a specific source file
        
        Args:
            source_file: Name of the source file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents with this source
            results = self.collection.get(
                where={"source_file": source_file},
                include=['metadatas']
            )
            
            if not results['ids']:
                print(f"⚠️  No documents found from {source_file}")
                return False
            
            # Delete them
            self.collection.delete(ids=results['ids'])
            
            print(f"🗑️  Deleted {len(results['ids'])} chunks from {source_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting documents: {str(e)}")
            return False
    
    def list_sources(self) -> List[str]:
        """
        Get list of all unique source files in the database
        
        Returns:
            List of source filenames
        """
        try:
            # Get all documents
            results = self.collection.get(include=['metadatas'])
            
            if not results['metadatas']:
                return []
            
            # Extract unique source files
            sources = set()
            for metadata in results['metadatas']:
                if 'source_file' in metadata:
                    sources.add(metadata['source_file'])
            
            return sorted(list(sources))
            
        except Exception as e:
            print(f"❌ Error listing sources: {str(e)}")
            return []
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the database
        
        Returns:
            Dictionary with statistics
        """
        try:
            total_docs = self.collection.count()
            sources = self.list_sources()
            
            return {
                'total_chunks': total_docs,
                'total_sources': len(sources),
                'sources': sources,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            print(f"❌ Error getting stats: {str(e)}")
            return {}
    
    def reset(self) -> bool:
        """
        Delete all documents from the collection (use with caution!)
        
        Returns:
            True if successful
        """
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate it
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Personal knowledge base for RAG chatbot"}
            )
            
            print("🗑️  Database reset successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error resetting database: {str(e)}")
            return False


# Test function
def test_vector_store():
    """Test ChromaDB operations"""
    print("\n" + "="*60)
    print("🧪 TESTING VECTOR STORE (ChromaDB)")
    print("="*60)
    
    # Initialize
    from app.config import CHROMA_DB_PATH, COLLECTION_NAME
    store = VectorStore(db_path=CHROMA_DB_PATH, collection_name=COLLECTION_NAME)
    store.reset()
    # Create sample chunks
    sample_chunks = [
        {
            'text': 'I am a software engineer with 5 years of experience.',
            'chunk_index': 0,
            'source_file': 'about_me.txt',
            'chunk_size': 56,
            'created_at': datetime.now().isoformat()
        },
        {
            'text': 'I love playing guitar and photography.',
            'chunk_index': 1,
            'source_file': 'about_me.txt',
            'chunk_size': 40,
            'created_at': datetime.now().isoformat()
        }
    ]
    
    # Create fake embeddings (in real use, these come from Google API)
    # Each embedding is a vector of 768 dimensions (for Google's model)
    import random
    fake_embeddings = [
        [random.random() for _ in range(768)] for _ in range(len(sample_chunks))
    ]
    
    print("\n📤 Testing: Add documents...")
    success = store.add_documents(sample_chunks, fake_embeddings)
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    
    print("\n🔍 Testing: Search...")
    fake_query_embedding = [random.random() for _ in range(768)]
    results = store.search(fake_query_embedding, top_k=2)
    print(f"Found {results['count']} results")
    
    if results['documents']:
        print("\nTop result:")
        print(f"  Text: {results['documents'][0][:50]}...")
        print(f"  Source: {results['metadatas'][0]['source_file']}")
    
    print("\n📊 Testing: Get stats...")
    stats = store.get_stats()
    print(f"Total chunks: {stats.get('total_chunks', 0)}")
    print(f"Total sources: {stats.get('total_sources', 0)}")
    print(f"Sources: {stats.get('sources', [])}")
    
    print("\n📋 Testing: List sources...")
    sources = store.list_sources()
    print(f"Sources: {sources}")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_vector_store()