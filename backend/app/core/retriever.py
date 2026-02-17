from typing import List, Dict, Optional
from app.core.vector_store import VectorStore
from app.core.embeddings import EmbeddingGenerator
from app.core.llm_handler import LLMHandler


class Retriever:
    """
    Retrieves relevant information and generates responses
    Combines VectorStore, Embeddings, and LLM
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        llm_handler: LLMHandler,
        top_k: int = 5
    ):
        """
        Initialize the retriever
        
        Args:
            vector_store: ChromaDB vector store
            embedding_generator: Embedding generator
            llm_handler: LLM handler for generating responses
            top_k: Number of chunks to retrieve
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.llm_handler = llm_handler
        self.top_k = top_k
        
        print("✅ Retriever initialized")
        print(f"🔍 Top-K: {top_k}")
    
    def retrieve_and_generate(
        self, 
        query: str,
        top_k: Optional[int] = None
    ) -> Dict:
        """
        Main RAG pipeline: Retrieve relevant chunks and generate answer
        
        Args:
            query: User's question
            top_k: Override default top_k if provided
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not query or not query.strip():
            return {
                'success': False,
                'answer': None,
                'error': 'Empty query provided',
                'sources': [],
                'chunks_used': 0
            }
        
        top_k = top_k or self.top_k
        
        print(f"\n{'='*60}")
        print(f"🔍 Processing query: {query}")
        print(f"{'='*60}")
        
        # Step 1: Generate query embedding
        print("\n📊 Step 1: Generating query embedding...") # embedding of the query(question whics is asked )
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        if not query_embedding:
            return {
                'success': False,
                'answer': None,
                'error': 'Failed to generate query embedding',
                'sources': [],
                'chunks_used': 0
            }
        
        print("✅ Query embedding generated")
        
        # Step 2: Search vector database
        print(f"\n🔎 Step 2: Searching database (top-{top_k})...")
        search_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        if search_results['count'] == 0:
            return {
                'success': True,
                'answer': "I don't have any information to answer that question. Please make sure you've uploaded relevant documents.",
                'sources': [],
                'chunks_used': 0
            }
        
        print(f"✅ Found {search_results['count']} relevant chunks")
        
        # Step 3: Extract chunks and metadata
        print("\n📝 Step 3: Extracting context...")
        chunks = search_results['documents']
        metadatas = search_results['metadatas']
        distances = search_results['distances']
        
        # Prepare sources information
        sources = []
        for i, (chunk, metadata, distance) in enumerate(zip(chunks, metadatas, distances)):
            sources.append({
                'chunk_text': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                'source_file': metadata.get('source_file', 'unknown'),
                'chunk_index': metadata.get('chunk_index', 0),
                'similarity_score': 1 - distance  # Convert distance to similarity
            })
        
        print(f"✅ Extracted {len(chunks)} chunks")
        
        # Step 4: Generate response using LLM
        print("\n🤖 Step 4: Generating response...")
        answer = self.llm_handler.generate_rag_response(
            query=query,
            context_chunks=chunks
        )
        
        if not answer:
            return {
                'success': False,
                'answer': None,
                'error': 'Failed to generate response',
                'sources': sources,
                'chunks_used': len(chunks)
            }
        
        print("✅ Response generated successfully")
        print(f"\n{'='*60}\n")
        
        return {
            'success': True,
            'answer': answer,
            'sources': sources,
            'chunks_used': len(chunks),
            'error': None
        }
    
    def retrieve_only(
        self, 
        query: str,
        top_k: Optional[int] = None
    ) -> Dict:
        """
        Only retrieve relevant chunks without generating response
        
        Args:
            query: User's question
            top_k: Override default top_k
            
        Returns:
            Dictionary with retrieved chunks
        """
        top_k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        if not query_embedding:
            return {
                'success': False,
                'chunks': [],
                'error': 'Failed to generate query embedding'
            }
        
        # Search
        search_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return {
            'success': True,
            'chunks': search_results['documents'],
            'metadatas': search_results['metadatas'],
            'distances': search_results['distances'],
            'count': search_results['count']
        }


# Test function
def test_retriever():
    """Test the retriever with sample data"""
    print("\n" + "="*60)
    print("🧪 TESTING RETRIEVER")
    print("="*60)
    
    # Load configuration
    import os
    from dotenv import load_dotenv
    from app.config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K_RESULTS
    
    load_dotenv(override=True)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found")
        return
    
    # Initialize components
    print("\n🔧 Initializing components...")
    
    vector_store = VectorStore(
        db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    embedding_generator = EmbeddingGenerator(api_key=api_key)
    llm_handler = LLMHandler(api_key=api_key)
    
    retriever = Retriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        llm_handler=llm_handler,
        top_k=TOP_K_RESULTS
    )
    
    # Add some test data if database is empty
    if vector_store.get_stats()['total_chunks'] == 0:
        print("\n📤 Database is empty. Adding test data...")
        
        test_chunks = [
            {
                'text': 'My name is John Doe. I am a software engineer with 5 years of experience in Python and AI.',
                'chunk_index': 0,
                'source_file': 'about_me.txt',
                'chunk_size': 95,
                'created_at': '2024-01-01T00:00:00'
            },
            {
                'text': 'I love playing guitar in my free time. I have been playing for 10 years and enjoy acoustic music.',
                'chunk_index': 1,
                'source_file': 'about_me.txt',
                'chunk_size': 98,
                'created_at': '2024-01-01T00:00:00'
            },
            {
                'text': 'Photography is another passion. I enjoy landscape and street photography, especially during travel.',
                'chunk_index': 2,
                'source_file': 'about_me.txt',
                'chunk_size': 102,
                'created_at': '2024-01-01T00:00:00'
            }
        ]
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in test_chunks]
        embeddings = embedding_generator.generate_embeddings_batch(texts, show_progress=False)
        
        # Add to database
        vector_store.add_documents(test_chunks, embeddings)
        print("✅ Test data added")
    
    # Test 1: Full RAG pipeline
    print("\n" + "="*60)
    print("📝 Test 1: Full RAG Pipeline")
    print("="*60)
    
    query = "What are your hobbies?"
    result = retriever.retrieve_and_generate(query)
    
    if result['success']:
        print(f"\n✅ SUCCESS!")
        print(f"\n❓ Question: {query}")
        print(f"\n💬 Answer:\n{result['answer']}")
        print(f"\n📚 Used {result['chunks_used']} chunks")
        print(f"\n📄 Sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"   {i}. {source['source_file']} (similarity: {source['similarity_score']:.2f})")
            print(f"      Preview: {source['chunk_text']}")
    else:
        print(f"\n❌ FAILED: {result['error']}")
    
    # Test 2: Retrieve only
    print("\n" + "="*60)
    print("📝 Test 2: Retrieve Only (No Generation)")
    print("="*60)
    
    query = "Tell me about your work experience"
    result = retriever.retrieve_only(query, top_k=2)
    
    if result['success']:
        print(f"\n✅ Found {result['count']} chunks")
        for i, chunk in enumerate(result['chunks'], 1):
            print(f"\n{i}. {chunk[:100]}...")
    else:
        print(f"\n❌ FAILED: {result['error']}")
    
    print("\n" + "="*60)
    print("✅ All retriever tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_retriever()