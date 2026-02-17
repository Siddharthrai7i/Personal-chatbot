import google.generativeai as genai
from typing import List, Optional
import time


class EmbeddingGenerator:
    """
    Generates embeddings using Google's Gemini API
    """
    
    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        """
        Initialize the embedding generator
        
        Args:
            api_key: Google API key
            model_name: Name of the embedding model
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Google API
        genai.configure(api_key=api_key)
        
        print(f"✅ Embedding generator initialized")
        print(f"📊 Model: {model_name}")
    
    def generate_embedding(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            retry_count: Number of retries on failure
            
        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            print("  Empty text provided")
            return None
        
        for attempt in range(retry_count):
            try:
                # Generate embedding
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                
                embedding = result['embedding']
                
                # Verify embedding dimensions
                if len(embedding) != 3072:
                    print(f"⚠️  Unexpected embedding dimension: {len(embedding)}")
                
                return embedding
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1}/{retry_count} failed: {str(e)}")
                
                if attempt < retry_count - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt
                    print(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print("❌ All retry attempts failed")
                    return None
        
        return None
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress
            
        Returns:
            List of embedding vectors (None for failed ones)
        """
        embeddings = []
        total = len(texts)
        
        print(f"\n🔄 Generating embeddings for {total} texts...")
        
        for idx, text in enumerate(texts, 1):
            if show_progress and idx % 5 == 0:
                print(f"   Progress: {idx}/{total}")
            
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
            
            # Small delay to avoid rate limiting
            if idx < total:
                time.sleep(0.1)
        
        successful = sum(1 for e in embeddings if e is not None)
        print(f"✅ Generated {successful}/{total} embeddings successfully\n")
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for a query (optimized for search)
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector or None
        """
        if not query or not query.strip():
            print("⚠️  Empty query provided")
            return None
        
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"  # Optimized for queries
            )
            
            return result['embedding']
            
        except Exception as e:
            print(f"❌ Error generating query embedding: {str(e)}")
            return None


# Convenience functions
def create_embedding(text: str, api_key: str) -> Optional[List[float]]:
    """
    Quick function to create a single embedding
    
    Args:
        text: Text to embed
        api_key: Google API key
        
    Returns:
        Embedding vector or None
    """
    generator = EmbeddingGenerator(api_key)
    return generator.generate_embedding(text)


# Test function
def test_embeddings():
    """Test the embedding generator"""
    print("\n" + "="*60)
    print("🧪 TESTING EMBEDDING GENERATOR")
    print("="*60)
    
    # Load API key
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file")
        return
    
    # Initialize generator
    generator = EmbeddingGenerator(api_key=api_key)
    
    # Test 1: Single embedding
    print("\n📝 Test 1: Generate single embedding...")
    text = "I am a software engineer with 5 years of experience."
    embedding = generator.generate_embedding(text)
    
    if embedding:
        print(f"    Success!")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
    else:
        print("❌ Failed to generate embedding")
        return
    
    # Test 2: Batch embeddings
    print("\n📝 Test 2: Generate batch embeddings...")
    texts = [
        "I love playing guitar.",
        "I enjoy photography.",
        "I like hiking in nature."
    ]
    embeddings = generator.generate_embeddings_batch(texts)
    
    successful = sum(1 for e in embeddings if e is not None)
    print(f"Generated {successful}/{len(texts)} embeddings")
    
    # Test 3: Query embedding
    print("\n📝 Test 3: Generate query embedding...")
    query = "What are your hobbies?"
    query_embedding = generator.generate_query_embedding(query)
    
    if query_embedding:
        print(f"✅ Query embedding generated")
        print(f"   Dimension: {len(query_embedding)}")
    
    # Test 4: Similarity comparison
    if embedding and query_embedding:
        print("\n📝 Test 4: Compare similarity...")
        
        # Calculate cosine similarity (simple version)
        import math
        
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = math.sqrt(sum(x * x for x in a))
            magnitude_b = math.sqrt(sum(x * x for x in b))
            return dot_product / (magnitude_a * magnitude_b)
        
        # Compare work-related text with hobby query
        similarity = cosine_similarity(embedding, query_embedding)
        print(f"   Similarity between work text and hobby query: {similarity:.4f}")
        print(f"   (0 = completely different, 1 = identical)")
    
    print("\n" + "="*60)
    print("✅ All embedding tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_embeddings()