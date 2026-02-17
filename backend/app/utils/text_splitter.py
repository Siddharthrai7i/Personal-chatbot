import re
from typing import List, Dict
from datetime import datetime


class TextSplitter:
    """
    Splits text into chunks with overlap for better context preservation
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text splitter
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def split_text(
        self, 
        text: str, 
        source_file: str = "unknown",
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Split text into chunks with metadata
        
        Args:
            text: Text to split
            source_file: Name of source file
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        if not text or not text.strip():
            print("⚠️  Empty text provided")
            return []
        
        # Clean text first
        text = self._prepare_text(text)
        
        # Split into chunks
        chunks = self._create_chunks(text)
        
        # Add metadata to each chunk
        chunk_dicts = []
        for idx, chunk in enumerate(chunks):
            chunk_dict = {
                'text': chunk,
                'chunk_index': idx,
                'source_file': source_file,
                'chunk_size': len(chunk),
                'created_at': datetime.now().isoformat(),
            }
            
            # Add any additional metadata
            if metadata:
                chunk_dict.update(metadata)
            
            chunk_dicts.append(chunk_dict)
        
        print(f"✂️  Split into {len(chunk_dicts)} chunks from {source_file}")
        return chunk_dicts
    
    def _prepare_text(self, text: str) -> str:
        """
        Prepare text for chunking
        
        Args:
            text: Raw text
            
        Returns:
            Prepared text
        """
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Create overlapping chunks from text - SIMPLE & RELIABLE VERSION
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            # Don't go beyond text length
            if end > text_length:
                end = text_length
            
            # Extract chunk
            chunk = text[start:end].strip()
            
            # Only add non-empty chunks
            if chunk:
                chunks.append(chunk)
            
            # Move start forward (with overlap)
            # New start = current end - overlap
            start = end - self.chunk_overlap
            
            # Safety check: if we've reached the end, stop
            if end >= text_length:
                break
            
            # Safety check: ensure we're making progress
            if start <= (end - self.chunk_size):
                start = end
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about the chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [chunk['chunk_size'] for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_sizes),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes)
        }


# Convenience function
def split_document(
    text: str,
    source_file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Dict = None
) -> List[Dict]:
    """
    Quick function to split a document
    
    Args:
        text: Text to split
        source_file: Source filename
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        metadata: Additional metadata
        
    Returns:
        List of chunks with metadata
    """
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text, source_file, metadata)


# Test function
def test_text_splitter():
    """Test the text splitter"""
    print("\n" + "="*60)
    print("🧪 TESTING TEXT SPLITTER")
    print("="*60)
    
    # Create sample text
    sample_text = """
    My name is John Doe and I'm a software engineer. I have been working in the tech industry for over 5 years. 
    I specialize in building AI applications and machine learning systems.
    
    My educational background includes a Bachelor's degree in Computer Science from MIT. 
    During my time at university, I focused on artificial intelligence and natural language processing.
    
    In my free time, I enjoy playing guitar and photography. I've been playing guitar for 10 years and love 
    acoustic music. Photography became a passion of mine during travels to various countries.
    """
    
    # Initialize splitter with small chunks for testing
    splitter = TextSplitter(chunk_size=200, chunk_overlap=50)
    
    # Split the text
    chunks = splitter.split_text(
        text=sample_text,
        source_file="about_me.txt",
        metadata={'category': 'personal_info'}
    )
    
    # Print results
    print(f"\n📊 Created {len(chunks)} chunks\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"{'='*60}")
        print(f"Chunk {i}/{len(chunks)} - Size: {chunk['chunk_size']} characters")
        print(f"{'='*60}")
        preview = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
        print(preview)
        print()
    
    # Get statistics
    stats = splitter.get_chunk_stats(chunks)
    print(f"\n📈 STATISTICS:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total characters: {stats['total_characters']}")
    print(f"Average chunk size: {stats['avg_chunk_size']:.0f}")
    print(f"Min chunk size: {stats['min_chunk_size']}")
    print(f"Max chunk size: {stats['max_chunk_size']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_text_splitter()