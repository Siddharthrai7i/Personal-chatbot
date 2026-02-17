import os
import re
from pathlib import Path
from typing import Optional
import PyPDF2


"""
takes the file path and 
extract the text from there 

"""

class DocumentProcessor:
    """
    Processes documents (PDF, TXT) and extracts clean text
    """
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt'}
    
    def process_file(self, file_path: str) -> Optional[str]:
        """
        Main function to process any supported file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted and cleaned text, or None if failed
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            print(f" File not found: {file_path}")
            return None
        
        # Get file extension
        extension = file_path.suffix.lower()
        
        # Check if supported
        if extension not in self.supported_formats:
            print(f" Unsupported file format: {extension}")
            return None
        
        print(f"📄 Processing {extension} file: {file_path.name}")
        
        # Process based on file type
        if extension == '.pdf':
            text = self._extract_from_pdf(file_path)
        elif extension == '.txt':
            text = self._extract_from_txt(file_path)
        else:
            return None
        
        # Clean the extracted text
        if text:
            text = self._clean_text(text)
            print(f" Extracted {len(text)} characters from {file_path.name}")
            return text
        else:
            print(f"No text extracted from {file_path.name}")
            return None
    
    def _extract_from_pdf(self, file_path: Path) -> Optional[str]:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text or None
        """
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                print(f"📖 PDF has {num_pages} pages")
                
                # Extract text from each page
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text:
                        text += page_text + "\n\n"
                
                if not text.strip():
                    print("  PDF appears to be empty or scanned (no extractable text)")
                    return None
                
                return text
                
        except Exception as e:
            print(f" Error reading PDF: {str(e)}")
            return None
    
    def _extract_from_txt(self, file_path: Path) -> Optional[str]:
        """
        Extract text from TXT file
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            Extracted text or None
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    
                    print(f"✅ Successfully read with {encoding} encoding")
                    return text
                    
                except UnicodeDecodeError:
                    continue
            
            print(" Could not read file with any standard encoding")
            return None
            
        except Exception as e:
            print(f" Error reading TXT file: {str(e)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove special characters that might cause issues
        # Keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'[^\w\s\.,!?;:\-\(\)\[\]\'\"]+', '', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def validate_file(self, file_path: str, max_size_mb: int = 10) -> tuple[bool, str]:
        """
        Validate if file can be processed
        
        Args:
            file_path: Path to file
            max_size_mb: Maximum file size in MB
            
        Returns:
            (is_valid, error_message)
        """
        file_path = Path(file_path)
        
        # Check if exists
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check extension
        if file_path.suffix.lower() not in self.supported_formats:
            return False, f"Unsupported format. Supported: {', '.join(self.supported_formats)}"
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File too large ({file_size_mb:.2f}MB). Max size: {max_size_mb}MB"
        
        return True, "Valid"


# Convenience function for quick use
def extract_text_from_file(file_path: str) -> Optional[str]:
    """
    Quick function to extract text from a file
    
    Args:
        file_path: Path to file
        
    Returns:
        Extracted text or None
    """
    processor = DocumentProcessor()
    return processor.process_file(file_path)


# Test function
def test_document_processor():
    """Test the document processor with sample files"""
    processor = DocumentProcessor()
    
    print("\n" + "="*60)
    print("🧪 TESTING DOCUMENT PROCESSOR")
    print("="*60)
    
    # Test with a sample text
    test_text = """
    This    is   a    test   document.
    
    
    
    It has      multiple     spaces.
    
    And excessive newlines!
    """
    
    cleaned = processor._clean_text(test_text)
    print(f"\n📝 Original text length: {len(test_text)}")
    print(f"✨ Cleaned text length: {len(cleaned)}")
    print(f"\n🔍 Cleaned text:\n{cleaned}")
    print("\n" + "="*60)


if __name__ == "__main__":
    test_document_processor()