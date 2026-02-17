from app.core.document_processor import DocumentProcessor
from pathlib import Path

# Initialize processor
processor = DocumentProcessor()

BASE_DIR = Path(__file__).resolve().parent.parent

# Test with a real file
# Replace with path to any PDF or TXT file you have
# test_file = Path("../Data/uploads/test.txt")  # Change this path
test_file = BASE_DIR / "Data" / "uploads" / "test.txt"
# print(test_file)
# Create a sample test file if it doesn't exist
if not test_file.exists():
    test_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
        Hello! My name is Siddharth Rai .
        I am a software engineer with 5 years of experience.
        I love building AI applications and working with Python.
        
        My hobbies include:
        - Playing guitar
        - Photography
        - Hiking
        Feel free to ask me anything about my background!
        """)
    print(f"✅ Created test file: {test_file}")

# Validate file
is_valid, message = processor.validate_file(str(test_file))
print(f"\n📋 Validation: {message}")

if is_valid:
    # Process the file
    text = processor.process_file(str(test_file))
    
    if text:
        print(f"\n✅ Successfully extracted text!")
        print(f"📊 Text length: {len(text)} characters")
        print(f"\n📄 First 200 characters:\n{text[:200]}...")
    else:
        print("\n❌ Failed to extract text")
else:
    print(f"\n❌ File validation failed: {message}")