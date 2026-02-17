import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory - project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ==============================================
# GOOGLE GEMINI CONFIGURATION
# ==============================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")

# ==============================================
# DATABASE CONFIGURATION
# ==============================================
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "personal_knowledge")

# Convert to absolute path
CHROMA_DB_PATH = str(BASE_DIR / CHROMA_DB_PATH)

# ==============================================
# FILE STORAGE CONFIGURATION
# ==============================================
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
LOG_DIR = os.getenv("LOG_DIR", "./data/logs")

# Convert to absolute paths
UPLOAD_DIR = str(BASE_DIR / UPLOAD_DIR)
LOG_DIR = str(BASE_DIR / LOG_DIR)

# ==============================================
# CHUNKING CONFIGURATION
# ==============================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# ==============================================
# RETRIEVAL CONFIGURATION
# ==============================================
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))

# ==============================================
# SERVER CONFIGURATION
# ==============================================
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))

# ==============================================
# ALLOWED FILE TYPES
# ==============================================
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ==============================================
# CREATE DIRECTORIES IF THEY DON'T EXIST
# ==============================================
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [UPLOAD_DIR, LOG_DIR, CHROMA_DB_PATH]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✅ All required directories created/verified")


# ==============================================
# PRINT CONFIGURATION (FOR DEBUGGING)
# ==============================================
def print_config():
    """Print current configuration"""
    print("\n" + "="*50)
    print("📋 CONFIGURATION LOADED")
    print("="*50)
    print(f"🔑 API Key: {'✅ Set' if GOOGLE_API_KEY else '❌ Missing'}")
    print(f"🤖 LLM Model: {LLM_MODEL}")
    print(f"📊 Embedding Model: {EMBEDDING_MODEL}")
    print(f"💾 ChromaDB Path: {CHROMA_DB_PATH}")
    print(f"📁 Upload Directory: {UPLOAD_DIR}")
    print(f"📝 Log Directory: {LOG_DIR}")
    print(f"✂️  Chunk Size: {CHUNK_SIZE}")
    print(f"🔄 Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"🔍 Top K Results: {TOP_K_RESULTS}")
    print("="*50 + "\n")

# Initialize directories on import
ensure_directories()
print_config()