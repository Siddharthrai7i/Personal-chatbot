from app.core.vector_store import VectorStore
from app.config import CHROMA_DB_PATH, COLLECTION_NAME

print("\n⚠️  WARNING: This will delete ALL data in ChromaDB!")
confirm = input("Type 'yes' to continue: ")

if confirm.lower() == 'yes':
    store = VectorStore(db_path=CHROMA_DB_PATH, collection_name=COLLECTION_NAME)
    store.reset()
    print("✅ Database reset complete!")
else:
    print("❌ Reset cancelled")