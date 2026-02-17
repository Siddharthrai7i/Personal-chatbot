import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("\n" + "="*60)
print("📋 AVAILABLE EMBEDDING MODELS")
print("="*60)

for model in genai.list_models():
    if 'embedContent' in model.supported_generation_methods:
        print(f"\n✅ {model.name}")
        print(f"   Description: {model.description}")

print("="*60 + "\n")