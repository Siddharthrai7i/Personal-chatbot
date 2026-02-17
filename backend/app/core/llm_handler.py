import google.generativeai as genai
from typing import Optional, List, Dict
import time


class LLMHandler:
    """
    Handles interactions with Google's Gemini LLM for generating responses
    """
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7
    ):
        """
        Initialize the LLM handler
        
        Args:
            api_key: Google API key
            model_name: Name of the Gemini model
            temperature: Temperature for generation (0-1, higher = more creative)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        
        # Configure Google API
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        
        print(f"✅ LLM handler initialized")
        print(f"🤖 Model: {model_name}")
        print(f"🌡️  Temperature: {temperature}")
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        retry_count: int = 3
    ) -> Optional[str]:
        """
        Generate a response from the LLM
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum tokens in response
            retry_count: Number of retries on failure
            
        Returns:
            Generated text or None if failed
        """
        if not prompt or not prompt.strip():
            print("⚠️  Empty prompt provided")
            return None
        
        for attempt in range(retry_count):
            try:
                # Generate response
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=max_tokens,
                    )
                )
                
                # Extract text from response
                if response.text:
                    return response.text.strip()
                else:
                    print("⚠️  Empty response from model")
                    return None
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1}/{retry_count} failed: {str(e)}")
                
                if attempt < retry_count - 1:
                    # Wait before retrying
                    wait_time = 2 ** attempt
                    print(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print("❌ All retry attempts failed")
                    return None
        
        return None
    
    def generate_rag_response(
        self,
        query: str,
        context_chunks: List[str],
        max_tokens: int = 1000
    ) -> Optional[str]:
        """
        Generate a response using RAG (Retrieval-Augmented Generation)
        """
        if not context_chunks:
            print("⚠️  No context chunks provided")
            return "I don't have enough information to answer that question."
        
        # Extract name from context if available
        context_text = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Build the improved prompt
        prompt = f"""You are a friendly AI assistant representing someone based on their personal information.
        Never reply with more than 50 words 

CONTEXT INFORMATION:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. IDENTITY & GREETINGS:
   - If asked "who are you" or "tell me about yourself", identify yourself as the virtual AI assistant of the person whose information is in the context
   - Extract the person's name from the context and use it naturally (e.g., "I'm the AI assistant for [Name]")
   - For greetings (hi, hello, hey), respond warmly and offer to help with questions about the person

2. ANSWERING STYLE:
   - Be warm, friendly, and conversational - like a helpful friend
   - Use enthusiastic and positive language
   - Give comprehensive answers with relevant details from the context
   - Structure longer answers with natural flow (not bullet points unless asked)
   - Use first-person perspective when talking about the person's information (e.g., "I have 5 years of experience" instead of "They have")

3. ACCURACY RULES:
   - Answer ONLY based on information in the context above
   - If the context doesn't contain the answer, say: "I don't have that specific information in my knowledge base. Feel free to ask me something else!"
   - Never make up or assume information not present in the context
   - Never mention "context", "documents", "chunks", or technical terms

4. RESPONSE GUIDELINES:
   - Keep answers natural and conversational
   - Be specific with examples and details when available
   - For vague questions, provide a helpful overview and invite follow-up questions
   - Match the tone to the question (professional for work questions, casual for personal questions)

5. SPECIAL CASES:
   - Questions about contact/social media: Only share if explicitly mentioned in context
   - Questions about availability/hiring: Respond positively but mention they should reach out directly
   - Unrelated questions: Politely redirect to topics you can help with

Now answer the user's question in a friendly, natural, and helpful way:"""
        
        return self.generate_response(prompt, max_tokens=max_tokens)


# Convenience function
def generate_answer(
    query: str,
    context_chunks: List[str],
    api_key: str
) -> Optional[str]:
    """
    Quick function to generate an answer
    
    Args:
        query: User question
        context_chunks: Retrieved context
        api_key: Google API key
        
    Returns:
        Generated answer or None
    """
    handler = LLMHandler(api_key)
    return handler.generate_rag_response(query, context_chunks)


# Test function
def test_llm_handler():
    """Test the LLM handler"""
    print("\n" + "="*60)
    print("🧪 TESTING LLM HANDLER")
    print("="*60)
    
    # Load API key
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file")
        return
    
    # Initialize handler
    handler = LLMHandler(api_key=api_key)
    
    # Test 1: Simple generation
    print("\n📝 Test 1: Simple text generation...")
    prompt = "Say hello in a friendly way."
    response = handler.generate_response(prompt, max_tokens=100)
    
    if response:
        print(f"✅ Response: {response}")
    else:
        print("❌ Failed to generate response")
        return
    
    # Test 2: RAG response
    print("\n📝 Test 2: RAG-based response...")
    
    query = "What are my hobbies?"
    context_chunks = [
        "I love playing guitar. I've been playing for 10 years.",
        "Photography is another passion of mine. I enjoy landscape photography.",
        "I also enjoy hiking on weekends in the mountains."
    ]
    
    rag_response = handler.generate_rag_response(query, context_chunks)
    
    if rag_response:
        print(f"✅ Question: {query}")
        print(f"✅ Answer: {rag_response}")
    else:
        print("❌ Failed to generate RAG response")
    
    # Test 3: RAG with no relevant context
    print("\n📝 Test 3: RAG with irrelevant context...")
    
    query = "What programming languages do you know?"
    context_chunks = [
        "I love playing guitar.",
        "Photography is my hobby."
    ]
    
    rag_response = handler.generate_rag_response(query, context_chunks)
    
    if rag_response:
        print(f"✅ Question: {query}")
        print(f"✅ Answer: {rag_response}")
    
    print("\n" + "="*60)
    print("✅ All LLM tests completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_llm_handler()