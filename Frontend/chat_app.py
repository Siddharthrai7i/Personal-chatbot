import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
# BACKEND_URL = os.getenv("BACKEND_URL", "https://personal-chatbot-4.onrender.com")
BACKEND_URL= "https://personal-chatbot-4.onrender.com"
# Page config
st.set_page_config(
    page_title="Chat",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# WhatsApp-like CSS
st.markdown("""
    <style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Remove all Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background */
    .main {
        background: #0a0e27;
        padding: 0;
    }
    
    .block-container {
        padding: 1rem 1rem 5rem 1rem;
        max-width: 800px;
    }
    
    /* Header */
    .chat-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: #1a1f3a;
        padding: 1rem;
        z-index: 999;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .chat-header h3 {
        color: white;
        margin: 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .chat-header p {
        color: #8e9aaf;
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: transparent !important;
        padding: 0.5rem 0 !important;
        border: none !important;
    }
    
    /* User message bubble */
    [data-testid="stChatMessageContent"]:has(+ [data-testid="user-message"]),
    .stChatMessage[data-testid="user-message"] [data-testid="stChatMessageContent"] {
        background: #005c4b !important;
        color: white !important;
        padding: 0.8rem 1rem !important;
        border-radius: 10px !important;
        margin-left: auto !important;
        max-width: 70% !important;
        float: right !important;
    }
    
    
    /* Assistant message bubble */
    [data-testid="stChatMessageContent"]:has(+ [data-testid="assistant-message"]),
    .stChatMessage[data-testid="assistant-message"] [data-testid="stChatMessageContent"] {
        background: #1f2937 !important;
        color: white !important;
        padding: 0.8rem 1rem !important;
        border-radius: 10px !important;
        max-width: 70% !important;
        float: left !important;
    }
    
    /* Message text */
    .stChatMessage p {
        color: white !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        margin: 0 !important;
    }
    
    /* Input box */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #1a1f3a;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
        z-index: 999;
    }
    
    .stChatInput {
        background: #2d3548 !important;
        border: none !important;
        color: white !important;
    }
    
    textarea {
        background: #2d3548 !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 0.8rem 1.2rem !important;
        font-size: 1rem !important;
    }
    
    textarea::placeholder {
        color: #8e9aaf !important;
    }
    
    /* Hide avatars */
    .stChatMessage img {
        display: none !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #00a884 !important;
    }
    
    /* Error messages */
    .stAlert {
        background: #1f2937 !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }
    
    /* Welcome suggestions */
    .stButton button {
        background: #1f2937 !important;
        color: white !important;
        border: 1px solid #2d3548 !important;
        border-radius: 20px !important;
        padding: 0.6rem 1.2rem !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        background: #2d3548 !important;
        border-color: #00a884 !important;
    }
    
    /* Add top padding for fixed header */
    .main .block-container {
        padding-top: 6rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Fixed header
st.markdown("""
    <div class="chat-header">
        <h3>💬 Personal AI</h3>
        <p>Ask me anything</p>
    </div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

# Welcome suggestions
    st.markdown("#### 👋 Try asking:")

# Display chat messages - NO SOURCES
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type a message..."):
    # Hide welcome
    st.session_state.show_welcome = False
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        try:
            # Stream response from backend
            response = requests.post(
                f"{BACKEND_URL}/query/stream",
                json={"query": prompt, "top_k": 5},
                timeout=60,
                stream=True
            )
            
            if response.status_code == 200:
                # Generator that parses SSE events and yields text tokens
                def stream_tokens():
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith("data: "):
                            token = line[6:]  # Remove "data: " prefix
                            if token == "[DONE]":
                                return
                            # Skip JSON metadata events (sources info)
                            if token.startswith("{") and '"sources"' in token:
                                continue
                            yield token
                
                # Use st.write_stream for live typing effect
                full_response = st.write_stream(stream_tokens())
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
            else:
                st.error(f"Backend error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to backend at {BACKEND_URL}")
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

