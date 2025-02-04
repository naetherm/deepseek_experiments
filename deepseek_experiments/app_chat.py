import streamlit as st
from langchain_community.llms import Ollama
import uuid
import re

# Page configuration
st.set_page_config(page_title="Deepseek-R1 Chat", page_icon="💬")

# Initialize session state for unique chat ID and messages
if 'chat_id' not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'hide_tags' not in st.session_state:
    st.session_state.hide_tags = True

# System prompt
SYSTEM_PROMPT = """You are a helpful, respectful, and honest AI assistant. 
Always answer as helpfully as possible while being safe and avoiding harmful content. 
If a question is unclear or lacks context, ask for clarification."""

# Initialize Ollama LLM
def get_ollama_llm():
    return Ollama(model="deepseek-r1:1.5b")

# Function to clean response
def clean_response(response):
    if st.session_state.hide_tags:
        # Remove <think> tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    return response.strip()

# Function to generate LLM response
def generate_response(messages, user_input):
    # Prepare the LLM input with system prompt and chat history
    llm = get_ollama_llm()
    
    # Construct full context including system prompt and chat history
    full_context = SYSTEM_PROMPT + "\n\n"
    
    # Add previous messages to context
    for msg in messages:
        full_context += f"{msg['role']}: {msg['content']}\n"
    
    # Add current user input
    full_context += f"Human: {user_input}\n"
    full_context += "Assistant:"
    
    # Generate response
    response = llm.invoke(full_context)
    return clean_response(response)

# Streamlit app layout
def main():
    st.title("Deepseek-R1 Chat Application")

    # Tag filtering toggle
    st.session_state.hide_tags = st.checkbox(
        "Hide AI-generated tags", 
        value=st.session_state.hide_tags
    )
    
    # Display chat messages from history
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])
    
    # Chat input
    if user_input := st.chat_input("Your message"):
        # Display user message
        st.chat_message("human").write(user_input)
        
        # Add user message to history
        st.session_state.messages.append({
            'role': 'human', 
            'content': user_input
        })
        
        # Generate and display AI response
        with st.chat_message("assistant"):
            response = generate_response(
                st.session_state.messages, 
                user_input
            )
            st.write(response)
        
        # Add AI response to history
        st.session_state.messages.append({
            'role': 'assistant', 
            'content': response
        })

# Run the app
if __name__ == "__main__":
    main()