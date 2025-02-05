import streamlit as st
from langchain_community.llms import Ollama
import uuid
import re

# Page configuration
st.set_page_config(page_title="Python Dev Assistant", page_icon="🐍")

# Initialize session state for unique chat ID and messages
if 'chat_id' not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'hide_tags' not in st.session_state:
    st.session_state.hide_tags = True

# System prompt tailored for Python development
SYSTEM_PROMPT = """You are an expert Python developer assistant. Your role is to:
- Provide high-quality, Pythonic code solutions
- Offer best practices and design patterns in Python
- Help debug and optimize Python code
- Give concise, clear explanations of Python concepts
- Suggest modern Python techniques and libraries
- Assist with code structure, performance, and readability
- Handle Python development across web, data science, automation, and more

When providing code:
- Use type hints
- Follow PEP 8 style guidelines
- Provide comments for complex logic
- Suggest potential improvements or alternative approaches
- Include error handling and input validation
- Recommend appropriate libraries and frameworks

Always prioritize clean, efficient, and maintainable Python code."""

# Initialize Ollama LLM
def get_ollama_llm():
    return Ollama(model="deepseek-r1:1.5b")

# Function to clean response
def clean_response(response):
    if st.session_state.hide_tags:
        # Remove <think> tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        # Remove any remaining XML-like tags
        response = re.sub(r'<[^>]+>', '', response)
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
    st.title("Python Development Assistant")
    
    # Tag filtering toggle
    st.session_state.hide_tags = st.checkbox(
        "Hide AI-generated tags", 
        value=st.session_state.hide_tags
    )
    
    # Display chat messages from history
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])
    
    # Chat input
    if user_input := st.chat_input("Ask about Python development"):
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