import streamlit as st
import asyncio
import os
import requests
from io import BytesIO
from dotenv import load_dotenv
import logging
import uuid
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent import VaaniState

# Import the graph directly from the module
import agent

# Load environment variables
load_dotenv()

# Set Replicate API token - using the exact environment variable name expected by the library
api_key = os.getenv("REPLICATE_API_KEY")
if api_key:
    os.environ["REPLICATE_API_TOKEN"] = api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Cognio Agent",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="stChatMessageContent"] {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stSidebar .block-container {
        padding-top: 2rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "file_url" not in st.session_state:
    st.session_state.file_url = ""

if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

# Sidebar for configuration
with st.sidebar:
    st.title("Cognio Agent")
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    
    st.subheader("Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a document for analysis", type=["pdf", "txt", "docx", "csv", "json"])
    if uploaded_file:
        # Save the file and get its path
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.file_url = file_path
        st.success(f"File uploaded: {uploaded_file.name}")
    
    # Feature toggles
    col1, col2 = st.columns(2)
    with col1:
        web_search_enabled = st.toggle("Web Search", value=True, help="Enable web search capabilities")
    with col2:
        deep_research = st.toggle("Deep Research", value=False, help="Enable more thorough analysis")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_summary = ""
        st.session_state.thread_id = str(uuid.uuid4())
        st.success("Conversation cleared!")
    
    st.divider()
    st.caption("Built by Ashutosh Upadhyay | Cognio Labs")

# Main chat interface
st.title("Cognio Agent Chat")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If this message contains an image, display it
        if message["role"] == "assistant" and "image_url" in message:
            try:
                image_url = message["image_url"]
                image_response = requests.get(image_url)
                image_response.raise_for_status()
                image_bytes = BytesIO(image_response.content)
                st.image(image_bytes, caption="Generated Image", use_column_width=True)
            except Exception as e:
                logger.error(f"Error displaying image in history: {e}")
                st.error(f"Could not display the image. You can view it at: {image_url}")

# Function to process messages through the agent
async def process_message(user_input):
    try:
        # Convert session messages to LangChain format for context
        langchain_messages = []
        
        # Add system message with summary if available
        if st.session_state.conversation_summary:
            langchain_messages.append(SystemMessage(content=f"Conversation summary: {st.session_state.conversation_summary}"))
        
        # Add recent conversation history (last 5 messages)
        for msg in st.session_state.messages[-5:]:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            else:
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        # Add the current user message
        langchain_messages.append(HumanMessage(content=user_input))
        
        # Prepare state for agent
        state = {
            "messages": langchain_messages,
            "summary": st.session_state.conversation_summary,
            "file_url": st.session_state.file_url,
            "web_search_enabled": web_search_enabled,
            "deep_research": deep_research,
            "agent_name": "",
            "extra_question": "",
            "user_token": "valid_token"  # Using the default token from agent.py
        }
        
        # Run the agent
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = await agent.graph.ainvoke(state, config)
        
        # Extract the response and update summary if provided
        response_content = "I couldn't process your request. Please try again."
        image_url = None
        
        # Completely rewritten response extraction logic
        if "messages" in result:
            # Get the last message which should be the response
            messages = result["messages"]
            if messages:
                last_message = messages[-1]  # Get the last message
                
                # Check if it's an AIMessage
                if isinstance(last_message, AIMessage):
                    content = last_message.content
                    
                    # Handle different content formats
                    if isinstance(content, str):
                        response_content = content
                        # Check if this is an image generation response
                        if "Generated image:" in response_content:
                            # Extract the image URL
                            image_url = response_content.split("Generated image:")[1].strip()
                            # Update the response to indicate we'll display the image
                            response_content = "Here's the generated image:"
                    elif isinstance(content, list):
                        # For structured content (like from Claude)
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and 'text' in part:
                                text_parts.append(part['text'])
                        if text_parts:
                            response_content = ' '.join(text_parts)
        
        # Update conversation summary if provided
        if "summary" in result and result["summary"]:
            st.session_state.conversation_summary = result["summary"]
        
        # Handle extra questions if provided
        if "extra_question" in result and result["extra_question"]:
            extra_question = result["extra_question"]
            if isinstance(extra_question, str):
                extra_questions = extra_question.split("\n")
                if extra_questions:
                    response_content += "\n\n**Follow-up questions you might consider:**\n"
                    for q in extra_questions[:3]:  # Limit to 3 suggestions
                        q = q.strip().strip('-').strip()
                        if q:
                            response_content += f"- {q}\n"
        
        return response_content, image_url
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        import traceback
        logger.error(traceback.format_exc())  # Log the full traceback for debugging
        return f"An error occurred: {str(e)}", None

# User input
user_input = st.chat_input("Ask me anything...")

# Process user input
if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Show thinking indicator
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🧠 Thinking...")
        
        # Process the message
        response, image_url = asyncio.run(process_message(user_input))
        
        # Replace thinking indicator with response
        thinking_placeholder.empty()
        st.markdown(response)
        
        # If there's an image URL, download and display it
        if image_url:
            try:
                # Download the image
                image_response = requests.get(image_url)
                image_response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Display the image
                image_bytes = BytesIO(image_response.content)
                st.image(image_bytes, caption="Generated Image", use_column_width=True)
                
                # Store both text and image URL in session state
                response_with_url = f"{response}\n\n[Image URL]({image_url})"
                st.session_state.messages.append({"role": "assistant", "content": response_with_url, "image_url": image_url})
            except Exception as e:
                logger.error(f"Error displaying image: {e}")
                st.error(f"Could not display the image. You can view it at: {image_url}")
                st.session_state.messages.append({"role": "assistant", "content": f"{response}\n\nImage URL: {image_url}"})
        else:
            # Add assistant response to chat history (text only)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Display current agent state in an expander (for debugging)
with st.expander("Debug Information", expanded=False):
    st.write("Thread ID:", st.session_state.thread_id)
    st.write("File URL:", st.session_state.file_url)
    st.write("Web Search:", web_search_enabled)
    st.write("Deep Research:", deep_research)
    
    if st.session_state.conversation_summary:
        st.subheader("Current Conversation Summary")
        st.info(st.session_state.conversation_summary) 