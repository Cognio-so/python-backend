import asyncio
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import replicate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence

# Load environment variables
load_dotenv()

# Set Replicate API token - using the exact environment variable name expected by the library
api_key = os.getenv("REPLICATE_API_KEY")
if api_key:
    os.environ["REPLICATE_API_TOKEN"] = api_key

# --- Logging Setup ---
# Configures logging to capture info and error messages for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- State Schema ---
# Defines the structure of the application state, including all necessary fields
class VaaniState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]  # List of conversation messages
    summary: str  # Summary of the conversation
    file_url: str  # URL of an attached file, if any
    web_search_enabled: bool  # Flag to enable/disable web search
    deep_research: bool  # Flag for deep research mode
    agent_name: str  # Name of the agent to route to
    extra_question: str  # Additional questions generated based on user input
    user_token: str  # User token for session management and authentication

# --- Input Validation ---
def validate_user_input(user_input: str) -> str:
    """Sanitizes user input to prevent injection attacks by allowing only safe characters."""
    return ''.join(c for c in user_input if c.isalnum() or c in (' ', '.', ',', '?', '!'))

# --- Authentication Placeholder ---
def authenticate_user(token: str) -> bool:
    """Checks user token validity. Replace with actual authentication logic (e.g., JWT)."""
    return token == "valid_token"  # Simplified example for demonstration

# --- Checkpointing with Memory Saver ---
# Sets up memory saver for state persistence instead of SQLite
memory = MemorySaver()

# --- System Prompt for Orchestrator ---
# Defines the logic for routing queries to appropriate agents
system_prompt = """
You are an orchestrator agent. Based on the user's query and context, decide which agent to route the query to.
Possible agents are:
- image_generator: if the user asks to generate an image
- rag_agent: if a file is attached (file_url is set) and the query is related to the file
- web_search_agent: if web_search_enabled is true and the query requires web search
- default: for general queries
Output only the name of the agent to route to.
"""

# --- Node Functions ---

async def extra_questions_node(state: VaaniState) -> VaaniState:
    """Generates additional questions based on the user's last message."""
    last_message = state["messages"][-1]
    if last_message.type == "human":
        user_input = validate_user_input(last_message.content)
        llm = ChatGroq(model_name="Llama-3.1-8b-instant")  # Lightweight model for quick responses
        try:
            response = await llm.ainvoke(f"Generate additional questions based on: {user_input}")
            return {"extra_question": response.content}
        except Exception as e:
            logger.error(f"Error in extra_questions_node: {e}")
            return state  # Return unchanged state on error
    return state

async def orchestrator_node(state: VaaniState) -> VaaniState:
    """Routes the query to the appropriate agent based on context and user input."""
    messages = state["messages"]
    summary = state.get("summary", "")
    file_url = state.get("file_url", "")
    web_search_enabled = state.get("web_search_enabled", False)
    deep_research = state.get("deep_research", False)
    # Build context string for the orchestrator
    context = f"Summary: {summary}\nFile attached: {'yes' if file_url else 'no'}\nWeb search enabled: {web_search_enabled}\nDeep research mode: {deep_research}"
    user_query = validate_user_input(messages[-1].content if messages else "")
    prompt = f"{system_prompt}\n\nContext: {context}\nUser query: {user_query}"
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")  # Powerful model for decision-making
    try:
        response = await llm.ainvoke(prompt)
        return {"agent_name": response.content.strip()}
    except Exception as e:
        logger.error(f"Error in orchestrator_node: {e}")
        return {"agent_name": "default_agent"}  # Fallback to default agent on error

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def image_generator_node(state: VaaniState) -> VaaniState:
    """Generates an image based on the user's query with retry logic for API failures."""
    user_query = validate_user_input(state["messages"][-1].content)
    try:
        # Log the API key status (without revealing the full key)
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            logger.error("REPLICATE_API_TOKEN is not set in environment variables")
            return {"messages": [AIMessage(content="Error: Replicate API token is not configured properly. Please check your environment variables.")]}
        
        # Log that we're about to make the API call
        logger.info(f"Making Replicate API call with prompt: {user_query[:50]}...")
        
        # Use the exact format from the reference implementation
        input = {"prompt": user_query}
        
        try:
            # Run Replicate API call in a thread to avoid blocking the event loop
            output = await asyncio.to_thread(replicate.run, "black-forest-labs/flux-schnell", input=input)
            
            if not output:
                logger.warning("Replicate API returned empty output")
                return {"messages": [AIMessage(content="The image generation service returned an empty response. Please try again with a different prompt.")]}
            
            # Format the output URL
            image_url = output[0] if isinstance(output, list) else output
            response = f"Generated image: {image_url}"
            logger.info("Successfully generated image")
        except replicate.exceptions.ReplicateError as e:
            if "503" in str(e) or "Service Unavailable" in str(e):
                logger.error(f"Replicate service unavailable: {e}")
                return {"messages": [AIMessage(content="Sorry, the image generation service is temporarily unavailable. This is an issue with the external service, not your request. Please try again in a few minutes.")]}
            elif "401" in str(e) or "Unauthenticated" in str(e):
                logger.error(f"Replicate authentication error: {e}")
                return {"messages": [AIMessage(content="Error: Authentication failed with the image service. Please check your API key configuration.")]}
            else:
                logger.error(f"Replicate API error: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {"messages": [AIMessage(content=f"There was an error with the image generation service: {str(e)}. Please try again later.")]}
    except Exception as e:
        import traceback
        logger.error(f"Error generating image: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Provide a more helpful error message based on the exception
        if "503" in str(e) or "Service Unavailable" in str(e):
            response = "Sorry, the image generation service is temporarily unavailable. Please try again later."
        elif "401" in str(e) or "Unauthenticated" in str(e):
            response = "Error: Authentication failed with the image service. Please check your API key configuration."
        else:
            response = f"Error generating image: {str(e)}"
            
    return {"messages": [AIMessage(content=response)]}  # Return as a list containing one AIMessage

async def rag_agent_node(state: VaaniState) -> VaaniState:
    """Processes file-based queries using aRetrieval-Augmented Generation (RAG) approach."""
    file_url = state["file_url"]
    user_query = validate_user_input(state["messages"][-1].content)
    # Placeholder for actual RAG implementation
    # Steps would include:
    # 1. Parse file at file_url (e.g., PyPDF2 for PDFs)
    # 2. Extract and chunk text
    # 3. Embed chunks (e.g., with sentence-transformers)
    # 4. Store in Qdrant vector DB
    # 5. Query Qdrant with user_query
    # 6. Generate response from retrieved chunks
    response = f"Response based on file at {file_url}: [Placeholder]"
    return {"messages": [AIMessage(content=response)]}  # Return as a list containing one AIMessage

async def web_search_agent_node(state: VaaniState) -> VaaniState:
    """Performs a web search using Tavily and processes results with an LLM to generate a coherent answer."""
    messages = state["messages"]
    user_query = validate_user_input(messages[-1].content)
    deep_research = state.get("deep_research", False)
    
    try:
        # Adjust search parameters based on deep_research flag
        max_results = 8 if deep_research else 5
        search_depth = "advanced" if deep_research else "basic"
        
        # Perform web search with adjusted parameters
        tool = TavilySearchResults(max_results=max_results, search_depth=search_depth)
        search_results = tool.invoke(user_query)
        
        # Store source URLs for later inclusion
        sources = []
        
        # Format search results for the LLM
        formatted_results = "Web search results:\n\n"
        for i, result in enumerate(search_results, 1):
            sources.append(f"[{i}] {result['url']}")
            formatted_results += f"Source {i}: {result['url']}\n"
            formatted_results += f"Content: {result['content']}\n\n"
        
        # Adjust system prompt based on deep_research
        system_message = SystemMessage(content=f"""
You are a helpful assistant that answers questions based on web search results.
{
  "Be extremely thorough and comprehensive in your answer. Go into detail analyzing multiple perspectives from different sources."
  if deep_research else 
  "Provide a concise and helpful answer based on the search results."
}
Use the provided search results to answer the user's question.
If the search results don't contain enough information to answer the question, say so.

When using information from the search results, cite your sources using numbers in square brackets like [1], [2], etc.
At the end of your response, include a "Sources:" section that lists all the sources you referenced.

Here are the search results:
{formatted_results}
""")
        
        # Use a more powerful model for deep research if needed
        model_name = "llama-3.3-70b-versatile"  # Use same model for now
        
        # Get conversation history for context
        conversation_history = messages[-5:] if len(messages) > 1 else messages
        
        # Add the system message at the beginning
        prompt_messages = [system_message] + conversation_history
        
        # Process with LLM
        llm = ChatGroq(model_name=model_name)
        response = await llm.ainvoke(prompt_messages)
        
        # Check if the response already includes sources
        content = response.content
        if "Sources:" not in content and sources:
            # Append sources if not already included
            content += "\n\nSources:\n" + "\n".join(sources)
        
        # Return the processed response
        return {"messages": [AIMessage(content=content)]}
        
    except Exception as e:
        logger.error(f"Error in web_search_agent_node: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"messages": [AIMessage(content=f"I encountered an error while searching the web: {str(e)}. Let me try to answer based on my knowledge instead.")]}

async def default_agent_node(state: VaaniState) -> VaaniState:
    """Handles general queries with conversation context and summary."""
    messages = state["messages"]
    summary = state.get("summary", "")
    # Prepend summary as a system message if it exists
    if summary:
        messages = [SystemMessage(content=f"Summary: {summary}")] + messages
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")  # Versatile model for general queries
    try:
        response = await llm.ainvoke(messages)
        # Create a new AIMessage with the response content
        ai_message = AIMessage(content=response.content)
        # Return a dictionary with the messages key containing the AIMessage
        return {"messages": [ai_message]}
    except Exception as e:
        logger.error(f"Error in default_agent_node: {e}")
        return {"messages": [AIMessage(content="Error processing your request")]}

async def summarize_node(state: VaaniState) -> VaaniState:
    """Summarizes the conversation and keeps only the last 4 messages."""
    messages = state["messages"]
    summary = state.get("summary", "")
    summary_message = f"Summary so far: {summary}\nExtend with new messages:" if summary else "Summarize the conversation:"
    llm = ChatGroq(model_name="Llama-3.1-8b-instant")  # Placeholder until Gemini 2.0 Flash is integrated
    try:
        response = await llm.ainvoke(messages + [HumanMessage(content=summary_message)])
        # Remove all but the last 4 messages to manage state size
        delete_messages = [RemoveMessage(id=m.id) for m in messages[:-4]]
        return {"summary": response.content, "messages": delete_messages}
    except Exception as e:
        logger.error(f"Error in summarize_node: {e}")
        return state  # Return unchanged state on error

def summarize_check_node(state: VaaniState) -> VaaniState:
    """Passthrough node to check if summarization is needed."""
    return state

def should_summarize(state: VaaniState) -> str:
    """Decides if summarization is required based on the number of messages."""
    if len(state["messages"]) > 8:
        return "summarize"
    return "end"

# --- Workflow Setup ---
# Initializes the state graph for the Vaani.pro workflow
workflow = StateGraph(VaaniState)

# Add all nodes to the workflow
workflow.add_node("extra_questions", extra_questions_node)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("image_generator", image_generator_node)
workflow.add_node("rag_agent", rag_agent_node)
workflow.add_node("web_search_agent", web_search_agent_node)
workflow.add_node("default_agent", default_agent_node)
workflow.add_node("summarize_check", summarize_check_node)
workflow.add_node("summarize", summarize_node)

# Define workflow edges
workflow.set_entry_point("extra_questions")  # Start with generating extra questions
workflow.add_edge("extra_questions", "orchestrator")  # Route to orchestrator next

def route_to_agent(state: VaaniState) -> str:
    """Routes to the selected agent, falling back to 'default_agent' if invalid."""
    agent_name = state["agent_name"]
    valid_agents = {"image_generator", "rag_agent", "web_search_agent", "default_agent"}
    return agent_name if agent_name in valid_agents else "default_agent"

# Conditional routing from orchestrator to specific agents
workflow.add_conditional_edges("orchestrator", route_to_agent, {
    "image_generator": "image_generator",
    "rag_agent": "rag_agent",
    "web_search_agent": "web_search_agent",
    "default_agent": "default_agent"
})

# Connect all agents to summarize_check node
for agent in ["image_generator", "rag_agent", "web_search_agent", "default_agent"]:
    workflow.add_edge(agent, "summarize_check")

# Conditional edge to either summarize or end the workflow
workflow.add_conditional_edges("summarize_check", should_summarize, {
    "summarize": "summarize",
    "end": END
})
workflow.add_edge("summarize", END)  # End workflow after summarization

# Compile the Graph with checkpointing
graph = workflow.compile(checkpointer=memory)

# --- Example Usage ---
async def run_example():
    """Runs an example query through the workflow."""
    config = {"configurable": {"thread_id": "1"}}  # Thread ID for state persistence
    state_update = {
        "messages": [HumanMessage(content="Generate an image of a dog")],
        "summary": "",
        "file_url": "",
        "web_search_enabled": False,
        "deep_research": False,
        "agent_name": "",
        "extra_question": "",
        "user_token": "valid_token"  # Example token for authentication
    }
    # Authenticate user before processing
    if not authenticate_user(state_update["user_token"]):
        logger.error("Authentication failed")
        return
    # Run the workflow and print the final message
    output = await graph.ainvoke(state_update, config)
    for m in output["messages"][-1:]:
        m.pretty_print()

if __name__ == "__main__":
    asyncio.run(run_example())

# Export the graph and VaaniState for use in other modules
__all__ = ["graph", "VaaniState"]