import os
from dotenv import load_dotenv
import logging
import google.generativeai as genai
from openai import AsyncOpenAI
import anthropic
import fireworks.client as fireworks
import groq
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Debug: Print API keys
logger.debug(f"Google API Key: {os.getenv('GOOGLE_API_KEY')}")
logger.debug(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')}")
logger.debug(f"Anthropic API Key: {os.getenv('ANTHROPIC_API_KEY')}")
logger.debug(f"Fireworks API Key: {os.getenv('FIREWORKS_API_KEY')}")
logger.debug(f"Groq API Key: {os.getenv('GROQ_API_KEY')}")

# Configure Gemini AI
try:
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY', '').strip())
    # Initialize default model once
    default_gemini = genai.GenerativeModel('gemini-1.0-pro')  # Use correct model name
except Exception as e:
    logger.error(f"Failed to configure Gemini: {str(e)}")

# Configure OpenAI (GPT)
openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY').strip())

# Configure Anthropic (Claude)
claude_client = anthropic.AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY').strip())

# Configure Fireworks
fireworks.api_key = os.getenv('FIREWORKS_API_KEY').strip()

# Configure Groq
groq_client = groq.Client(api_key=os.getenv('GROQ_API_KEY').strip())

# Add a global conversation memory dictionary to store memories for different sessions
conversation_memories = {}

# Define model mappings
MODEL_MAPPINGS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gemini-flash-2.0": "gemini-1.5-flash",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "llama-3.3": "accounts/fireworks/models/llama-v2-70b"
}

def get_or_create_memory(session_id):
    """Get or create a conversation memory for a session."""
    if not session_id:
        return None
    if session_id not in conversation_memories:
        conversation_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return conversation_memories[session_id]

def get_model_instance(model_name):
    """Get the appropriate model client based on the model name."""
    # Normalize model name using mappings if present
    model_name = MODEL_MAPPINGS.get(model_name, model_name)
    
    logger.debug(f"Getting model instance for: {model_name}")
    
    if model_name.startswith("gemini"):
        return genai.GenerativeModel(model_name)
    elif model_name in ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"]:
        return openai_client
    elif model_name in ["claude-3-haiku", "claude-3-opus", "claude-3-sonnet"]:
        return claude_client
    elif "llama" in model_name:
        return fireworks
    elif model_name.startswith("groq"):
        return groq_client
    else:
        logger.warning(f"Unknown model: {model_name}, defaulting to gemini-pro")
        return genai.GenerativeModel('gemini-pro')

# Add this helper function
async def handle_claude_response(model, messages, max_tokens=1000):
    @retry(
        stop=stop_after_attempt(3),  # Try 3 times
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Wait between attempts
        retry=retry_if_exception_type((Exception))
    )
    async def _make_request():
        try:
            return await claude_client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=True
            )
        except Exception as e:
            if "overloaded" in str(e).lower():
                logger.warning(f"Claude API overloaded, retrying... {str(e)}")
                await asyncio.sleep(2)  # Wait 2 seconds before retry
                raise  # Raise to trigger retry
            raise  # Re-raise other exceptions

    return await _make_request()

async def generate_response(messages, model, session_id):
    """Generate a streaming response based on the model selected."""
    try:
        memory = get_or_create_memory(session_id)
        current_message = messages[-1] if messages else None
        
        if not current_message:
            logger.warning("No current message found")
            yield "Error: No message provided"
            return
            
        content = current_message["content"].strip()
        if not content:
            logger.warning("Empty message content received")
            yield "Error: Empty message content"
            return
        
        # Normalize model name using mappings if present
        mapped_model = MODEL_MAPPINGS.get(model, model)
        logger.info(f"Generating response with model: {model} (mapped to: {mapped_model})")
        
        # Handle Gemini models
        if mapped_model in ["gemini-pro", "gemini-1.5-flash"]:
            try:
                # Use correct model name for Gemini
                model_name = "gemini-1.0-pro" if mapped_model == "gemini-pro" else mapped_model
                gemini_model = genai.GenerativeModel(model_name)
                response = gemini_model.generate_content(content, stream=True)
                for chunk in response:
                    if hasattr(chunk, 'text'):
                        yield chunk.text
            except Exception as e:
                logger.error(f"Gemini model error: {str(e)}")
                yield f"Error with Gemini model: {str(e)}"

        # Handle Claude models with retries
        elif mapped_model == "claude-3-5-haiku-20241022":
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = await claude_client.messages.create(
                        model=mapped_model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=1000,
                        stream=True
                    )
                    async for chunk in response:
                        if chunk.delta.text:
                            yield chunk.delta.text
                    break  # If successful, break the retry loop
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Claude attempt {retry_count} failed: {str(e)}")
                    if "overloaded" in str(e).lower():
                        if retry_count < max_retries:
                            await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                            continue
                    # If all retries failed or it's not an overload error, try GPT-4
                    logger.info("Falling back to GPT-4 model")
                    try:
                        completion = await openai_client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": content}],
                            stream=True,
                            max_tokens=1000
                        )
                        async for chunk in completion:
                            if chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content
                        break
                    except Exception as fallback_e:
                        logger.error(f"Fallback GPT-4 error: {str(fallback_e)}")
                        yield f"Error: All models failed. Please try again later."
                        break

        # Handle Llama models
        elif "llama" in mapped_model:
            try:
                response = fireworks.ChatCompletion.create(
                    model="accounts/fireworks/models/llama-v2-70b",
                    messages=[{"role": "user", "content": content}],
                    stream=True,
                    max_tokens=1000
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                logger.error(f"Llama model error: {str(e)}")
                yield f"Error with Llama model: {str(e)}"
                
        # Handle unknown models with fallback to GPT-4
        else:
            logger.warning(f"Unsupported model: {model}, falling back to GPT-4")
            try:
                completion = await openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": content}],
                    stream=True,
                    max_tokens=1000
                )
                async for chunk in completion:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                logger.error(f"Fallback model error: {str(e)}")
                yield f"Error with fallback model: {str(e)}"
        
        if memory and not isinstance(content, Exception):
            try:
                memory.chat_memory.add_user_message(content)
            except Exception as e:
                logger.error(f"Error updating memory: {str(e)}")
                
    except Exception as e:
        logger.error(f"General error in generate_response: {str(e)}")
        yield f"Error: {str(e)}"

async def generate_related_questions(message: str, model_name: str) -> list:
    try:
        model = get_model_instance(model_name)
        prompt = f"Based on this message: '{message}', generate 3 related follow-up questions. Return them as a simple array of strings."
        
        if model_name.startswith("gemini"):
            response = model.generate_content(prompt)
            text_response = response.text
        elif model_name.startswith("gpt"):
            response = await model.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            text_response = response.choices[0].message.content
        elif model_name.startswith("claude"):
            response = await model.messages.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            text_response = response.content[0].text
        elif model_name.startswith("fireworks"):
            response = model.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            text_response = response.choices[0].message.content
        elif model_name.startswith("groq"):
            response = model.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            text_response = response.choices[0].message.content
        elif model_name == "gemini-1.5-flash":
            response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
            text_response = response.text
        elif model_name == "claude-3-5-haiku-20241022":
            response = await claude_client.messages.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            text_response = response.content[0].text
        elif model_name == "accounts/fireworks/models/llama-v2-70b":
            response = fireworks.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            text_response = response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Parse the response into an array
        questions = [
            q.strip() for q in text_response.split('\n') 
            if q.strip() and not q.strip().startswith('[') and not q.strip().startswith(']')
        ][:3]

        return questions

    except Exception as e:
        logger.error(f"Error generating related questions: {str(e)}")
        return []