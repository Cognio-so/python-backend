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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Debug: Print API keys
logger.info("API Keys loaded successfully")

# Configure Gemini AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY').strip())

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
    if model_name == "gemini-1.5-flash":
        return genai.GenerativeModel('gemini-1.5-flash')
    elif model_name == "gpt-4o-mini":
        return openai_client
    elif model_name == "claude-3-haiku-20240307":
        return claude_client
    elif model_name == "llama-v3-7b":  # Updated model name
        return fireworks
    else:
        raise ValueError(f"Unsupported model: {model_name}")

async def generate_response(messages, model_name, session_id=None):
    try:
        logger.info(f"Generating response with model: {model_name}, session_id: {session_id}")
        model = get_model_instance(model_name)
        memory = get_or_create_memory(session_id)
        language = 'en-US'
        for msg in messages:
            if msg['role'] == 'system':
                if 'Respond in' in msg['content']:
                    language = msg['content'].split('Respond in')[1].split('.')[0].strip()
                    break

        buffer = ""
        MIN_CHUNK_SIZE = 50
        PUNCTUATION_MARKS = ['.', '!', '?', '\n']
        
        if model_name == "gemini-1.5-flash":
            prompt = f"You must respond in {language}. Maintain the same language throughout the response. "
            if 'hi' in language.lower():
                prompt += "Use Hindi script (Devanagari) for Hindi responses. "
            
            if memory:
                for msg in memory.chat_memory.messages:
                    if isinstance(msg, HumanMessage):
                        prompt += f"\nUser: {msg.content}"
                    elif isinstance(msg, AIMessage):
                        prompt += f"\nAssistant: {msg.content}"
            
            prompt += f"\nUser: {messages[-1]['content']}\nAssistant:"
            
            response = model.generate_content(prompt)
            if memory:
                memory.chat_memory.add_user_message(messages[-1]['content'])
                memory.chat_memory.add_ai_message(response.text)
            yield response.text

        elif model_name == "gpt-4o-mini":
            lang_message = {
                "role": "system",
                "content": f"You must respond in {language}. If the user speaks in Hindi, respond in Hindi using Devanagari script. Maintain consistent language throughout."
            }
            messages.insert(0, lang_message)
            
            response = await model.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    buffer += chunk.choices[0].delta.content
                    if any(buffer.endswith(p) for p in PUNCTUATION_MARKS) and len(buffer.strip()) >= MIN_CHUNK_SIZE:
                        yield buffer
                        buffer = ""
            if buffer.strip():
                yield buffer

        elif model_name == "claude-3-haiku-20240307":
            response = await model.messages.create(
                model="claude-3-haiku-20240307",
                messages=messages,
                max_tokens=1000,
                stream=True
            )
            
            async for chunk in response:
                # Handle different event types from Anthropic streaming
                if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                    buffer += chunk.delta.text
                    if any(buffer.endswith(p) for p in PUNCTUATION_MARKS) and len(buffer.strip()) >= MIN_CHUNK_SIZE:
                        yield buffer
                        buffer = ""
                elif chunk.type == "message_delta" or chunk.type == "message_stop":
                    # End of message, yield remaining buffer if any
                    if buffer.strip():
                        yield buffer
                        buffer = ""

        elif model_name == "llama-v3-7b":
            response = model.ChatCompletion.create(
                model="accounts/fireworks/models/llama-v3p1-8b-instruct",  # Updated model ID
                messages=messages,
                stream=True
            )
            
            # Handle non-async streaming for Fireworks
            for chunk in response:
                if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                    buffer += chunk.choices[0].delta.content
                    if any(buffer.endswith(p) for p in PUNCTUATION_MARKS) and len(buffer.strip()) >= MIN_CHUNK_SIZE:
                        yield buffer
                        buffer = ""
            if buffer.strip():
                yield buffer

        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        yield f"I apologize, but I encountered an error: {str(e)}"

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