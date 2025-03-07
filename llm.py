import os
from dotenv import load_dotenv
import logging
import google.generativeai as genai
from openai import AsyncOpenAI
import anthropic
from groq import AsyncGroq
import groq
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Configure logging to DEBUG level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

logger.info("API Keys loaded successfully")

# Configure APIs
genai.configure(api_key=os.getenv('GOOGLE_API_KEY').strip())
openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY').strip())
claude_client = anthropic.AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY').strip())
groq_client = AsyncGroq(api_key=os.getenv('GROQ_API_KEY').strip())
groq_sync_client = groq.Client(api_key=os.getenv('GROQ_API_KEY').strip())

# Conversation memory
conversation_memories = {}

def get_or_create_memory(session_id):
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
    elif model_name == "llama-3.3-70b-versatile":
        return groq_client
    else:
        raise ValueError(f"Unsupported model: {model_name}")

async def generate_response(messages, model_name, session_id=None):
    try:
        logger.info(f"Generating response with model: {model_name}, session_id: {session_id}")
        model = get_model_instance(model_name)
        memory = get_or_create_memory(session_id)
        language = 'en-US'
        for msg in messages:
            if msg['role'] == 'system' and 'Respond in' in msg['content']:
                language = msg['content'].split('Respond in')[1].split('.')[0].strip()
                break

        buffer = ""
        MIN_CHUNK_SIZE = 20
        PUNCTUATION_MARKS = ['.', '!', '?', '\n']
        SPACE = ' '

        if model_name == "gemini-1.5-flash":
            prompt_parts = [f"You must respond in {language}. Maintain the same language throughout."]
            if memory and memory.chat_memory.messages:
                for msg in memory.chat_memory.messages:
                    prompt_parts.append(msg.content)
            prompt_parts.append(messages[-1]['content'])
            
            import asyncio
            # Prioritize non-streaming for reliability
            logger.debug("Starting Gemini response (non-streaming)")
            response = model.generate_content(prompt_parts, stream=False)
            full_text = response.text if hasattr(response, 'text') else str(response)
            logger.debug(f"Gemini full text: '{full_text[:100]}...' (length: {len(full_text)})")
            
            # Chunk and yield manually
            for i in range(0, len(full_text), MIN_CHUNK_SIZE):
                chunk = full_text[i:i + MIN_CHUNK_SIZE].strip()
                if chunk:
                    logger.debug(f"Yielding chunk: '{chunk}'")
                    yield chunk
                await asyncio.sleep(0.01)  # Slower delay for smooth frontend rendering
            logger.debug("Gemini response complete")

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
            logger.debug("Starting GPT-4o-mini streaming")
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    buffer += chunk.choices[0].delta.content
                    logger.debug(f"GPT chunk: '{chunk.choices[0].delta.content}'")
                    if any(buffer.endswith(p) for p in PUNCTUATION_MARKS) and len(buffer.strip()) >= MIN_CHUNK_SIZE:
                        yield buffer
                        logger.debug(f"Yielded: '{buffer}'")
                        buffer = ""
            if buffer.strip():
                logger.debug(f"GPT final buffer: '{buffer}'")
                yield buffer
            logger.debug("GPT-4o-mini streaming complete")

        elif model_name == "llama-3.3-70b-versatile":
            response = await model.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    *[{"role": msg["role"], "content": msg["content"]} for msg in messages]
                ],
                stream=True
            )
            logger.debug("Starting Llama streaming")
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    buffer += chunk.choices[0].delta.content
                    logger.debug(f"Llama chunk: '{chunk.choices[0].delta.content}'")
                    while SPACE in buffer or any(p in buffer for p in PUNCTUATION_MARKS):
                        next_space = buffer.find(SPACE)
                        next_punct = min([buffer.find(p) for p in PUNCTUATION_MARKS if p in buffer] + [len(buffer)])
                        split_point = min(next_space if next_space >= 0 else len(buffer), next_punct)
                        
                        if split_point > 0 and len(buffer[:split_point].strip()) >= MIN_CHUNK_SIZE:
                            yield buffer[:split_point].strip()
                            logger.debug(f"Yielded: '{buffer[:split_point].strip()}'")
                            buffer = buffer[split_point:].strip()
                        elif any(buffer.endswith(p) for p in PUNCTUATION_MARKS) and len(buffer.strip()) >= MIN_CHUNK_SIZE:
                            yield buffer.strip()
                            logger.debug(f"Yielded: '{buffer.strip()}'")
                            buffer = ""
                        else:
                            break
            if buffer.strip():
                logger.debug(f"Llama final buffer: '{buffer}'")
                yield buffer.strip()
            logger.debug("Llama streaming complete")

        elif model_name == "claude-3-haiku-20240307":
            system_prompt = next((msg['content'] for msg in messages if msg['role'] == 'system'), 
                               "You are a helpful assistant.")
            user_messages = []
            for msg in messages:
                if msg['role'] == 'user':
                    user_messages.append({"role": "user", "content": msg['content']})
                elif msg['role'] == 'assistant':
                    user_messages.append({"role": "assistant", "content": msg['content']})
            
            response = await model.messages.create(
                model=model_name,
                system=system_prompt,
                messages=user_messages,
                max_tokens=4096,
                stream=True
            )
            logger.debug("Starting Claude streaming")
            
            async for chunk in response:
                if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                    buffer += chunk.delta.text
                    logger.debug(f"Claude chunk: '{chunk.delta.text}'")
                    while len(buffer) > MIN_CHUNK_SIZE or any(p in buffer for p in PUNCTUATION_MARKS):
                        next_space = buffer.find(SPACE)
                        next_punct = min([buffer.find(p) for p in PUNCTUATION_MARKS if p in buffer] + [len(buffer)])
                        split_point = min(next_space if next_space >= 0 else len(buffer), next_punct)
                        
                        if split_point > 0:
                            yield buffer[:split_point].strip()
                            logger.debug(f"Yielded: '{buffer[:split_point].strip()}'")
                            buffer = buffer[split_point:].strip()
                        elif any(buffer.endswith(p) for p in PUNCTUATION_MARKS):
                            yield buffer.strip()
                            logger.debug(f"Yielded: '{buffer.strip()}'")
                            buffer = ""
                        else:
                            break
                elif chunk.type in ["content_block_stop", "message_stop"]:
                    if buffer.strip():
                        logger.debug(f"Claude stop event buffer: '{buffer}'")
                        yield buffer.strip()
                        buffer = ""
                    logger.debug(f"Claude chunk stop: {chunk.type}")
                elif chunk.type in ["message_start", "content_block_start"]:
                    logger.debug(f"Claude non-content chunk: {chunk.type}")
                    continue
                else:
                    logger.warning(f"Unhandled Claude chunk type: {chunk.type}")
            if buffer.strip():
                logger.debug(f"Claude final buffer: '{buffer}'")
                yield buffer.strip()
            logger.debug("Claude streaming complete")

        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
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
        elif model_name == "llama-3.3-70b-versatile":
            response = groq_sync_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            text_response = response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        questions = [
            q.strip() for q in text_response.split('\n') 
            if q.strip() and not q.strip().startswith('[') and not q.strip().startswith(']')
        ][:3]
        return questions

    except Exception as e:
        logger.error(f"Error generating related questions: {str(e)}")
        return []