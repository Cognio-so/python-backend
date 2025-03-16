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
import asyncio

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

        # Add memory context
        if memory and memory.chat_memory.messages:
            history = [{"role": msg.type, "content": msg.content} for msg in memory.chat_memory.messages]
            messages = history + messages

        buffer = ""
        MIN_CHUNK_SIZE = 50
        COMPLETE_BOUNDARIES = ['\n\n', '. ', '! ', '? ', '\n']  # Include '\n' for table rows

        def enforce_markdown(text):
            """Ensure proper Markdown formatting, replacing malformed separators with proper ones."""
            lines = text.split('\n')
            formatted = []
            in_table = False
            table_headers = None
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('|') and line.count('|') > 2:  # Detect table rows
                    if not in_table and table_headers is None:
                        table_headers = line
                        formatted.append(line)
                        # Skip adding separator here; we'll handle it below
                    else:
                        formatted.append(line)
                    in_table = True
                elif in_table and line.strip() and not line.startswith('|'):
                    # Check if this is a malformed separator (dashes with tabs or spaces)
                    if all(c in '- \t' for c in line.strip()) and i > 0 and lines[i-1].startswith('|'):
                        # Replace with proper separator based on header
                        separator = '|'.join(['-' * (col.strip().find(' ') + 2 if ' ' in col.strip() else len(col.strip()) + 2) 
                                            for col in table_headers.split('|') if col.strip()])
                        formatted.append(f"|{separator}|")
                    else:
                        in_table = False
                        formatted.append('')
                        formatted.append(line)
                elif in_table and not line.strip():
                    formatted.append(line)
                else:
                    if line.startswith(('- ', '* ', '• ')) and not line.startswith(('* ', '- ')):
                        formatted.append(f"* {line[2:].strip()}")
                    else:
                        formatted.append(line)
            # If table ended without a separator, add one
            if in_table and table_headers and not any(l.startswith('|-') or all(c in '- \t' for c in l.strip()) for l in formatted[1:]):
                separator = '|'.join(['-' * (col.strip().find(' ') + 2 if ' ' in col.strip() else len(col.strip()) + 2) 
                                    for col in table_headers.split('|') if col.strip()])
                formatted.insert(1, f"|{separator}|")
            return '\n'.join(formatted).replace('\n\n\n', '\n\n')

        def yield_buffer_if_complete():
            """Yield chunks at logical boundaries, ensuring table rows are intact."""
            nonlocal buffer
            formatted_buffer = enforce_markdown(buffer)
            if len(formatted_buffer) >= MIN_CHUNK_SIZE or '\n' in formatted_buffer:
                lines = formatted_buffer.split('\n')
                accumulated = ''
                for line in lines:
                    if line.strip():
                        accumulated += line + '\n'
                        if (len(accumulated) >= MIN_CHUNK_SIZE or 
                            line.startswith('|') or 
                            any(boundary in accumulated for boundary in COMPLETE_BOUNDARIES[:-1])):
                            yield accumulated
                            logger.debug(f"Yielded chunk: '{accumulated.strip()}'")
                            accumulated = ''
                if accumulated.strip():
                    yield accumulated
                    logger.debug(f"Yielded remaining: '{accumulated.strip()}'")
                buffer = ''
            elif formatted_buffer.strip():
                yield formatted_buffer
                logger.debug(f"Yielded small buffer: '{formatted_buffer.strip()}'")
                buffer = ''

        if model_name == "gemini-1.5-flash":
            prompt_parts = [
                f"You must respond in {language}. Maintain the same language throughout. "
                f"Format tables using Markdown with proper alignment (e.g., | Feature | Apple | Samsung | Notes |)."
            ]
            if memory and memory.chat_memory.messages:
                for msg in memory.chat_memory.messages:
                    prompt_parts.append(msg.content)
            prompt_parts.append(messages[-1]['content'])
            response = model.generate_content(prompt_parts, stream=True)
            for chunk in response:
                if hasattr(chunk, 'text'):
                    logger.debug(f"Raw Gemini chunk: '{chunk.text}'")
                    buffer += chunk.text
                    yield_buffer_if_complete()
                    await asyncio.sleep(0)  # Yield control to event loop
            if buffer.strip():
                yield enforce_markdown(buffer)
                logger.debug(f"Gemini final buffer: '{buffer}'")

        elif model_name == "gpt-4o-mini":
            lang_message = {
                "role": "system",
                "content": f"You must respond in {language}. If the user speaks in Hindi, respond in Hindi using Devanagari script. "
                           f"Format tables using Markdown with proper alignment (e.g., | Feature | Apple | Samsung | Notes |)."
            }
            messages.insert(0, lang_message)
            response = await model.chat.completions.create(model=model_name, messages=messages, stream=True)
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    buffer += chunk.choices[0].delta.content
                    yield_buffer_if_complete()
            if buffer.strip():
                yield enforce_markdown(buffer)
                logger.debug(f"GPT final buffer: '{buffer}'")

        elif model_name == "llama-3.3-70b-versatile":
            response = await model.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "You are a helpful assistant. Format tables using Markdown with proper alignment."}, *messages],
                stream=True
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    buffer += chunk.choices[0].delta.content
                    yield_buffer_if_complete()
            if buffer.strip():
                yield enforce_markdown(buffer)
                logger.debug(f"Llama final buffer: '{buffer}'")

        elif model_name == "claude-3-haiku-20240307":
            system_prompt = next((msg['content'] for msg in messages if msg['role'] == 'system'), 
                                 "You are a helpful assistant. Format tables using Markdown with proper alignment.")
            user_messages = [{"role": m['role'], "content": m['content']} for m in messages if m['role'] != 'system']
            response = await model.messages.create(model=model_name, system=system_prompt, messages=user_messages, max_tokens=4096, stream=True)
            async for chunk in response:
                if chunk.type == "content_block_delta" and hasattr(chunk.delta, 'text'):
                    buffer += chunk.delta.text
                    yield_buffer_if_complete()
            if buffer.strip():
                yield enforce_markdown(buffer)
                logger.debug(f"Claude final buffer: '{buffer}'")

        else:
            raise ValueError(f"Unsupported model: {model_name}")

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        yield f"**Error**: I apologize, but I encountered an error: {str(e)}"

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