import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.append(str(src_path.absolute()))

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from llm import generate_response, generate_related_questions
import json
import logging
import time
from dotenv import load_dotenv
from uuid import uuid4
from starlette.background import BackgroundTask
# Import the React Agent
from react_agent.graph import graph as react_graph
from react_agent.configuration import Configuration
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
# Import the Cognio Agent
from agt.agent import graph as cognio_graph, VaaniState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Verify required API keys are present
required_keys = ['GEMINI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GROQ_API_KEY', 'REPLICATE_API_KEY', 'TAVILY_API_KEY']
missing_keys = [key for key in required_keys if not os.getenv(key)]
if missing_keys:
    logger.warning(f"Missing API keys: {', '.join(missing_keys)}. Some functionality may be limited.")

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session management
sessions = {}

async def get_session_id(request: Request):
    """Get or create a session ID from request headers."""
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = f"session_{uuid4()}"
    
    # Initialize session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = {
            'created_at': time.time(),
            'last_accessed': time.time()
        }
    else:
        sessions[session_id]['last_accessed'] = time.time()
    
    return session_id

@app.post("/chat")
async def chat_endpoint(request: Request, session_id: str = Depends(get_session_id)):
    try:
        # Cancel previous request if header is present
        if request.headers.get('X-Cancel-Previous') == 'true':
            previous_request = sessions[session_id].get('current_request')
            if previous_request:
                sessions[session_id]['cancelled'] = True

        body = await request.json()
        message = body.get('message', '').strip()
        model = body.get('model', 'gemini-1.5-flash').strip()

        # Map model names to their backend versions
        model_mapping = {
            'gpt-4o-mini': 'gpt-4o-mini',
            'gemini-1.5-flash': 'gemini-1.5-flash',
            'claude-3-haiku-20240307': 'claude-3-haiku-20240307',
            'llama-3.3-70b-versatile': 'llama-3.3-70b-versatile',
        }

        model = model_mapping.get(model, model)  # Get the mapped model, or the original if not found
        request_id = request.headers.get('X-Request-ID')

        sessions[session_id]['current_request'] = request_id
        sessions[session_id]['cancelled'] = False

        if not message:
            raise HTTPException(status_code=400, detail="No message provided")

        # Prepare messages for the model
        messages = [
            {"role": "user", "content": message}
        ]

        # Stream the response
        async def generate():
            try:
                async for text in generate_response(messages, model, session_id):
                    if sessions[session_id].get('cancelled', False):
                        break
                    yield f"data: {json.dumps({'response': text})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in generate: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice-chat")
async def voice_chat_endpoint(request: Request, session_id: str = Depends(get_session_id)):
    try:
        logger.info(f"Received voice chat request for session {session_id}")
        
        # Update session last accessed time
        sessions[session_id]['last_accessed'] = time.time()
        
        body = await request.json()
        message = body.get('message', '').strip()
        model = body.get('model', 'gemini-pro').strip()
        language = body.get('language', 'en-US').strip()
        request_id = request.headers.get('X-Request-ID')

        logger.info(f"Processing voice request {request_id} with message: {message[:50]}...")

        if not message:
            logger.warning("Empty message received")
            return JSONResponse({
                "success": False,
                "detail": "No message provided"
            }, status_code=400)

        # Store the current request ID in the session
        previous_request = sessions[session_id].get('current_request')
        sessions[session_id]['current_request'] = request_id
        logger.info(f"Updated session request ID from {previous_request} to {request_id}")

        # Enhanced system prompt for multilingual support
        system_prompt = (
            f"You are a helpful assistant. Respond in {language}. "
            f"If the user speaks in Hindi, respond in Hindi. "
            f"If they speak in English, respond in English. "
            f"Maintain the same language and style as the user's input. "
            f"Keep responses natural and conversational in the detected language."
        )

        # Prepare messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        # Stream the response
        async def generate():
            try:
                async for chunk in generate_response(messages, model, session_id):
                    if sessions[session_id].get('current_request') != request_id:
                        break
                    yield f"data: {json.dumps({'response': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Model error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        logger.error(f"Voice chat error: {str(e)}")
        return JSONResponse({
            "success": False,
            "detail": str(e)
        }, status_code=500)

@app.post("/agent-chat")
async def agent_chat_endpoint(request: Request, session_id: str = Depends(get_session_id)):
    try:
        # Update session last accessed time
        sessions[session_id]['last_accessed'] = time.time()

        # Cancel previous request if header is present
        if request.headers.get('X-Cancel-Previous') == 'true':
            previous_request = sessions[session_id].get('current_request')
            if previous_request:
                sessions[session_id]['cancelled'] = True

        body = await request.json()
        message = body.get('message', '').strip()
        model = body.get('model', 'gemini-1.5-flash').strip()
        request_id = request.headers.get('X-Request-ID')

        if not message:
            raise HTTPException(status_code=400, detail="No message provided")

        sessions[session_id]['current_request'] = request_id
        sessions[session_id]['cancelled'] = False

        async def generate():
            try:
                config = {
                    "configurable": {
                        "model": f"custom/{model}",
                        "max_search_results": 5
                    }
                }
                input_state = {"messages": [HumanMessage(content=message)]}
                async for event in react_graph.astream(input_state, config):
                    if sessions[session_id].get('cancelled', False):
                        break
                    if "messages" in event:
                        for msg in event["messages"]:
                            if isinstance(msg, AIMessage) and not msg.tool_calls:
                                yield f"data: {json.dumps({'response': msg.content})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in agent generate: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        logger.error(f"Agent chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/related-questions")
async def related_questions_endpoint(request: Request):
    try:
        body = await request.json()
        message = body.get('message', '').strip()
        model = body.get('model', 'gemini-pro').strip()

        if not message:
            raise HTTPException(status_code=400, detail="No message provided")

        # Generate related questions
        questions = await generate_related_questions(message, model)

        return JSONResponse({
            "success": True,
            "questions": questions
        })

    except Exception as e:
        logger.error(f"Related questions error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cognio-agent")
async def cognio_agent_endpoint(request: Request, session_id: str = Depends(get_session_id)):
    try:
        # Update session last accessed time
        sessions[session_id]['last_accessed'] = time.time()
        
        body = await request.json()
        message = body.get('message', '').strip()
        model = body.get('model', 'llama-3.3-70b-versatile').strip()
        file_url = body.get('file_url', '')
        web_search_enabled = body.get('web_search_enabled', True)
        deep_research = body.get('deep_research', False)
        request_id = request.headers.get('X-Request-ID')

        if not message:
            raise HTTPException(status_code=400, detail="No message provided")
        
        logger.info(f"Cognio agent request from session {session_id}: {message[:50]}...")
        
        # Store the current request ID in the session
        previous_request = sessions[session_id].get('current_request')
        sessions[session_id]['current_request'] = request_id
        sessions[session_id]['cancelled'] = False
        
        # Stream the response from Cognio Agent
        async def generate():
            try:
                # Configure the agent state based on VaaniState schema
                input_state = VaaniState(
                    messages=[HumanMessage(content=message)],
                    summary="",
                    file_url=file_url,
                    web_search_enabled=web_search_enabled,
                    deep_research=deep_research,
                    agent_name="",
                    extra_question="",
                    user_token="valid_token"  # Using default token from agent.py
                )
                
                # Configuration for agent
                config = {"configurable": {"thread_id": session_id}}
                
                async for event in cognio_graph.astream(input_state, config):
                    if sessions[session_id].get('cancelled', False):
                        break
                    if "messages" in event:
                        for msg in event["messages"]:
                            if isinstance(msg, AIMessage):
                                yield f"data: {json.dumps({'response': msg.content})}\n\n"
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"Error in cognio agent: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                yield f"data: {json.dumps({'error': f'Agent error: {str(e)}'})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        logger.error(f"Cognio agent error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)