from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, Optional
import asyncio
import json
import logging
import os
from uuid import uuid4
import time
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.background import BackgroundTask
from llm import generate_response, generate_related_questions

# Load environment variables
load_dotenv()

# Initialize FastAPI app
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

# Set API keys
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Set environment variables
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY if TAVILY_API_KEY else ''
os.environ['PERPLEXITY_API_KEY'] = PERPLEXITY_API_KEY if PERPLEXITY_API_KEY else ''
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY if OPENAI_API_KEY else ''
os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY if ANTHROPIC_API_KEY else ''

# Import react agent modules
try:
    # Import with explicit names to avoid confusion
    from react_agent.graph import graph as agent_graph
    from react_agent.configuration import Configuration as AgentConfiguration
except ImportError as e:
    logger.error(f"Failed to import react agent modules: {str(e)}")
    # Don't raise here, as we might still want to use other functionality

# Session management
sessions: Dict[str, Any] = {}

# Request models
class ResearchRequest(BaseModel):
    topic: str
    config: Dict[str, Any]

class FeedbackRequest(BaseModel):
    topic: str
    feedback: str

def format_sse_message(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"

async def get_session_id(request: Request) -> str:
    session_id = request.headers.get('X-Session-ID', f"session_{uuid4()}")
    if session_id not in sessions:
        sessions[session_id] = {
            'created_at': time.time(),
            'last_accessed': time.time(),
            'thread': {"configurable": {}}
        }
    else:
        sessions[session_id]['last_accessed'] = time.time()
    return session_id

async def stream_research_events(topic: str, config: Dict[str, Any], thread: Dict[str, Any]):
    try:
        yield format_sse_message({
            'type': 'status',
            'content': 'Starting research process...'
        })

        research_input = {"topic": topic}
        async for event in research_graph.astream(
            research_input,
            thread,
            stream_mode="updates"
        ):
            if '__interrupt__' in event:
                yield format_sse_message({
                    'type': 'interrupt',
                    'value': event['__interrupt__'][0].value
                })
                break
            elif 'status' in event:
                yield format_sse_message({
                    'type': 'status',
                    'content': event['status']
                })
            elif 'error' in event:
                yield format_sse_message({
                    'type': 'error',
                    'content': event['error']
                })
                break

        yield format_sse_message({'type': 'status', 'content': 'Research complete'})
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Research stream error: {str(e)}")
        yield format_sse_message({
            'type': 'error',
            'content': str(e)
        })
        yield "data: [DONE]\n\n"

async def stream_feedback_events(feedback: str, thread: Dict[str, Any]):
    try:
        yield format_sse_message({
            'type': 'status',
            'content': 'Processing input...'
        })

        command = None
        if feedback == "true":
            # For approval, use exact format the graph expects
            command = {
                "resume": "true",  # This matches what the graph checks for
                "topic": thread.get("topic", "")
            }
        else:
            # For feedback, pass it directly
            command = {
                "topic": thread.get("topic", ""),
                "feedback": feedback
            }

        async for event in research_graph.astream(
            command,
            thread,
            stream_mode="updates"
        ):
            if "final_report" in event:
                yield format_sse_message({
                    'type': 'report',
                    'content': event['final_report']
                })
                break
            elif 'status' in event:
                yield format_sse_message({
                    'type': 'status',
                    'content': event['status']
                })
            elif '__interrupt__' in event:
                # Skip interrupt events during approval
                if feedback == "true":
                    continue
                else:
                    yield format_sse_message({
                        'type': 'interrupt',
                        'value': event['__interrupt__'][0].value
                    })
                    break
            elif 'completed_sections' in event:
                yield format_sse_message({
                    'type': 'progress',
                    'completed': len(event['completed_sections'])
                })

        yield format_sse_message({'type': 'status', 'content': 'Processing complete'})
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Feedback stream error: {str(e)}")
        yield format_sse_message({
            'type': 'error',
            'content': str(e)
        })
        yield "data: [DONE]\n\n"

@app.post("/api/research/start")
async def start_research(
    request: ResearchRequest,
    session_id: str = Depends(get_session_id)
):
    try:
        search_api = request.config.get('search_api', 'tavily')
        if search_api == 'tavily' and not TAVILY_API_KEY:
            if PERPLEXITY_API_KEY:
                logger.warning("Switching to Perplexity search as Tavily API key is not configured")
                request.config['search_api'] = 'perplexity'
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No search API keys configured. Please set either TAVILY_API_KEY or PERPLEXITY_API_KEY."
                )

        sessions[session_id]['thread'] = {
            "configurable": request.config,
            "topic": request.topic
        }
        
        return StreamingResponse(
            stream_research_events(
                request.topic,
                request.config,
                sessions[session_id]['thread']
            ),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        logger.error(f"Start research error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/research/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    session_id: str = Depends(get_session_id)
):
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=400, detail="Invalid session")

        if "topic" not in sessions[session_id]['thread']:
            sessions[session_id]['thread']["topic"] = request.topic

        return StreamingResponse(
            stream_feedback_events(
                request.feedback,
                sessions[session_id]['thread']
            ),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        logger.error(f"Submit feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent-chat")
async def agent_chat_endpoint(request: Request, session_id: str = Depends(get_session_id)):
    try:
        # Update session last accessed time
        sessions[session_id]['last_accessed'] = time.time()
        
        body = await request.json()
        message = body.get('message', '').strip()
        
        if not message:
            raise HTTPException(status_code=400, detail="No message provided")
        
        logger.info(f"Agent chat request from session {session_id}: {message[:50]}...")
        
        # Format the message for the agent
        formatted_message = [("user", message)]
        
        # Setup agent response streaming
        async def response_generator():
            try:
                # Call the React agent with the correct input structure
                # React agent expects a messages field but not a topic field
                result = await agent_graph.ainvoke(
                    {"messages": formatted_message},
                    {"configurable": {"system_prompt": "You are a helpful AI assistant."}}
                )
                
                # Get the final AI message
                final_message = result["messages"][-1]
                if hasattr(final_message, "content"):
                    content = final_message.content
                    if isinstance(content, str):
                        yield f"data: {json.dumps({'content': content})}\n\n"
                    else:
                        yield f"data: {json.dumps({'content': str(content)})}\n\n"
                
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Agent error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            response_generator(),
            media_type='text/event-stream',
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

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

# Update model mapping
model_mapping = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gemini-flash-2.0": "gemini-1.5-flash",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "llama-3.3": "accounts/fireworks/models/llama-v2-70b"
}

def create_streaming_response(text: str) -> str:
    """Format text for SSE streaming response"""
    return f"data: {json.dumps({'content': text})}\n\n"

@app.post("/chat")
async def chat_endpoint(request: Request, session_id: str = Depends(get_session_id)):
    try:
        if request.headers.get('X-Cancel-Previous') == 'true':
            if sessions[session_id].get('current_request'):
                sessions[session_id]['cancelled'] = True

        body = await request.json()
        message = body.get('message', '').strip()
        model = body.get('model', 'gemini-pro').strip()
        
        # Validate message content
        if not message:
            logger.warning("Empty message received")
            return JSONResponse({
                "error": "Message cannot be empty"
            }, status_code=400)
            
        # Validate model selection and use mapping
        model = model_mapping.get(model, model)
        if not model:
            logger.warning(f"Invalid model selected: {model}")
            return JSONResponse({
                "error": "Invalid model selection"
            }, status_code=400)

        request_id = request.headers.get('X-Request-ID')

        sessions[session_id]['current_request'] = request_id
        sessions[session_id]['cancelled'] = False

        messages = [{"role": "user", "content": message}]

        async def response_generator():
            try:
                async for chunk in generate_response(messages, model, session_id):
                    if sessions[session_id].get('cancelled', False):
                        logger.info(f"Request {request_id} was cancelled")
                        break
                    
                    if chunk:
                        # Properly format as SSE data
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                
                yield "data: [DONE]\n\n"  # Send DONE marker
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            response_generator(),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice-chat")
async def voice_chat_endpoint(request: Request, session_id: str = Depends(get_session_id)):
    try:
        logger.info(f"Received voice chat request for session {session_id}")

        body = await request.json()
        message = body.get('message', '').strip()
        model = body.get('model')
        language = body.get('language', 'en-US').strip()
        request_id = request.headers.get('X-Request-ID')

        logger.info(f"Processing voice request {request_id} with message: {message[:50]}...")

        if not message:
            logger.warning("Empty message received")
            return JSONResponse({
                "success": False,
                "detail": "No message provided"
            }, status_code=400)

        if not model:
            logger.warning("No model specified")
            return JSONResponse({
                "success": False,
                "detail": "Model must be specified"
            }, status_code=400)

        model = model_mapping.get(model, model)
        sessions[session_id]['current_request'] = request_id
        logger.info(f"Using model: {model}")

        system_prompt = (
            f"You are a helpful assistant. Respond in {language}. "
            f"Maintain the same language and style as the user's input. "
            f"Keep responses natural and conversational."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        response = ""  # Initialize an empty string to accumulate the response
        async for chunk in generate_response(messages, model, session_id):
            if sessions[session_id].get('current_request') != request_id:
                logger.warning(f"Request {request_id} was superseded")
                break
            if chunk:
                response += chunk # Accumulate the chunks

        logger.info(f"Complete voice chat response (first 100 chars): {response[:100]}...")

        return JSONResponse({
            "success": True,
            "response": response,  # Return the complete response
            "language": language,
            "model": model
        })

    except Exception as e:
        logger.exception(f"Voice chat error: {str(e)}")
        return JSONResponse({
            "success": False,
            "detail": str(e)
        }, status_code=500)

@app.post("/related-questions")
async def related_questions_endpoint(request: Request):
    try:
        body = await request.json()
        message = body.get('message', '').strip()
        model = body.get('model', 'gemini-pro').strip()

        if not message:
            raise HTTPException(status_code=400, detail="No message provided")

        questions = await generate_related_questions(message, model)

        return JSONResponse({
            "success": True,
            "questions": questions
        })

    except Exception as e:
        logger.error(f"Related questions error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
