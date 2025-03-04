from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from llm import generate_response, generate_related_questions
import json
import logging
import os
from dotenv import load_dotenv
from uuid import uuid4
import time
from starlette.background import BackgroundTask

# Load environment variables
load_dotenv()

# Verify required API keys are present
if not os.getenv('GOOGLE_API_KEY') or not os.getenv('OPENAI_API_KEY') or not os.getenv('ANTHROPIC_API_KEY'):
    raise ValueError("One or more API keys are missing in environment variables!")

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
            'llama-v3p1-8b-instruct': 'llama-v3p1-8b-instruct',
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
                    # Format the response as a proper SSE data chunk
                    yield f"data: {text}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in generate: {str(e)}")
                yield f"data: Error: {str(e)}\n\n"

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

        try:
            response = ""
            async for chunk in generate_response(messages, model, session_id):
                # Check if the request is still valid
                if sessions[session_id].get('current_request') != request_id:
                    logger.warning(f"Request {request_id} was superseded")
                    raise HTTPException(status_code=409, detail="Request superseded")
                response += chunk

            if not response:
                raise ValueError("No response generated")

            logger.info(f"Successfully generated response for voice request {request_id}")
            return JSONResponse({
                "success": True,
                "response": response,
                "language": language
            })

        except Exception as e:
            logger.error(f"Model error: {str(e)}")
            return JSONResponse({
                "success": False,
                "detail": f"Model error: {str(e)}"
            }, status_code=500)

    except Exception as e:
        logger.error(f"Voice chat error: {str(e)}")
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

        # Generate related questions
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