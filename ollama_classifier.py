from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import httpx
import asyncio
import json
from typing import Optional
import re
import aiofiles

# Initialize FastAPI app
app = FastAPI(title="Ollama Mistral Intent Classifier", version="1.0.0")

# Ollama configuration
OLLAMA_URL = "http://localhost:11434"  # Use the quantized version for better performance

# Request model
class ClassificationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.1
    system_prompt: Optional[str] = None

# Response model
class ClassificationResponse(BaseModel):
    intent: str
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    raw_response: Optional[str] = None




async def check_ollama_health():
    """Check if Ollama is running and Mistral model is available"""
    try:
        async with httpx.AsyncClient() as client:
            # Check if Ollama is running
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                models = response.json()
                model_names = [model["name"] for model in models.get("models", [])]
                
                # Check if Mistral model is available
                mistral_available = any("mistral" in name for name in model_names)
                if not mistral_available:
                    return False, f"Mistral model not found. Available models: {model_names}"
                
                return True, "Ollama and Mistral are ready"
            else:
                return False, f"Ollama not responding: {response.status_code}"
    except Exception as e:
        return False, f"Cannot connect to Ollama: {str(e)}"

async def classify_with_ollama(prompt: str, system_prompt: str, temperature: float = 0.1) -> tuple[str, str]:
    """
    Send prompt to Ollama Mistral model for intent classification
    Returns: (classified_intent, raw_response)
    """
    
    # Create a focused system prompt for intent classification
    classification_prompt = f"""
You are an expert in understanding user behavior. Given a user message, extract the core intent in a short natural language phrase (not more than 10 words). Be precise and avoid repeating the full message.

Respond only with the intent.

Message: {prompt}
Intent:
"""

    payload = {
        "model": "mistral",
        "prompt": classification_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_k": 10,
            "top_p": 0.9,
            "num_predict": 10  # Limit response to just the intent
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "").strip().lower()
                
                # Extract intent from response
                intent = extract_intent_from_response(raw_response)
                return intent, raw_response
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Ollama API error: {response.status_code} - {response.text}"
                )
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {str(e)}")


def extract_intent_from_response(raw_response: str) -> str:
    return raw_response.strip().capitalize()

    

@app.on_event("startup")
async def startup_event():
    """Check Ollama health on startup"""
    health, message = await check_ollama_health()
    if health:
        print(f"{message}")
        print(f"Using Mistral model")
    else:
        print(f"Warning: {message}")
        print("The API will start but may fail requests until Ollama/Mistral is available")

@app.get("/")
async def root():
    """Health check endpoint"""
    health, message = await check_ollama_health()
    return {
        "message": "Ollama Mistral Intent Classifier API",
        "status": "healthy" if health else "warning",
        "ollama_status": message,
        "model": "mistral"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    health, message = await check_ollama_health()
    return {
        "status": "healthy" if health else "degraded",
        "service": "Ollama Mistral Intent Classifier",
        "ollama_status": message,
        "valid_intents": VALID_INTENTS,
        "model": "mistral",
        "ollama_url": OLLAMA_URL
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_intent(request: ClassificationRequest):
    """
    Classify user intent using Ollama Mistral model
    
    Args:
        request: Classification request with prompt and parameters
        
    Returns:
        ClassificationResponse with predicted intent
    """
    try:
        import time
        start_time = time.time()
        
        # Validate input
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Set defaults
        temperature = min(max(request.temperature or 0.1, 0.0), 1.0)  # Clamp between 0-1
        system_prompt = request.system_prompt or "Classify the user intent."
        
        # Classify using Ollama Mistral
        predicted_intent, raw_response = await classify_with_ollama(
            prompt=request.prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        processing_time = time.time() - start_time

        async def save_to_jsonl(prompt: str, intent: str, path="pseudo_labels.jsonl"):
            async with aiofiles.open(path, mode="a") as f:
                line = json.dumps({"input": prompt, "output": intent})
                await f.write(line + "\n")

        await save_to_jsonl(request.prompt, predicted_intent)

        return ClassificationResponse(
            intent=predicted_intent,
            processing_time=processing_time,
            raw_response=raw_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/intents")
async def get_valid_intents():
    """Get list of valid intent categories"""
    return {"valid_intents": VALID_INTENTS}

@app.get("/ollama/models")
async def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=502, detail="Cannot fetch Ollama models")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama connection failed: {str(e)}")


@app.post("/batch-label")
async def batch_label(data: list[str]):
    results = []
    for prompt in data:
        try:
            intent, raw = await classify_with_ollama(prompt, system_prompt="")
            await save_to_jsonl(prompt, intent)
            results.append({"input": prompt, "intent": intent})
        except Exception as e:
            results.append({"input": prompt, "error": str(e)})
    return results


if __name__ == "__main__":
    print("Starting Ollama Mistral Intent Classifier API...")
    print(f"Connecting to Ollama at: {OLLAMA_URL}")
    print("\nAvailable endpoints:")
    print("  GET  /              - Health check")
    print("  GET  /health        - Detailed health check")
    print("  POST /classify      - Classify intent with Mistral")
    print("  GET  /intents       - Get valid intent categories")
    print("  GET  /ollama/models - List available Ollama models")
    print(f"\n API docs: http://localhost:8000/docs")
    print("\n Make sure Ollama is running and Mistral model is available:")
    print("   ollama serve")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    ) 