import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import os

# Set a dummy key if not present for the demo
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "sk-dummy"

# We try to import LangChain, but wrap in try-except so proper errors show if deps are missing
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
except ImportError:
    print("Please install langchain-openai and langchain")

app = FastAPI(title="Pro-Level AI Streaming API")

class GenRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"

async def mock_generator(prompt: str):
    """
    Simulates a streaming LLM response for demonstration purposes
    without needing a paid API key active constantly.
    """
    response_text = f"Analyzed request: '{prompt}'.\nHere is the streaming response:\n"
    tokens = response_text.split(" ") + ["This ", " is ", " a ", " simulated ", " stream ", " showing ", " FastAPI ", " capabilities."]
    
    for token in tokens:
        yield token + " "
        await asyncio.sleep(0.1) # Simulate network/inference latency

async def real_generator(prompt: str):
    """
    Real LangChain connector.
    """
    llm = ChatOpenAI(streaming=True, temperature=0)
    chain = ChatPromptTemplate.from_template("{msg}") | llm
    async for chunk in chain.astream({"msg": prompt}):
        yield chunk.content

@app.post("/generate")
async def generate_response(req: GenRequest):
    """
    Endpoint that streams tokens in real-time.
    """
    # Switch to real_generator(req.prompt) if you have a valid key
    return StreamingResponse(mock_generator(req.prompt), media_type="text/event-stream")

if __name__ == "__main__":
    print("Starting Server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
