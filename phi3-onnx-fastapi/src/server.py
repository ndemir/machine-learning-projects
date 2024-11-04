import onnxruntime_genai as og
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uuid
import os
from typing import Optional, Dict
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize model
    init_model("Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/")
    yield
    # Shutdown: cleanup (if needed)

app = FastAPI(lifespan=lifespan)

# Pydantic model for request body
class ChatRequest(BaseModel):
    message: str = Field(
        description="The input text prompt for the model to generate from"
    )
    do_sample: bool = Field(
        default=False,
        description="Whether to use random sampling. If False, uses greedy decoding"
    )
    max_length: int = Field(
        default=2048,
        description="Maximum number of tokens to generate including the prompt",
        ge=1  # greater than or equal to 1
    )
    min_length: Optional[int] = Field(
        default=None,
        description="Minimum number of tokens to generate including the prompt",
        ge=1
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p sampling probability (nucleus sampling)",
        ge=0.0,
        le=1.0
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Top-k tokens to sample from",
        ge=1
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature. Higher values make output more random, lower values more deterministic",
        ge=0.0
    )
    repetition_penalty: Optional[float] = Field(
        default=None,
        description="Penalty for repetition in generated text. Higher values discourage repetition",
        ge=0.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello!",
                "do_sample": True,
                "temperature": 0.7,
                "max_length": 2048
            }
        }

# Global variables for model and tokenizer
model = None
tokenizer = None
tokenizer_stream = None

# Initialize model and tokenizer
def init_model(model_path: str):
    global model, tokenizer, tokenizer_stream
    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

# Add new constants
CHAT_HISTORY_DIR = "chat_histories"

# Create chat history directory if it doesn't exist
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


@app.get("/chat")
async def create_chat() -> Dict[str, str]:
    # Generate a new UUID for the conversation
    chat_id = str(uuid.uuid4())
    
    # Create an empty file for this conversation
    chat_file = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.txt")
    with open(chat_file, "w") as f:
        pass  # Create empty file
    
    return {"chat_id": chat_id}

@app.post("/chat/{chat_id}")
async def chat(chat_id: str, request: ChatRequest) -> Dict[str, str]:
    # Validate chat_id format
    try:
        uuid.UUID(chat_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat ID format")
    
    # Check if chat history exists
    chat_file = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.txt")
    if not os.path.exists(chat_file):
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Read previous conversation
    with open(chat_file, "r") as f:
        conversation_history = f.read()
    
    # Prepare the prompt with conversation history
    if conversation_history:
        chat_template = f"{conversation_history}\n<|user|>\n{request.message} <|end|>\n<|assistant|>"
    else:
        chat_template = f"<|user|>\n{request.message} <|end|>\n<|assistant|>"
    
    # Generate response using existing logic
    input_tokens = tokenizer.encode(chat_template)
    params = og.GeneratorParams(model)
    params.set_search_options()  # You can adjust parameters as needed
    params.input_ids = input_tokens
    generator = og.Generator(model, params)

    generated_text = []
    try:
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            generated_text.append(tokenizer_stream.decode(new_token))
    finally:
        del generator
    
    response = "".join(generated_text)
    
    # Update conversation history with the new message and response
    with open(chat_file, "a") as f:
        message_format = "\n" if conversation_history else ""
        message_format += f"<|user|>\n{request.message} <|end|>\n<|assistant|>{response}"
        f.write(message_format)
    
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)