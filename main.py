from fastapi import FastAPI
from pydantic import BaseModel
from openrouter_chat import chat_with_openrouter

app = FastAPI()

class UserMessage(BaseModel):
    message: str

@app.post("/chat")
def chat(user_message: UserMessage):
    reply = chat_with_openrouter(user_message.message)
    return {"reply": reply}
