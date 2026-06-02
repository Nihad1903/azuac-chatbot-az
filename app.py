from fastapi import FastAPI
from pydantic import BaseModel
from rag import get_bot

app = FastAPI()
bot = get_bot()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(data: QuestionRequest):

    answer, chunks = bot.answer(data.question)

    return {
        "question": data.question,
        "answer": answer
    }