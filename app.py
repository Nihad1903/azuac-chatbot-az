from fastapi import FastAPI
from pydantic import BaseModel
from rag import bot

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(data: QuestionRequest):

    answer, chunks = bot.answer(data.question)

    return {
        "question": data.question,
        "answer": answer
    }