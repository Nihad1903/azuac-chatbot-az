import logging
from typing import Any, Dict, Optional

import runpod

from rag import get_bot


def _extract_question(event: Dict[str, Any]) -> Optional[str]:
    if not isinstance(event, dict):
        return None

    payload = event.get("input", event)
    if isinstance(payload, dict):
        question = payload.get("question")
        if isinstance(question, str):
            return question.strip()
    elif isinstance(payload, str):
        return payload.strip()

    return None


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        question = _extract_question(event)
        if not question:
            return {"error": "Missing 'question' in input."}

        bot = get_bot()
        answer, chunks = bot.answer(question)
        return {
            "question": question,
            "answer": answer,
            "chunks": chunks,
        }
    except Exception as exc:
        logging.exception("Handler error")
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
