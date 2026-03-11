import json
import os
import logging
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)

def register_chat_route(app, json_search):
    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "API_KEY not set — add it to your .env file"}), 500

        client = LLMClient(api_key=api_key)

        startups = json_search(user_message)

        context_text = "\n\n---\n\n".join(
            (
                f"Name: {s['name']}\n"
                f"Stage: {s['stage']}\n"
                f"YC Batch: {s['yc_batch']}\n"
                f"Industry: {s['industry']}\n"
                f"Location: {s['location']}\n"
                f"Description: {s['description']}\n"
                f"Tech Stack: {', '.join(s['tech_stack'])}\n"
                f"Roles: {', '.join(s['roles'])}\n"
                f"Match Score: {s['match_score']}\n"
                f"Matched Terms: {', '.join(s['matched_terms'])}"
            )
            for s in startups[:5]
        ) or "No matching startups found."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant for StartupMatch. "
                    "Given a student's skills and interests, recommend the most relevant startups "
                    "using only the provided startup data. Explain why each startup matches."
                ),
            },
            {
                "role": "user",
                "content": f"Startup data:\n\n{context_text}\n\nStudent request: {user_message}",
            },
        ]

        def generate():
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )