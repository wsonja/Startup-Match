import json
import os
import logging
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)


def _get_client():
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY not set — add it to your .env file")
    return LLMClient(api_key=api_key)


def _extract_text_from_response(response):
    if response is None:
        return ""

    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        if isinstance(response.get("content"), str):
            return response["content"].strip()

        message = response.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()

        if isinstance(response.get("text"), str):
            return response["text"].strip()

    if isinstance(response, list):
        parts = []
        for item in response:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
        return "".join(parts).strip()

    return str(response).strip()


def interpret_user_query(text, field_type="interests"):
    """
    Use the LLM only to normalize messy free-text experience/interests
    into a cleaner retrieval query.
    """
    text = (text or "").strip()
    if not text:
        return {
            "normalized_text": "",
            "keywords": []
        }

    try:
        client = _get_client()
    except Exception as e:
        logger.warning(f"interpret_user_query could not create client: {e}")
        return {
            "normalized_text": text,
            "keywords": [text]
        }

    messages = [
        {
            "role": "system",
            "content": (
                "You normalize startup-search inputs.\n"
                "Return valid JSON only.\n"
                "No markdown, no explanation, no extra text.\n"
                "The JSON schema is:\n"
                "{\n"
                '  "normalized_text": string,\n'
                '  "keywords": string[]\n'
                "}\n"
                "Rules:\n"
                "- Correct spelling mistakes.\n"
                "- Remove filler words.\n"
                "- Keep only the core user intent.\n"
                "- keywords should be 3 to 6 concise search terms.\n"
                "- For interests, include adjacent concepts if strongly relevant.\n"
                "- For experience, focus on skills, tools, responsibilities, and role-like terms.\n"
                "- Do not invent biography details.\n"
            ),
        },
        {
            "role": "user",
            "content": f"Field type: {field_type}\nUser input: {text}",
        },
    ]

    try:
        response = client.chat(messages, stream=False)
        raw = _extract_text_from_response(response)
        parsed = json.loads(raw)

        normalized_text = str(parsed.get("normalized_text", "")).strip()
        keywords = parsed.get("keywords", [])

        if not isinstance(keywords, list):
            keywords = []

        cleaned_keywords = []
        for kw in keywords:
            kw = str(kw).strip()
            if kw and kw not in cleaned_keywords:
                cleaned_keywords.append(kw)

        return {
            "normalized_text": normalized_text,
            "keywords": cleaned_keywords
        }

    except Exception as e:
        logger.warning(f"interpret_user_query failed for {field_type}: {e}")
        return {
            "normalized_text": text,
            "keywords": [text]
        }


def register_chat_route(app, json_search):
    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        try:
            client = _get_client()
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500

        startups = json_search(user_message)

        context_text = "\n\n---\n\n".join(
            (
                f"Name: {s['name']}\n"
                f"Stage: {s['stage']}\n"
                f"YC Batch: {s.get('yc_batch')}\n"
                f"Industry: {s['industry']}\n"
                f"Location: {s.get('location')}\n"
                f"Description: {s['description']}\n"
                f"Tech Stack: {', '.join(s.get('tech_stack', []))}\n"
                f"Roles: {', '.join(s.get('roles', []))}\n"
                f"Match Score: {s['match_score']}\n"
                f"Matched Terms: {', '.join(s.get('matched_terms', []))}\n"
                f"Explanation: {s.get('rag_explanation', '')}"
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