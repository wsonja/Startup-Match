import json
import os
import logging
from flask import request, jsonify
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)


def _get_client():
    api_key = os.getenv("SPARK_API_KEY")
    if not api_key:
        raise RuntimeError("SPARK_API_KEY not set — add it to your .env file")
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


def register_llm_route(app):
    @app.route("/api/llm", methods=["POST"])
    def llm():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        try:
            client = _get_client()
        except Exception as e:
            return jsonify({
                "ok": False,
                "stage": "client_init",
                "error": str(e),
            }), 500

        messages = [
            {"role": "system", "content": "Reply with exactly one short sentence."},
            {"role": "user", "content": user_message},
        ]

        try:
            response = client.chat(messages, stream=False)
            text = _extract_text_from_response(response)
            return jsonify({
                "ok": True,
                "stage": "chat_complete",
                "response": text,
            })
        except Exception as e:
            return jsonify({
                "ok": False,
                "stage": "chat_call",
                "error": str(e),
            }), 500


def register_llm_test_route(app):
    @app.route("/api/llm-test", methods=["GET"])
    def llm_test():
        try:
            client = _get_client()
        except Exception as e:
            return jsonify({
                "ok": False,
                "stage": "client_init",
                "error": str(e),
            }), 500

        messages = [
            {"role": "system", "content": "Reply with exactly: OK"},
            {"role": "user", "content": "Test"},
        ]


        try:
            response = client.chat(messages, stream=False)
            text = _extract_text_from_response(response)
            return jsonify({
                "ok": True,
                "stage": "chat_complete",
                "response": text,
            })
        except Exception as e:
            return jsonify({
                "ok": False,
                "stage": "chat_call",
                "error": str(e),
            }), 500


def generate_rag_explanation(startup, user_query):
    try:
        client = _get_client()
    except Exception as e:
        logger.warning(f"generate_rag_explanation could not create client: {e}")
        return ""
    
    lines = []
    for d in startup.get("svd_dimensions", [])[:3]:
        label = d.get("label") or f"Dimension {d.get('dimension')}"
        terms = ", ".join(t.get("term", "") for t in d.get("top_terms", []))
        lines.append(f"- {label}: {terms}")
    
    dimensions_text = "\n".join(lines)

    context = (
        f"Student query: {user_query}\n\n"
        f"Startup name: {startup.get('name', '')}\n"
        f"Industry: {startup.get('industry', '')}\n"
        f"Location: {startup.get('location', '')}\n"
        f"Stage: {startup.get('stage', '')}\n"
        f"Description: {startup.get('description', '')}\n"
        f"Tech stack: {', '.join(startup.get('tech_stack', []))}\n"
        f"Roles: {', '.join(startup.get('roles', []))}\n"
        f"Keywords: {', '.join(startup.get('keywords', []))}\n"
        f"Matched terms: {', '.join(startup.get('matched_terms', []))}\n"
        f"Related terms: {', '.join(startup.get('svd_expansion_terms', []))}\n"
        f"Match score: {startup.get('match_score', '')}\n"
        f"Top dimensions:\n{dimensions_text}\n"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You explain why a startup matches a student's query.\n"
                "Use only the provided startup data.\n"
                "Do not invent facts.\n"
                "Return exactly one sentence.\n"
                "The sentence must begin with exactly one of these three phrases:\n"
                "1. This is an excellent fit because\n"
                "2. This is a good fit because\n"
                "3. This is a weak fit because\n\n"
                "Choose the label using the evidence:\n"
                "- 'excellent fit' if there are multiple strong direct matches across matched terms, roles, tech stack, or dimensions\n"
                "- 'good fit' if there is some clear alignment but it is not especially strong or comprehensive\n"
                "- 'weak fit' if the overlap is limited, indirect, or mostly based on broader related terms\n\n"
                "Mention the most important specific reasons, such as role-title overlap, matched skills, tech stack, or company focus."
            ),
        },
        {
            "role": "user",
            "content": context,
        },
    ]

    try:
        response = client.chat(messages, stream=False)
        text = _extract_text_from_response(response).strip()

        allowed_prefixes = (
            "This is an excellent fit because",
            "This is a good fit because",
            "This is a weak fit because",
        )

        if not any(text.startswith(prefix) for prefix in allowed_prefixes):
            logger.warning(f"generate_rag_explanation returned unexpected format: {text}")
            return ""

        return text
    except Exception as e:
        logger.warning(f"generate_rag_explanation failed for {startup.get('name')}: {e}")
        return ""
