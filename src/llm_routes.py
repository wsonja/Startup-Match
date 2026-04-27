import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from functools import lru_cache
from flask import request, jsonify
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)

_LLM_TIMEOUT = 8  # seconds before a non-streaming LLM call is abandoned
_executor = ThreadPoolExecutor(max_workers=4)


def _timed_chat(client, messages):
    """Run client.chat in a thread so we can enforce a hard timeout."""
    future = _executor.submit(client.chat, messages, False)
    try:
        return future.result(timeout=_LLM_TIMEOUT)
    except FuturesTimeout:
        logger.warning("LLM call timed out after %ds", _LLM_TIMEOUT)
        return None
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return None


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


@lru_cache(maxsize=512)
def interpret_user_query(text, field_type="interests"):
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

    field_instructions = {
        "skills": (
            "- Expand abbreviations and acronyms to their full technical terms "
            "(e.g. 'ai' → 'artificial intelligence machine learning', 'ml' → 'machine learning deep learning', "
            "'nlp' → 'natural language processing', 'cv' → 'computer vision').\n"
            "- Add 3-5 closely related skills or technologies that employers often list alongside the input.\n"
            "- Keep the original term plus all expanded and related terms.\n"
            "- Focus on concrete technologies, frameworks, and tools.\n"
        ),
        "experience": (
            "- Focus on job titles, responsibilities, industries, and seniority levels.\n"
            "- Convert free-text descriptions into role-like noun phrases "
            "(e.g. 'worked on payments' → 'payments engineer fintech backend').\n"
            "- Include domain synonyms a recruiter would search for.\n"
        ),
        "interests": (
            "- Convert personal phrasing into industry/sector terms "
            "(e.g. 'I love helping people eat healthy' → 'food health wellness consumer').\n"
            "- Include adjacent startup verticals if strongly implied.\n"
            "- Keep 3-6 distinct thematic keywords.\n"
        ),
    }

    specific = field_instructions.get(field_type, field_instructions["interests"])

    messages = [
        {
            "role": "system",
            "content": (
                "You normalize startup-search inputs into concise retrieval queries.\n"
                "Return valid JSON only. No markdown, no explanation, no extra text.\n"
                "Schema: { \"normalized_text\": string, \"keywords\": string[] }\n\n"
                "General rules:\n"
                "- Correct spelling mistakes.\n"
                "- Strip filler words: I, me, my, worked, on, in, with, love, like, interested, using, built, doing.\n"
                "- Output noun phrases and technical terms only — no full sentences.\n"
                "- Do not invent facts about the user.\n"
                "- keywords: 3 to 6 concise, searchable terms.\n\n"
                f"Field-specific rules for '{field_type}':\n{specific}"
            ),
        },
        {
            "role": "user",
            "content": f"User input: {text}",
        },
    ]

    response = _timed_chat(client, messages)
    if response is None:
        return {"normalized_text": text, "keywords": [text]}

    try:
        raw = _extract_text_from_response(response)
        parsed = json.loads(raw)

        normalized_text = str(parsed.get("normalized_text", "")).strip()
        keywords = parsed.get("keywords", [])

        if not isinstance(keywords, list):
            keywords = []

        cleaned_keywords = [str(kw).strip() for kw in keywords if str(kw).strip()]
        return {
            "normalized_text": normalized_text,
            "keywords": cleaned_keywords
        }

    except Exception as e:
        logger.warning(f"interpret_user_query failed for {field_type}: {e}")
        return {"normalized_text": text, "keywords": [text]}


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

        response = _timed_chat(client, messages)
        if response is None:
            return jsonify({"ok": False, "stage": "chat_call", "error": "LLM timeout"}), 504
        text = _extract_text_from_response(response)
        return jsonify({"ok": True, "stage": "chat_complete", "response": text})


def register_llm_test_route(app):
    @app.route("/api/llm-test", methods=["GET"])
    def llm_test():
        try:
            client = _get_client()
        except Exception as e:
            return jsonify({"ok": False, "stage": "client_init", "error": str(e)}), 500

        messages = [
            {"role": "system", "content": "Reply with exactly: OK"},
            {"role": "user", "content": "Test"},
        ]

        response = _timed_chat(client, messages)
        if response is None:
            return jsonify({"ok": False, "stage": "chat_call", "error": "LLM timeout"}), 504
        text = _extract_text_from_response(response)
        return jsonify({"ok": True, "stage": "chat_complete", "response": text})


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
                "You explain why a startup matches a student's query. Use only the provided data.\n"
                "Do not invent facts.\n"
                "Return exactly one sentence (max 40 words).\n"
                "Begin with exactly one of:\n"
                "1. This is an excellent fit because\n"
                "2. This is a good fit because\n"
                "3. This is a weak fit because\n\n"
                "Fit level:\n"
                "- 'excellent' → multiple direct matches: specific skills, matching role titles, and relevant tech stack\n"
                "- 'good' → clear alignment in at least one area (skills OR roles OR sector)\n"
                "- 'weak' → only indirect or broad topic overlap\n\n"
                "Your sentence MUST cite specific evidence: name actual matched skills, role titles, or technologies — "
                "never say only 'aligns with your interests' without specifics.\n"
                "Example good output: 'This is a good fit because they use PyTorch and TensorFlow and hire machine learning engineers.'"
            ),
        },
        {
            "role": "user",
            "content": context,
        },
    ]

    response = _timed_chat(client, messages)
    if response is None:
        return ""

    try:
        text = _extract_text_from_response(response).strip()
        allowed_prefixes = (
            "This is an excellent fit because",
            "This is a good fit because",
            "This is a weak fit because",
        )
        if not any(text.startswith(prefix) for prefix in allowed_prefixes):
            logger.warning(f"generate_rag_explanation unexpected format: {text}")
            return ""
        return text
    except Exception as e:
        logger.warning(f"generate_rag_explanation failed for {startup.get('name')}: {e}")
        return ""
def create_rag_retrieval_query(user_query):
    user_query = (user_query or "").strip()

    if not user_query:
        return {
            "modified_query": "",
            "keywords": []
        }

    try:
        client = _get_client()
    except Exception as e:
        logger.warning(f"create_rag_retrieval_query could not create client: {e}")
        return {
            "modified_query": user_query,
            "keywords": [user_query]
        }

    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite a student's natural language startup matching request "
                "into a concise retrieval query for an information retrieval system.\n"
                "Return valid JSON only. No markdown. No explanation.\n"
                "Schema:\n"
                "{\n"
                '  "modified_query": string,\n'
                '  "keywords": string[]\n'
                "}\n"
                "Rules:\n"
                "- Keep skills, tools, roles, industries, interests, and locations.\n"
                "- Remove filler words and personal phrasing.\n"
                "- Add closely related retrieval terms only when clearly useful.\n"
                "- Do not invent facts about the user.\n"
                "- The modified_query should be one searchable phrase."
            ),
        },
        {
            "role": "user",
            "content": f"Original user query: {user_query}",
        },
    ]

    response = _timed_chat(client, messages)
    if response is None:
        return {"modified_query": user_query, "keywords": [user_query]}

    try:
        raw = _extract_text_from_response(response)
        parsed = json.loads(raw)
        modified_query = str(parsed.get("modified_query", "")).strip()
        keywords = [str(k).strip() for k in parsed.get("keywords", []) if str(k).strip()]
        return {"modified_query": modified_query or user_query, "keywords": keywords}
    except Exception as e:
        logger.warning(f"create_rag_retrieval_query failed: {e}")
        return {"modified_query": user_query, "keywords": [user_query]}


def generate_rag_answer(original_query, modified_query, retrieved_results):
    try:
        client = _get_client()
    except Exception as e:
        logger.warning(f"generate_rag_answer could not create client: {e}")
        return ""

    compact_results = []

    for idx, startup in enumerate((retrieved_results or [])[:8], start=1):
        compact_results.append(
            {
                "rank": idx,
                "name": startup.get("name", ""),
                "match_score": startup.get("match_score", ""),
                "industry": startup.get("industry", ""),
                "location": startup.get("location", ""),
                "stage": startup.get("stage", ""),
                "description": startup.get("description", ""),
                "matched_terms": startup.get("matched_terms", []),
                "related_terms_used": startup.get("related_terms_used", []),
                "roles": startup.get("roles", [])[:8],
                "tech_stack": startup.get("tech_stack", [])[:12],
            }
        )

    context = json.dumps(compact_results, indent=2)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a startup career advisor helping a student find the right startup to join.\n"
                "Answer using ONLY the retrieved results below — do not invent companies or facts.\n"
                "Tone: friendly, direct, actionable.\n"
                "Length: 3 to 5 sentences.\n"
                "Structure your answer as:\n"
                "1. One sentence naming the top 1-2 matches and why they stand out.\n"
                "2. One or two sentences on what makes those companies a good fit "
                "(cite specific skills, roles, or tech stack from the data).\n"
                "3. One sentence with a practical next step "
                "(e.g. 'Check out their careers page' or 'Look at their open roles in X').\n"
                "Do not use bullet points. Do not start with 'Based on the retrieved results'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Student query: {original_query}\n"
                f"Search terms used: {modified_query}\n\n"
                f"Retrieved startups:\n{context}\n\n"
                "Give the student your best recommendation."
            ),
        },
    ]

    response = _timed_chat(client, messages)
    if response is None:
        return ""
    try:
        return _extract_text_from_response(response).strip()
    except Exception as e:
        logger.warning(f"generate_rag_answer failed: {e}")
        return ""