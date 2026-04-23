import difflib
import json
import os
import re

import joblib
from flask import send_from_directory, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from llm_routes import generate_rag_explanation, interpret_user_query

easyocr = None
EASYOCR_AVAILABLE = None


USE_LLM = False
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

SHORT_ACRONYMS = {"ai", "ml", "ui", "ux", "hr", "vc", "db", "os", "cv", "nlp", "llm", "api"}

KNOWN_SKILLS = {
    "python", "java", "c++", "c", "javascript", "typescript", "react", "node",
    "node.js", "flask", "django", "fastapi", "sql", "postgresql", "mysql",
    "mongodb", "aws", "gcp", "docker", "kubernetes", "pytorch", "tensorflow",
    "machine learning", "deep learning", "nlp", "llm", "data analysis",
    "pandas", "numpy", "scikit-learn", "opencv", "html", "css", "git",
    "backend", "frontend", "data structures", "algorithms", "linux", "bash",
    "r", "matlab", "ai", "ml", "ui", "ux", "api", "cv",
}

QUERY_ALIASES = {
    "front end": "frontend ui web react javascript typescript html css",
    "frontend": "frontend ui web react javascript typescript html css",
    "ai": "machine learning ai pytorch tensorflow nlp llm deep learning artificial intelligence",
    "ml": "machine learning ai pytorch tensorflow nlp llm deep learning",
    "backend": "backend api database sql postgresql mongodb flask django fastapi node.js infrastructure",
    "fullstack": "full stack",
    "full-stack": "full stack",
    "ux": "ux ui design product designer user experience",
    "ui": "ui ux design frontend react javascript",
}

ALLOWED_EXPANSION_TERMS = set(KNOWN_SKILLS) | {
    "ui",
    "ux",
    "web",
    "software engineer",
    "full stack",
    "machine learning",
    "ai",
    "data science",
    "product",
    "marketing",
    "security",
    "engineering",
    "frontend software engineer",
    "backend software engineer",
}

DISPLAY_STOPWORDS = {
    "i", "me", "my", "we", "our", "you",
    "a", "an", "the",
    "and", "or", "but",
    "on", "in", "at", "to", "for", "of", "with", "by",
    "is", "are", "was", "were", "be", "been", "being",
    "worked", "working", "work", "love", "like", "liked",
    "using", "used", "built", "doing", "interested",
}

TOKEN_RE = re.compile(r"(?u)\b\w[\w.+#-]*\b")

reader = None

STAGE_ORDER = ["Seed", "Series A", "Series B", "Series C", "Series D", "Series E", "Series F", "Other"]

LOCATION_REGIONS = {
    "Bay Area": ["san francisco", "palo alto", "mountain view", "berkeley", "oakland", "san jose",
                 "san mateo", "redwood city", "sunnyvale", "menlo park", "santa clara",
                 "san carlos", "burlingame", "south san francisco", "foster city",
                 "emeryville", "san leandro", "cupertino", "pleasanton", "walnut creek",
                 "newark", "fremont", "milpitas"],
    "New York": ["new york", "brooklyn", "manhattan", "queens", "bronx", "jersey city", "hoboken"],
    "Los Angeles": ["los angeles", "santa monica", "west hollywood", "culver city",
                    "long beach", "pasadena", "el segundo", "venice"],
    "Seattle": ["seattle", "bellevue", "redmond", "kirkland", "tacoma"],
    "Boston": ["boston", "cambridge, ma", "somerville", "cambridge, united kingdom"],
    "Austin": ["austin"],
    "Chicago": ["chicago"],
    "London": ["london"],
    "India": ["india", "bengaluru", "mumbai", "delhi", "hyderabad", "gurugram", "pune", "chennai"],
    "Canada": ["canada", "toronto", "waterloo", "kitchener", "montreal", "vancouver", "edmonton", "victoria"],
    "Europe": ["london", "paris", "berlin", "amsterdam", "barcelona", "stockholm", "oslo",
               "zürich", "zurich", "copenhagen", "munich", "madrid", "warsaw", "kraków", "krakow",
               "vienna", "brussels", "lisbon", "rome", "milan", "dublin"],
    "Latin America": ["mexico city", "são paulo", "sao paulo", "santiago", "monterrey",
                      "bogotá", "bogota", "guadalajara", "buenos aires", "lima",
                      "medellín", "medellin", "cali", "zapopan", "panama city", "rio de janeiro"],
    "Southeast Asia": ["singapore", "jakarta", "ho chi minh", "kuala lumpur", "manila", "bangkok", "makati"],
}
SERIES_STAGES = {"Seed", "Series A", "Series B", "Series C", "Series D", "Series E", "Series F"}


def load_companies():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_directory, "enriched_init.json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["companies"]


def normalize_text_list(items):
    return " ".join(str(x) for x in (items or []) if x)


def get_company_fields(company):
    return {
        "name": company.get("canonical_name", "") or "",
        "description": " ".join([
            company.get("short_description", "") or "",
            company.get("long_description", "") or "",
        ]).strip(),
        "tags": " ".join([
            normalize_text_list(company.get("tags", [])),
            normalize_text_list(company.get("categories", [])),
            company.get("sector", "") or "",
            company.get("subsector", "") or "",
            company.get("location", "") or "",
            company.get("city", "") or "",
            company.get("state", "") or "",
            company.get("country", "") or "",
        ]).strip(),
        "tech_stack": normalize_text_list(company.get("aggregated_skills", [])),
        "roles": " ".join([
            normalize_text_list(company.get("inferred_roles", [])),
            company.get("short_description", "") or "",
            company.get("long_description", "") or "",
        ]).strip(),
    }


def get_svd_doc(company):
    return " ".join([
        normalize_text_list(company.get("aggregated_skills", [])),
        normalize_text_list(company.get("inferred_roles", [])),
        normalize_text_list(company.get("tags", [])),
        normalize_text_list(company.get("categories", [])),
        company.get("sector", "") or "",
        company.get("subsector", "") or "",
        company.get("short_description", "") or "",
    ]).strip().lower()


def load_svd_space():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_directory, "svd_space.joblib")

    if not os.path.exists(path):
        print("WARNING: svd_space.joblib not found. SVD expansion disabled.")
        return None

    return joblib.load(path)


def build_tfidf_index(docs):
    cleaned_docs = [(doc or "").strip() for doc in docs]
    cleaned_docs = [doc if doc else "__empty__" for doc in cleaned_docs]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w[\w.+#-]*\b",
    )
    matrix = vectorizer.fit_transform(cleaned_docs)

    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def cosine_scores_from_index(query, index):
    if index is None:
        return []

    if not query or not query.strip():
        return [0.0] * index["matrix"].shape[0]

    query_vec = index["vectorizer"].transform([query])
    return cosine_similarity(query_vec, index["matrix"]).flatten()


COMPANIES = load_companies()
SVD_SPACE = load_svd_space()

def build_company_svd_matrix(companies, svd_space):
    if not svd_space:
        return None

    vectorizer = svd_space.get("vectorizer")
    svd = svd_space.get("svd")

    if vectorizer is None or svd is None:
        return None

    docs = [get_svd_doc(company) for company in companies]
    tfidf_matrix = vectorizer.transform(docs)
    return svd.transform(tfidf_matrix)


COMPANY_SVD_MATRIX = build_company_svd_matrix(COMPANIES, SVD_SPACE)

COMPANY_FIELDS = [get_company_fields(company) for company in COMPANIES]
SKILLS_DOCS = [cf["tech_stack"] for cf in COMPANY_FIELDS]
ROLES_DOCS = [cf["roles"] for cf in COMPANY_FIELDS]
CONTEXT_DOCS = [
    " ".join([cf["description"], cf["tags"]]).strip()
    for cf in COMPANY_FIELDS
]

SKILLS_INDEX = build_tfidf_index(SKILLS_DOCS)
ROLES_INDEX = build_tfidf_index(ROLES_DOCS)
CONTEXT_INDEX = build_tfidf_index(CONTEXT_DOCS)

COMPANY_INDEX_BY_NAME = {
    company.get("canonical_name"): i
    for i, company in enumerate(COMPANIES)
}


def get_term_vector(term, svd_space):
    if svd_space is None:
        return None

    vocab = svd_space["vocab"]
    term_vectors = svd_space["term_vectors"]

    if term in vocab:
        return term_vectors[vocab[term]]

    parts = term.split()
    part_vecs = [term_vectors[vocab[p]] for p in parts if p in vocab]
    if part_vecs:
        return sum(part_vecs) / len(part_vecs)

    return None


def normalize_query_for_svd(query):
    q = (query or "").strip().lower()
    return QUERY_ALIASES.get(q, q)


def get_svd_expansion_terms(query, svd_space, top_k=4, min_sim=0.2):
    if not query or not svd_space:
        return []

    normalized_query = normalize_query_for_svd(query)
    query_vec = get_term_vector(normalized_query, svd_space)

    if query_vec is None:
        return []

    sims = cosine_similarity(
        query_vec.reshape(1, -1),
        svd_space["term_vectors"]
    ).flatten()

    feature_names = svd_space["feature_names"]
    original_tokens = set(normalized_query.split())
    excluded_terms = {normalized_query} | original_tokens

    results = []
    ranked_indices = sims.argsort()[::-1]

    for idx in ranked_indices:
        term = feature_names[idx]
        sim = sims[idx]

        if sim < min_sim:
            break

        if term in excluded_terms:
            continue

        if term not in ALLOWED_EXPANSION_TERMS:
            continue

        if len(term) < 3 and term not in {"ui", "ux", "ai", "ml"}:
            continue

        results.append(term)
        if len(results) >= top_k:
            break

    return results


def fuzzy_correct_phrase(phrase, vocab, single_word_candidates, multi_word_candidates):
    phrase = phrase.strip().lower()
    if not phrase:
        return phrase

    if phrase in vocab or phrase in KNOWN_SKILLS or phrase in SHORT_ACRONYMS:
        return phrase

    if " " in phrase:
        close = difflib.get_close_matches(phrase, multi_word_candidates, n=1, cutoff=0.72)
        if close:
            return close[0]

    words = phrase.split()
    corrected = []
    for word in words:
        if not word:
            continue
        if word in vocab or word in KNOWN_SKILLS or word in SHORT_ACRONYMS or len(word) <= 1:
            corrected.append(word)
            continue

        prefix_matches = [c for c in single_word_candidates if c.startswith(word)]
        if prefix_matches:
            corrected.append(min(prefix_matches, key=len))
            continue

        close = difflib.get_close_matches(word, single_word_candidates, n=1, cutoff=0.72)
        corrected.append(close[0] if close else word)

    return " ".join(corrected)


def fuzzy_correct_query(query, svd_space):
    if not query or not svd_space:
        return query

    vocab = svd_space["vocab"]
    all_known = list(KNOWN_SKILLS) + [t for t in vocab if " " not in t and len(t) >= 2]
    single_word_candidates = [t.lower() for t in all_known if " " not in t]
    multi_word_candidates = [t.lower() for t in KNOWN_SKILLS if " " in t] + \
                            [t.lower() for t in vocab if " " in t]

    phrases = [p.strip() for p in query.split(",") if p.strip()]
    corrected = [fuzzy_correct_phrase(p, vocab, single_word_candidates, multi_word_candidates) for p in phrases]
    return " ".join(corrected)


def expand_skills_query(skills_query, svd_space):
    if not skills_query or not skills_query.strip():
        return "", []

    corrected_query = fuzzy_correct_query(skills_query.strip(), svd_space)

    if corrected_query.strip().lower() in SHORT_ACRONYMS:
        return corrected_query, []

    normalized_query = normalize_query_for_svd(corrected_query)
    expansion_terms = get_svd_expansion_terms(normalized_query, svd_space)

    pieces = []
    for part in [corrected_query, normalized_query] + expansion_terms:
        part = (part or "").strip()
        if part and part not in pieces:
            pieces.append(part)

    return " ".join(pieces), expansion_terms


def get_company_stage(company):
    rt = (company.get("funding_summary") or {}).get("latest_round_type")
    if rt:
        s = rt.strip()
        normalized = "Seed" if s.lower().startswith("seed") else s
        return normalized if normalized in SERIES_STAGES else "Other"
    if company.get("yc_batch"):
        return "Seed"
    return None


def exact_skill_match_bonus(raw_skills_query, company):
    raw_terms = [t.strip().lower() for t in raw_skills_query.split() if t.strip()]
    if not raw_terms:
        return 0.0

    searchable_parts = [
        " ".join(company.get("aggregated_skills", []) or []),
        " ".join(company.get("tags", []) or []),
        " ".join(company.get("categories", []) or []),
        company.get("short_description", "") or "",
        company.get("long_description", "") or "",
    ]
    searchable_text = " ".join(searchable_parts).lower()
    searchable_tokens = set(TOKEN_RE.findall(searchable_text))

    bonus = 0.0
    for term in raw_terms:
        if term in searchable_tokens:
            if term in SHORT_ACRONYMS:
                bonus += 0.12
            else:
                bonus += 0.08
        elif term.endswith("s") and term[:-1] in searchable_tokens:
            bonus += 0.10
        elif f"{term}s" in searchable_tokens:
            bonus += 0.10

    return bonus


def get_dimension_top_terms(svd_space, dim_idx, top_n=5):
    if not svd_space:
        return []

    components = svd_space.get("components")
    feature_names = svd_space.get("feature_names")

    if components is None or feature_names is None:
        return []

    if dim_idx < 0 or dim_idx >= components.shape[0]:
        return []

    weights = components[dim_idx]
    top_indices = weights.argsort()[::-1][:top_n]

    results = []
    for idx in top_indices:
        results.append({
            "term": feature_names[idx],
            "weight": round(float(weights[idx]), 4),
        })

    return results


def label_dimension(top_terms):
    terms = [t["term"].lower() for t in top_terms]

    scores = {
        "AI / LLM": 0,
        "Frontend / UI": 0,
        "Backend / Infra": 0,
        "Data / Analytics": 0,
        "Healthcare": 0,
        "Marketing / Growth": 0,
        "Product": 0,
    }

    for term in terms:
        if term in {"llm", "nlp", "rag", "transformers", "chatgpt", "ai"}:
            scores["AI / LLM"] += 2

        if term in {"frontend", "react", "javascript", "typescript", "ui", "css"}:
            scores["Frontend / UI"] += 2

        if term in {"backend", "infrastructure", "kubernetes", "aws", "docker"}:
            scores["Backend / Infra"] += 2

        if term in {"api"}:
            scores["Backend / Infra"] += 1

        if term in {"data", "analytics", "sql", "pandas", "machine learning", "looker"}:
            scores["Data / Analytics"] += 2

        if term in {"healthcare", "insurance", "clinical", "fertility"}:
            scores["Healthcare"] += 3

        if term in {"marketing", "growth", "seo", "brand"}:
            scores["Marketing / Growth"] += 2

        if term in {"product", "manager"}:
            scores["Product"] += 2

    best_label = max(scores, key=scores.get)

    if scores[best_label] == 0:
        return " / ".join(terms[:2])

    return best_label

def get_svd_query_vector(query, svd_space):
    if not query or not query.strip() or not svd_space:
        return None

    vectorizer = svd_space.get("vectorizer")
    svd = svd_space.get("svd")

    if vectorizer is None or svd is None:
        return None

    query_tfidf = vectorizer.transform([query])
    return svd.transform(query_tfidf)[0]


def get_overlap_dimensions(query_vec, company_vec, svd_space, top_k=3, top_terms_per_dim=5):
    if query_vec is None or company_vec is None or not svd_space:
        return []

    overlap_vec = query_vec * company_vec

    ranked_dims = sorted(
        range(len(overlap_vec)),
        key=lambda i: abs(overlap_vec[i]),
        reverse=True
    )[:top_k]

    dimensions = []
    for dim_idx in ranked_dims:
        score = float(overlap_vec[dim_idx])
        if abs(score) <= 0:
            continue

        top_terms = get_dimension_top_terms(svd_space, int(dim_idx), top_n=top_terms_per_dim)

        dimensions.append({
            "dimension": int(dim_idx),
            "label": label_dimension(top_terms),
            "score": round(score, 4),
            "top_terms": top_terms,
        })

    return dimensions


def get_svd_overlap_score(query_vec, company_vec):
    if query_vec is None or company_vec is None:
        return 0.0

    numerator = float((query_vec * company_vec).sum())
    query_norm = float((query_vec ** 2).sum() ** 0.5)
    company_norm = float((company_vec ** 2).sum() ** 0.5)

    if query_norm == 0 or company_norm == 0:
        return 0.0

    return max(0.0, numerator / (query_norm * company_norm))

def get_query_dimensions(query, svd_space, top_k=3, top_terms_per_dim=5):
    if not query or not query.strip() or not svd_space:
        return []

    vectorizer = svd_space.get("vectorizer")
    svd = svd_space.get("svd")
    components = svd_space.get("components")

    if vectorizer is None or svd is None or components is None:
        return []

    query_vec = vectorizer.transform([query])
    query_svd = svd.transform(query_vec)[0]

    ranked_dims = query_svd.argsort()[::-1][:top_k]

    dimensions = []
    for dim_idx in ranked_dims:
        score = float(query_svd[dim_idx])
        if score <= 0:
            continue

        top_terms = get_dimension_top_terms(svd_space, int(dim_idx), top_n=top_terms_per_dim)

        dimensions.append({
            "dimension": int(dim_idx),
            "label": label_dimension(top_terms),
            "score": round(score, 4),
            "top_terms": top_terms,
        })

    return dimensions


def split_query_terms(text):
    if not text:
        return []

    parts = re.split(r"[,\n/]+", text.lower())
    terms = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part not in terms:
            terms.append(part)

        for token in TOKEN_RE.findall(part):
            if token not in terms:
                terms.append(token)

    return terms


def role_skill_overlap_bonus(skills_query, company):
    query_terms = split_query_terms(skills_query)
    if not query_terms:
        return 0.0

    role_text = " ".join(company.get("inferred_roles", []) or []).lower()
    role_tokens = set(TOKEN_RE.findall(role_text))

    bonus = 0.0

    for term in query_terms:
        term = term.strip().lower()
        if not term:
            continue

        phrase_match = term in role_text
        token_match = (
            term in role_tokens
            or f"{term}s" in role_tokens
            or (term.endswith("s") and term[:-1] in role_tokens)
        )

        if phrase_match or token_match:
            if " " in term:
                bonus += 0.35
            elif term in SHORT_ACRONYMS:
                bonus += 0.25
            else:
                bonus += 0.30

    return min(bonus, 0.6)


def is_strict_comma_list(text):
    text = (text or "").strip()
    if not text:
        return False

    parts = [part.strip() for part in text.split(",")]
    if not parts or any(not part for part in parts):
        return False

    for part in parts:
        words = [w for w in part.split() if w.strip()]
        if len(words) == 0 or len(words) > 3:
            return False

    return True


def clean_term_list(terms):
    cleaned = []
    seen = set()

    for term in terms:
        term = str(term or "").strip().lower()
        term = term.strip(",.;:!?()[]{}\"'")
        if not term:
            continue
        if term in DISPLAY_STOPWORDS:
            continue
        if len(term) == 1 and term not in SHORT_ACRONYMS:
            continue
        if term not in seen:
            seen.add(term)
            cleaned.append(term)

    return cleaned


def extract_raw_terms_from_sentence(text):
    tokens = TOKEN_RE.findall((text or "").lower())
    return clean_term_list(tokens)


def build_search_text_from_llm_result(parsed, fallback_text):
    fallback_text = (fallback_text or "").strip()

    if not isinstance(parsed, dict):
        return fallback_text

    normalized_text = str(parsed.get("normalized_text", "")).strip()
    keywords = parsed.get("keywords", [])

    if not isinstance(keywords, list):
        keywords = []

    pieces = []
    for part in [normalized_text] + [str(k).strip() for k in keywords]:
        if part and part not in pieces:
            pieces.append(part)

    return " ".join(pieces).strip() or fallback_text


def build_query_field_payload(text, field_type):
    text = (text or "").strip()
    if not text:
        return {
            "search_text": "",
            "raw_terms": [],
            "llm_terms": [],
        }

    if is_strict_comma_list(text):
        structured_terms = clean_term_list(
            [part.strip().lower() for part in text.split(",") if part.strip()]
        )
        return {
            "search_text": text,
            "raw_terms": structured_terms,
            "llm_terms": [],
        }

    parsed = interpret_user_query(text, field_type=field_type)

    search_text = build_search_text_from_llm_result(parsed, text)
    raw_terms = extract_raw_terms_from_sentence(text)
    llm_terms = clean_term_list(parsed.get("keywords", []))

    return {
        "search_text": search_text,
        "raw_terms": raw_terms,
        "llm_terms": llm_terms,
    }


def build_company_search_blob(company):
    searchable_parts = [
        company.get("canonical_name", ""),
        company.get("short_description", ""),
        company.get("long_description", ""),
        " ".join(company.get("tags", []) or []),
        " ".join(company.get("categories", []) or []),
        company.get("sector", ""),
        company.get("subsector", ""),
        company.get("location", ""),
        company.get("country", ""),
        " ".join(company.get("aggregated_skills", []) or []),
        " ".join(company.get("inferred_roles", []) or []),
    ]

    searchable_text = " ".join(str(part or "") for part in searchable_parts).lower()
    searchable_tokens = set(TOKEN_RE.findall(searchable_text))
    return searchable_text, searchable_tokens


def term_matches_company(term, searchable_text, searchable_tokens):
    term = (term or "").strip().lower()
    if not term:
        return False

    if " " in term:
        return term in searchable_text

    return (
        term in searchable_tokens
        or f"{term}s" in searchable_tokens
        or (term.endswith("s") and term[:-1] in searchable_tokens)
    )


def match_terms_against_company(terms, company):
    searchable_text, searchable_tokens = build_company_search_blob(company)

    matched = []
    for term in clean_term_list(terms):
        if term_matches_company(term, searchable_text, searchable_tokens):
            matched.append(term)

    return matched


def rank_companies(
    skills_query,
    experience_query,
    interests_query,
    companies,
    top_k=20,
    location_filter=None,
    stage_filter=None,
    role_filter=None
):
    no_query = not skills_query and not experience_query and not interests_query

    if no_query and not location_filter and not stage_filter and not role_filter:
        return []

    original_count = len(companies)

    experience_payload = build_query_field_payload(experience_query, "experience")
    interests_payload = build_query_field_payload(interests_query, "interests")

    processed_experience_query = experience_payload["search_text"]
    processed_interests_query = interests_payload["search_text"]

    if location_filter:
        region_terms = LOCATION_REGIONS.get(location_filter)
        if region_terms:
            companies = [
                c for c in companies
                if any(
                    term in (c.get("location") or "").lower()
                    or term in (c.get("city") or "").lower()
                    or term in (c.get("country") or "").lower()
                    for term in region_terms
                )
            ]
        else:
            loc_lower = location_filter.lower()
            companies = [
                c for c in companies
                if loc_lower in (c.get("location") or "").lower()
                or loc_lower in (c.get("city") or "").lower()
                or loc_lower in (c.get("country") or "").lower()
            ]

    if stage_filter:
        companies = [
            c for c in companies
            if get_company_stage(c) == stage_filter
        ]

    if role_filter:
        role_lower = role_filter.lower()
        companies = [
            c for c in companies
            if any(role_lower == r.lower() for r in (c.get("inferred_roles") or []))
        ]

    if no_query:
        results = []
        for company in companies[:top_k]:
            results.append({
                "name": company.get("canonical_name"),
                "stage": company.get("yc_batch") or get_company_stage(company),
                "yc_batch": company.get("yc_batch"),
                "industry": (
                    company.get("sector")
                    or (company.get("tags")[0] if company.get("tags") else None)
                ),
                "location": company.get("location"),
                "description": company.get("short_description") or company.get("long_description"),
                "tech_stack": company.get("aggregated_skills", []),
                "roles": company.get("inferred_roles", []),
                "keywords": company.get("tags", []) + company.get("categories", []),
                "url": company.get("website"),
                "match_score": 0,
                "matched_terms": [],
                "related_terms_used": [],
                "svd_expansion_terms": [],
                "svd_dimensions": [],
                "rag_explanation": "",
            })
        return results

    expanded_skills_query, svd_expansion_terms = expand_skills_query(skills_query, SVD_SPACE)
    query_dimensions = get_query_dimensions(expanded_skills_query or skills_query, SVD_SPACE)

    expanded_skills_query, svd_expansion_terms = expand_skills_query(skills_query, SVD_SPACE)

    svd_query = " ".join(
        part.strip()
        for part in [expanded_skills_query, processed_experience_query, processed_interests_query]
        if part and part.strip()
    )

    query_vec = get_svd_query_vector(svd_query, SVD_SPACE)

    print("\n" + "=" * 80)
    print("SEARCH DEBUG")
    print(f"skills_query:      {skills_query!r}")
    print(f"experience_query:  {experience_query!r}")
    print(f"interests_query:   {interests_query!r}")
    print(f"processed_experience_query: {processed_experience_query!r}")
    print(f"processed_interests_query:  {processed_interests_query!r}")
    print(f"experience_raw_terms: {experience_payload['raw_terms']}")
    print(f"interests_raw_terms:  {interests_payload['raw_terms']}")
    print(f"experience_llm_terms: {experience_payload['llm_terms']}")
    print(f"interests_llm_terms:  {interests_payload['llm_terms']}")
    print(f"expanded_skills:   {expanded_skills_query!r} (fuzzy+SVD)")
    print(f"svd_expansion:     {svd_expansion_terms}")
    print(f"location_filter:   {location_filter!r}")
    print(f"stage_filter:      {stage_filter!r}")
    print(f"companies searched:{len(companies)} / {original_count}")

    skills_sims = cosine_scores_from_index(expanded_skills_query, SKILLS_INDEX)
    roles_sims = cosine_scores_from_index(processed_experience_query, ROLES_INDEX)
    context_sims = cosine_scores_from_index(processed_interests_query, CONTEXT_INDEX)

    ranked = []
    for company in companies:
        global_i = COMPANY_INDEX_BY_NAME.get(company.get("canonical_name"))
        if global_i is None:
            continue

        score_sum = 0.0
        weight_sum = 0.0

        if skills_query:
            score_sum += 3.0 * skills_sims[global_i]
            weight_sum += 3.0

        if experience_query:
            score_sum += 2.0 * roles_sims[global_i]
            weight_sum += 2.0

        if interests_query:
            score_sum += 1.0 * context_sims[global_i]
            weight_sum += 1.0

        final_score = (score_sum / weight_sum) if weight_sum > 0 else 0.0

        company_vec = None
        svd_overlap_score = 0.0

        if COMPANY_SVD_MATRIX is not None and query_vec is not None:
            company_vec = COMPANY_SVD_MATRIX[global_i]
            svd_overlap_score = get_svd_overlap_score(query_vec, company_vec)
            final_score += 0.35 * svd_overlap_score

        if skills_query:
            final_score += exact_skill_match_bonus(skills_query, company)
            final_score += role_skill_overlap_bonus(skills_query, company)

        if final_score > 0:
            raw_skill_terms = clean_term_list(split_query_terms(skills_query))
            raw_user_terms = clean_term_list(
                raw_skill_terms
                + experience_payload["raw_terms"]
                + interests_payload["raw_terms"]
            )

            llm_generated_terms = clean_term_list(
                experience_payload["llm_terms"] + interests_payload["llm_terms"]
            )

            raw_matches = match_terms_against_company(raw_user_terms, company)
            related_terms_used = match_terms_against_company(
                [term for term in llm_generated_terms if term not in raw_user_terms],
                company
            )

            ranked.append({
                "name": company.get("canonical_name"),
                "stage": company.get("yc_batch") or company.get("funding_summary", {}).get("latest_round_type"),
                "yc_batch": company.get("yc_batch"),
                "industry": (
                    company.get("sector")
                    or (company.get("tags")[0] if company.get("tags") else None)
                ),
                "location": company.get("location"),
                "description": company.get("short_description") or company.get("long_description"),
                "tech_stack": company.get("aggregated_skills", []),
                "roles": company.get("inferred_roles", []),
                "keywords": company.get("tags", []) + company.get("categories", []),
                "url": company.get("website"),
                "match_score": round(final_score * 100, 2),
                "matched_terms": raw_matches,
                "related_terms_used": related_terms_used,
                "svd_expansion_terms": svd_expansion_terms,
                "svd_dimensions": [],
                "rag_explanation": "",
                "_debug": {
                    "skills_sim": round(float(skills_sims[global_i]), 4),
                    "roles_sim": round(float(roles_sims[global_i]), 4),
                    "context_sim": round(float(context_sims[global_i]), 4),
                    "final_score_raw": round(float(final_score), 4),
                    "tech_stack_text": COMPANY_FIELDS[global_i]["tech_stack"],
                    "roles_text": COMPANY_FIELDS[global_i]["roles"],
                    "context_text": CONTEXT_DOCS[global_i][:220],
                    "raw_user_terms": raw_user_terms,
                    "llm_generated_terms": llm_generated_terms,
                    "svd_overlap_score": round(float(svd_overlap_score), 4),
                }
            })

    ranked.sort(
        key=lambda x: (len(x["matched_terms"]), x["match_score"]),
        reverse=True
    )

    print("\nTOP RESULTS")
    for idx, item in enumerate(ranked[:5], start=1):
        dbg = item["_debug"]
        print(
            f"{idx}. {item['name']} | match={item['match_score']} "
            f"| skills_sim={dbg['skills_sim']} roles_sim={dbg['roles_sim']} context_sim={dbg['context_sim']} svd_overlap={dbg['svd_overlap_score']}"
        )
        print(f"   matched_terms: {item['matched_terms']}")
        print(f"   related_terms_used: {item['related_terms_used']}")
        print(f"   tech_stack:    {dbg['tech_stack_text']}")
        print(f"   roles_text:    {dbg['roles_text'][:180]}")
        print(f"   context_text:  {dbg['context_text']}")
        print()

    top_results = ranked[:top_k]

    for item in top_results:
        global_i = COMPANY_INDEX_BY_NAME.get(item["name"])

        if COMPANY_SVD_MATRIX is not None and query_vec is not None and global_i is not None:
            company_vec = COMPANY_SVD_MATRIX[global_i]
            item["svd_dimensions"] = get_overlap_dimensions(query_vec, company_vec, SVD_SPACE)
        else:
            item["svd_dimensions"] = []

        item.pop("_debug", None)

    return top_results



def get_easyocr_reader():
    global reader, easyocr, EASYOCR_AVAILABLE

    if EASYOCR_AVAILABLE is None:
        try:
            import easyocr as easyocr_module
            easyocr = easyocr_module
            EASYOCR_AVAILABLE = True
        except ImportError:
            EASYOCR_AVAILABLE = False
            easyocr = None

    if not EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr module not installed. Install with 'pip install easyocr'.")

    if reader is None:
        try:
            reader = easyocr.Reader(["en"])
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize easyocr: {}. "
                "On macOS run /Applications/Python\\ x.x/Install\\ Certificates.command or configure CI certificates."
                .format(e)
            ) from e

    return reader


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_skill(skill):
    s = skill.strip().lower()
    replacements = {
        "nodejs": "node.js",
        "postgres": "postgresql",
        "ml": "machine learning",
        "ai": "machine learning",
        "js": "javascript",
        "ts": "typescript",
    }
    return replacements.get(s, s)


def extract_skills_from_text(text):
    text_lower = text.lower()
    found = []

    for skill in KNOWN_SKILLS:
        if skill in text_lower:
            found.append(normalize_skill(skill))

    return sorted(set(found))


def extract_text_from_image(image_path):
    reader = get_easyocr_reader()
    results = reader.readtext(image_path, detail=0)
    return " ".join(results)


def register_routes(app):
    upload_folder = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = upload_folder

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": USE_LLM})

    @app.route("/api/locations")
    def get_locations():
        counts = {}
        for company in COMPANIES:
            loc = (company.get("location") or "").strip()
            if loc:
                counts[loc] = counts.get(loc, 0) + 1
        sorted_locs = sorted(counts, key=lambda x: -counts[x])
        return jsonify(sorted_locs)

    @app.route("/api/roles")
    def get_roles():
        counts = {}
        for company in COMPANIES:
            for role in (company.get("inferred_roles") or []):
                role = role.strip()
                if role:
                    counts[role] = counts.get(role, 0) + 1
        sorted_roles = sorted(counts, key=lambda x: -counts[x])
        return jsonify(sorted_roles)

    @app.route("/api/regions")
    def get_regions():
        counts = {}
        for region, terms in LOCATION_REGIONS.items():
            for company in COMPANIES:
                loc = (company.get("location") or "").lower()
                city = (company.get("city") or "").lower()
                country = (company.get("country") or "").lower()
                if any(t in loc or t in city or t in country for t in terms):
                    counts[region] = counts.get(region, 0) + 1
        ordered = [r for r in LOCATION_REGIONS if r in counts]
        return jsonify([{"name": r, "count": counts[r]} for r in ordered])

    @app.route("/api/funding-stages")
    def get_funding_stages():
        counts = {}
        for company in COMPANIES:
            stage = get_company_stage(company)
            if stage:
                counts[stage] = counts.get(stage, 0) + 1
        ordered = [s for s in STAGE_ORDER if s in counts]
        remaining = sorted((s for s in counts if s not in STAGE_ORDER), key=lambda x: -counts[x])
        return jsonify(ordered + remaining)

    @app.route("/api/startups")
    def startups_search():
        skills = request.args.get("skills", "").strip()
        experience = request.args.get("experience", "").strip()
        interests = request.args.get("interests", "").strip()
        location = request.args.get("location", "").strip()
        stage = request.args.get("stage", "").strip()
        role = request.args.get("role", "").strip()

        return jsonify(
            rank_companies(
                skills,
                experience,
                interests,
                COMPANIES,
                location_filter=location or None,
                stage_filter=stage or None,
                role_filter=role or None,
            )
        )

    @app.route("/api/rag-explanation", methods=["POST"])
    def rag_explanation():
        data = request.get_json() or {}
        startup = data.get("startup")
        user_query = (data.get("query") or "").strip()

        if not startup:
            return jsonify({"error": "Missing startup"}), 400

        try:
            explanation = generate_rag_explanation(startup, user_query)
            return jsonify({"explanation": explanation})
        except Exception:
            return jsonify({"explanation": ""})

    @app.route("/api/parse-skills-image", methods=["POST"])
    def parse_skills_image():
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported file type"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            extracted_text = extract_text_from_image(filepath)
            skills = extract_skills_from_text(extracted_text)

            return jsonify({
                "skills": skills,
                "raw_text": extracted_text
            })
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            return jsonify({"error": f"Failed to parse image: {str(e)}"}), 500