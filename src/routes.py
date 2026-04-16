import difflib
import json
import os
import re
from flask import send_from_directory, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from llm_routes import interpret_user_query

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False

from werkzeug.utils import secure_filename

USE_LLM = True
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

SHORT_ACRONYMS = {"ai", "ml", "ui", "ux", "hr", "vc", "db", "os", "cv", "nlp", "llm", "api"}

KNOWN_SKILLS = {
    "python", "java", "c++", "c", "javascript", "typescript", "react", "node",
    "node.js", "flask", "django", "fastapi", "sql", "postgresql", "mysql",
    "mongodb", "aws", "gcp", "docker", "kubernetes", "pytorch", "tensorflow",
    "machine learning", "deep learning", "nlp", "llm", "data analysis",
    "data science", "analytics", "pandas", "numpy", "scikit-learn", "opencv",
    "html", "css", "git", "backend", "frontend", "data structures",
    "algorithms", "linux", "bash", "r", "matlab", "ai", "ml", "ui", "ux",
    "api", "cv", "go", "looker"
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
    "go": "golang",
    "data science": "data science analytics machine learning python sql pandas numpy looker",
    "data scientist": "data science analytics machine learning python sql pandas numpy looker",
    "data analytics": "data analytics analytics sql looker dashboard business intelligence",
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
    "analytics",
    "data analytics",
    "product",
    "marketing",
    "security",
    "engineering",
    "frontend software engineer",
    "backend software engineer",
}

reader = None


def load_companies():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_directory, "enriched_init.json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["companies"]


COMPANIES = load_companies()


def normalize_text_list(items):
    return " ".join(str(x) for x in (items or []) if x)


def get_company_fields(company):
    return {
        "name": company.get("canonical_name", "") or "",
        "description": " ".join([
            company.get("short_description", "") or "",
            company.get("long_description", "") or "",
            company.get("job_posting_text", "") or "",
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
            company.get("job_posting_text", "") or "",
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
        company.get("long_description", "") or "",
        company.get("job_posting_text", "") or "",
    ]).strip().lower()


def build_svd_term_space(companies):
    docs = [get_svd_doc(company) for company in companies]
    docs = [doc for doc in docs if doc.strip()]

    if len(docs) < 3:
        return None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w[\w.+#-]*\b",
    )
    X = vectorizer.fit_transform(docs)

    if X.shape[0] < 3 or X.shape[1] < 3:
        return None

    n_components = min(50, X.shape[0] - 1, X.shape[1] - 1)
    if n_components < 2:
        return None

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    term_vectors = svd.components_.T
    vocab = {term: idx for idx, term in enumerate(feature_names)}

    return {
        "vectorizer": vectorizer,
        "svd": svd,
        "feature_names": feature_names,
        "term_vectors": term_vectors,
        "vocab": vocab,
    }


SVD_SPACE = build_svd_term_space(COMPANIES)


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
    normalized_query = normalize_query_for_svd(corrected_query)
    expansion_terms = get_svd_expansion_terms(normalized_query, svd_space)

    pieces = []
    for part in [corrected_query, normalized_query] + expansion_terms:
        part = (part or "").strip()
        if part and part not in pieces:
            pieces.append(part)

    return " ".join(pieces), expansion_terms


def extract_matched_terms(query, company):
    query_lower = (query or "").lower().strip()
    query_terms = [t.strip(",.;:") for t in query_lower.split() if t.strip(",.;:")]

    searchable_parts = [
        company.get("canonical_name", ""),
        company.get("short_description", ""),
        company.get("long_description", ""),
        company.get("job_posting_text", ""),
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

    matched = []

    if query_lower and query_lower in searchable_text:
        matched.append(query_lower)

    for term in query_terms:
        if term in searchable_text and term not in matched:
            matched.append(term)

    return matched


def sentence_chunks(text, max_sentences=2):
    text = (text or "").strip()
    if not text:
        return []

    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return [text]

    chunks = []
    current = []
    for part in parts:
        current.append(part)
        if len(current) >= max_sentences:
            chunks.append(" ".join(current).strip())
            current = []

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def build_company_chunks(company):
    chunks = []

    def add_chunk(text, source, label=None):
        text = (text or "").strip()
        if text:
            chunks.append({
                "source": source,
                "label": label or source,
                "text": text
            })

    add_chunk(company.get("canonical_name", ""), "name", "Company Name")

    for chunk in sentence_chunks(company.get("short_description", ""), max_sentences=1):
        add_chunk(chunk, "short_description", "Short Description")

    for chunk in sentence_chunks(company.get("long_description", ""), max_sentences=2):
        add_chunk(chunk, "long_description", "Long Description")

    retrieval_document = (company.get("retrieval_document") or "").strip()
    if retrieval_document:
        for chunk in sentence_chunks(retrieval_document, max_sentences=2):
            add_chunk(chunk, "retrieval_document", "Retrieved Company Profile")

    for skill in company.get("aggregated_skills", []) or []:
        add_chunk(skill, "skill", "Skill")

    for role in company.get("inferred_roles", []) or []:
        add_chunk(role, "role", "Role")

    for tag in company.get("tags", []) or []:
        add_chunk(tag, "tag", "Tag")

    for category in company.get("categories", []) or []:
        add_chunk(category, "category", "Category")

    add_chunk(company.get("sector", ""), "sector", "Sector")
    add_chunk(company.get("subsector", ""), "subsector", "Subsector")
    add_chunk(company.get("location", ""), "location", "Location")

    deduped = []
    seen = set()
    for chunk in chunks:
        key = chunk["text"].strip().lower()
        if key and key not in seen:
            deduped.append(chunk)
            seen.add(key)

    return deduped


def retrieve_company_evidence(skills_query, experience_query, interests_query, location_filter, company, top_k=4):
    chunks = build_company_chunks(company)
    if not chunks:
        return []

    query_parts = []

    if skills_query and skills_query.strip():
        query_parts.append(skills_query.strip())

    if experience_query and experience_query.strip():
        query_parts.append(experience_query.strip())

    if interests_query and interests_query.strip():
        query_parts.append(interests_query.strip())

    if location_filter and location_filter.strip():
        query_parts.append(location_filter.strip())

    query = " ".join(query_parts).strip()
    if not query:
        return []

    texts = [chunk["text"] for chunk in chunks]
    corpus = [query] + texts

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w[\w.+#-]*\b",
    )
    matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    ranked = []
    for chunk, score in zip(chunks, sims):
        if score > 0:
            ranked.append({
                "source": chunk["source"],
                "label": chunk["label"],
                "text": chunk["text"],
                "score": round(float(score), 4)
            })

    source_priority = {
        "skill": 4,
        "role": 3,
        "short_description": 3,
        "long_description": 2,
        "retrieval_document": 1,
        "location": 2 if location_filter else 0,
        "tag": 0,
        "category": 0,
        "sector": 0,
        "subsector": 0,
        "name": 0,
    }

    ranked.sort(
        key=lambda x: (x["score"], source_priority.get(x["source"], 0)),
        reverse=True
    )

    return ranked[:top_k]

def compute_fit_strength(skills_query, experience_query, interests_query, location_filter, company, evidence):
    score = 0

    has_skill_reason = bool(skills_query) and any(e["source"] == "skill" for e in evidence)
    has_role_reason = bool(experience_query) and any(e["source"] == "role" for e in evidence)
    has_interest_reason = bool(interests_query) and any(
        e["source"] in {"short_description", "long_description"} for e in evidence
    )
    has_location_reason = bool(location_filter) and bool(company.get("location"))

    if has_skill_reason:
        score += 3
    if has_role_reason:
        score += 2
    if has_interest_reason:
        score += 2
    if has_location_reason:
        score += 1

    if score >= 5:
        return "Very strong fit"
    elif score >= 3:
        return "Strong fit"
    else:
        return "Weak fit"

def build_rag_explanation(skills_query, experience_query, interests_query, location_filter, company, evidence):
    company_name = company.get("canonical_name", "This company")

    skills_query = (skills_query or "").strip()
    experience_query = (experience_query or "").strip()
    interests_query = (interests_query or "").strip()
    location_filter = (location_filter or "").strip()

    reasons = []

    skills = [e["text"] for e in evidence if e["source"] == "skill"]
    roles = [e["text"] for e in evidence if e["source"] == "role"]
    descriptions = [
        e["text"] for e in evidence
        if e["source"] in {"short_description", "long_description"}
    ]

    def clean_sentence(text):
        text = (text or "").strip()
        if not text:
            return None

        text = re.split(r'(?<=[.!?])\s+', text)[0].strip()
        text = re.sub(r'[,.!?;:]+$', '', text).strip()

        text = re.sub(
            rf'^{re.escape(company_name)}\s+(is|serves as)\s+',
            '',
            text,
            flags=re.IGNORECASE
        ).strip()

        return text or None

    descriptions_clean = [clean_sentence(d) for d in descriptions]
    descriptions_clean = [d for d in descriptions_clean if d]

    if skills_query and skills:
        reasons.append(f"it uses {', '.join(skills[:2])}")

    if experience_query and roles:
        reasons.append(f"it aligns with roles like {', '.join(roles[:2])}")

    if interests_query and descriptions_clean:
        desc = descriptions_clean[0]
        desc = desc[0].lower() + desc[1:] if len(desc) > 1 else desc.lower()
        reasons.append(f"it focuses on {desc}")

    if location_filter:
        loc_text = (company.get("location") or "").strip()
        if loc_text:
            reasons.append(f"is based in {loc_text}")

    if not reasons and descriptions_clean:
        desc = descriptions_clean[0]
        desc = desc[0].lower() + desc[1:] if len(desc) > 1 else desc.lower()
        reasons.append(f"it focuses on {desc}")

    if not reasons:
        return f"{company_name} could be a potential fit."

    fit_label = compute_fit_strength(
        skills_query,
        experience_query,
        interests_query,
        location_filter,
        company,
        evidence
    )

    return f"This is a {fit_label.lower()} because " + " and ".join(reasons) + "."


def get_company_stage(company):
    rt = (company.get("funding_summary") or {}).get("latest_round_type")
    if rt:
        s = rt.strip()
        normalized = "Seed" if s.lower().startswith("seed") else s
        return normalized if normalized in SERIES_STAGES else "Other"
    if company.get("yc_batch"):
        return "Seed"
    return None


def expand_with_llm(text, field_type):
    text = (text or "").strip()
    if not USE_LLM or not text:
        return text

    try:
        parsed = interpret_user_query(text, field_type=field_type)
        normalized = str(parsed.get("normalized_text", "")).strip()
        keywords = parsed.get("keywords", [])

        pieces = []
        if normalized:
            pieces.append(normalized)

        for kw in keywords:
            kw = str(kw).strip()
            if kw and kw not in pieces:
                pieces.append(kw)

        return " ".join(pieces) if pieces else text
    except Exception:
        return text


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
                "svd_expansion_terms": [],
                "evidence": [],
                "rag_explanation": "",
            })
        return results

    expanded_skills_query, svd_expansion_terms = expand_skills_query(skills_query, SVD_SPACE)
    llm_experience_query = expand_with_llm(experience_query, "experience")
    llm_interests_query = expand_with_llm(interests_query, "interests")

    print(f"llm_experience:    {llm_experience_query!r}")
    print(f"llm_interests:     {llm_interests_query!r}")

    print("\n" + "=" * 80)
    print("SEARCH DEBUG")
    print(f"skills_query:      {skills_query!r}")
    print(f"experience_query:  {experience_query!r}")
    print(f"interests_query:   {interests_query!r}")
    print(f"expanded_skills:   {expanded_skills_query!r} (fuzzy+SVD)")
    print(f"svd_expansion:     {svd_expansion_terms}")
    print(f"location_filter:   {location_filter!r}")
    print(f"stage_filter:      {stage_filter!r}")
    print(f"companies searched:{len(companies)} / {original_count}")

    company_fields = [get_company_fields(company) for company in companies]

    skills_docs = [cf["tech_stack"] for cf in company_fields]
    roles_docs = [cf["roles"] for cf in company_fields]
    context_docs = [
        " ".join([cf["description"], cf["tags"]]).strip()
        for cf in company_fields
    ]

    def cosine_scores(query, docs):
        if not query or not query.strip():
            return [0.0] * len(docs)

        corpus = [query] + docs
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r"(?u)\b\w[\w.+#-]*\b",
        )
        matrix = vectorizer.fit_transform(corpus)
        return cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    skills_sims = cosine_scores(expanded_skills_query, skills_docs)
    roles_sims = cosine_scores(llm_experience_query, roles_docs)
    context_sims = cosine_scores(llm_interests_query, context_docs)

    ranked = []
    for i, company in enumerate(companies):
        score_sum = 0.0
        weight_sum = 0.0

        if skills_query:
            score_sum += 3.0 * skills_sims[i]
            weight_sum += 3.0

        if experience_query:
            score_sum += 2.0 * roles_sims[i]
            weight_sum += 2.0

        if interests_query:
            score_sum += 1.0 * context_sims[i]
            weight_sum += 1.0

        final_score = (score_sum / weight_sum) if weight_sum > 0 else 0.0

        if final_score > 0:
            combined_query = " ".join(
                part for part in [skills_query, experience_query, interests_query] if (part or "").strip()
            ).strip()

            evidence = retrieve_company_evidence(
                skills_query,
                llm_experience_query,
                llm_interests_query,
                location_filter,
                company,
                top_k=4
            )

            rag_explanation = build_rag_explanation(
                skills_query,
                experience_query,
                interests_query,
                location_filter,
                company,
                evidence
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
                "matched_terms": extract_matched_terms(combined_query, company),
                "svd_expansion_terms": svd_expansion_terms,
                "evidence": evidence,
                "rag_explanation": rag_explanation,
                "_debug": {
                    "skills_sim": round(float(skills_sims[i]), 4),
                    "roles_sim": round(float(roles_sims[i]), 4),
                    "context_sim": round(float(context_sims[i]), 4),
                    "final_score_raw": round(float(final_score), 4),
                    "tech_stack_text": company_fields[i]["tech_stack"],
                    "roles_text": company_fields[i]["roles"][:200],
                    "context_text": context_docs[i][:220],
                    "evidence": evidence,
                }
            })

    ranked.sort(key=lambda x: x["match_score"], reverse=True)

    print("\nTOP RESULTS")
    for idx, item in enumerate(ranked[:5], start=1):
        dbg = item["_debug"]
        print(
            f"{idx}. {item['name']} | match={item['match_score']} "
            f"| skills_sim={dbg['skills_sim']} roles_sim={dbg['roles_sim']} context_sim={dbg['context_sim']}"
        )
        print(f"   matched_terms: {item['matched_terms']}")
        print(f"   tech_stack:    {dbg['tech_stack_text']}")
        print(f"   roles_text:    {dbg['roles_text']}")
        print(f"   context_text:  {dbg['context_text']}")
        print(f"   evidence:      {[e['text'] for e in dbg['evidence']]}")
        print()

    for item in ranked:
        item.pop("_debug", None)

    return ranked[:top_k]


def get_easyocr_reader():
    global reader

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