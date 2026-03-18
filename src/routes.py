import json
import os
from flask import send_from_directory, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False

from werkzeug.utils import secure_filename

USE_LLM = False
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

KNOWN_SKILLS = {
    "python", "java", "c++", "c", "javascript", "typescript", "react", "node",
    "node.js", "flask", "django", "fastapi", "sql", "postgresql", "mysql",
    "mongodb", "aws", "gcp", "docker", "kubernetes", "pytorch", "tensorflow",
    "machine learning", "deep learning", "nlp", "llm", "data analysis",
    "pandas", "numpy", "scikit-learn", "opencv", "html", "css", "git",
    "backend", "frontend", "data structures", "algorithms", "linux", "bash",
    "r", "matlab"
}

reader = None


def load_companies():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_directory, "enriched_init.json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["companies"]


COMPANIES = load_companies()


def build_company_document(company):
    if company.get("retrieval_document"):
        return company["retrieval_document"]

    pieces = [
        company.get("canonical_name", ""),
        company.get("short_description", ""),
        company.get("long_description", ""),
        " ".join(company.get("tags", []) or []),
        " ".join(company.get("categories", []) or []),
        company.get("sector", "") or "",
        company.get("subsector", "") or "",
        company.get("location", "") or "",
        company.get("city", "") or "",
        company.get("state", "") or "",
        company.get("country", "") or "",
        " ".join(company.get("aggregated_skills", []) or []),
    ]
    return " ".join([p for p in pieces if p]).strip()

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
        "roles": normalize_text_list(company.get("inferred_roles", [])),
    }

def extract_matched_terms(query, company):
    query_lower = (query or "").lower().strip()
    query_terms = query_lower.split()

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
    ]

    searchable_text = " ".join(str(part or "") for part in searchable_parts).lower()

    matched = []

    if query_lower and query_lower in searchable_text:
        matched.append(query_lower)

    for term in query_terms:
        if term in searchable_text and term not in matched:
            matched.append(term)

    return matched

def rank_companies(query, companies, top_k=20):
    if not query or not query.strip():
        return []

    company_fields = [get_company_fields(company) for company in companies]

    name_docs = [cf["name"] for cf in company_fields]
    desc_docs = [cf["description"] for cf in company_fields]
    tags_docs = [cf["tags"] for cf in company_fields]
    tech_docs = [cf["tech_stack"] for cf in company_fields]
    roles_docs = [cf["roles"] for cf in company_fields]

    def cosine_scores(docs):
        corpus = [query] + docs
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )
        matrix = vectorizer.fit_transform(corpus)
        return cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    name_sims = cosine_scores(name_docs)
    desc_sims = cosine_scores(desc_docs)
    tags_sims = cosine_scores(tags_docs)
    tech_sims = cosine_scores(tech_docs)
    roles_sims = cosine_scores(roles_docs)

    ranked = []
    for i, company in enumerate(companies):
        final_score = (
            0.1 * name_sims[i] +
            0.4 * desc_sims[i] +
            1.2 * tags_sims[i] +
            3.0 * tech_sims[i] +
            2.0 * roles_sims[i]
        )

        if final_score > 0:
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
                "match_score": round(final_score, 2),
                "matched_terms": extract_matched_terms(query, company)
            })

    ranked.sort(key=lambda x: x["match_score"], reverse=True)
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

    @app.route("/api/startups")
    def startups_search():
        text = request.args.get("query", "")
        return jsonify(rank_companies(text, COMPANIES))

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