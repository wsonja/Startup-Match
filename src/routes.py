import os
from flask import send_from_directory, request, jsonify
from models import Startup

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False

from werkzeug.utils import secure_filename

USE_LLM = False
# USE_LLM = True

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

KNOWN_SKILLS = {
    "python", "java", "c++", "c", "javascript", "typescript", "react", "node",
    "node.js", "flask", "django", "fastapi", "sql", "postgresql", "mysql",
    "mongodb", "aws", "gcp", "docker", "kubernetes", "pytorch", "tensorflow",
    "machine learning", "deep learning", "nlp", "llm", "data analysis",
    "pandas", "numpy", "scikit-learn", "opencv", "html", "css", "git",
    "backend", "frontend", "data structures", "algorithms", "linux", "bash",
    "typescript", "java", "r", "matlab"
}

reader = None


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
                "On macOS run /Applications/Python\ x.x/Install\ Certificates.command or configure CI certificates."
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


def score_startup(startup, query):
    query_terms = set(query.lower().split())

    searchable_text = " ".join([
        startup.name or "",
        startup.stage or "",
        startup.yc_batch or "",
        startup.industry or "",
        startup.location or "",
        startup.description or "",
        startup.tech_stack or "",
        startup.roles or "",
        startup.keywords or ""
    ]).lower()

    score = 0
    matched_terms = []

    for term in query_terms:
        if term in searchable_text:
            score += 1
            matched_terms.append(term)

    return score, matched_terms


def json_search(query):
    if not query or not query.strip():
        return []

    startups = Startup.query.all()
    matches = []

    for startup in startups:
        score, matched_terms = score_startup(startup, query)
        if score > 0:
            matches.append({
                "id": startup.id,
                "name": startup.name,
                "stage": startup.stage,
                "yc_batch": startup.yc_batch,
                "industry": startup.industry,
                "location": startup.location,
                "description": startup.description,
                "tech_stack": startup.tech_stack.split(", "),
                "roles": startup.roles.split(", "),
                "keywords": startup.keywords.split(", "),
                "url": startup.url,
                "match_score": score,
                "matched_terms": matched_terms
            })

    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return matches


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
        return jsonify(json_search(text))

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

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)