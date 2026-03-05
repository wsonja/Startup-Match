from flask import send_from_directory, request, jsonify
from models import db, Startup
import os

def startup_search(query: str):
    q = (query or "").strip()
    if q == "":
        return []

    results = Startup.query.filter(
        db.or_(
            Startup.name.ilike(f"%{q}%"),
            Startup.description.ilike(f"%{q}%"),
            Startup.tags.ilike(f"%{q}%"),
            Startup.stage.ilike(f"%{q}%")
        )
    ).limit(50).all()

    return [
        {
            "id": s.id,
            "name": s.name,
            "stage": s.stage,
            "description": s.description,
            "tags": s.tags,
            "url": s.url
        }
        for s in results
    ]

def register_routes(app):
    # Serve React App (same as before)
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    # API route: startups search
    @app.route("/api/startups/search")
    def startups_search():
        q = request.args.get("q", "")
        return jsonify(startup_search(q))