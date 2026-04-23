import os
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from routes import register_routes, rank_companies, COMPANIES
from llm_routes import register_chat_route, register_llm_test_route

load_dotenv()

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)

app = Flask(
    __name__,
    static_folder=os.path.join(project_root, 'frontend', 'dist'),
    static_url_path=''
)
CORS(app)


def json_search(user_message):
    return rank_companies(
        skills_query="",
        experience_query="",
        interests_query=user_message,
        companies=COMPANIES,
        top_k=10,
        location_filter=None,
        stage_filter=None,
        role_filter=None,
    )


register_routes(app)
# register_chat_route(app, json_search)
register_chat_route(app, lambda user_message: [])
register_llm_test_route(app)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)