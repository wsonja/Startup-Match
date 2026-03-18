import os
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from routes import register_routes

load_dotenv()

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)

app = Flask(
    __name__,
    static_folder=os.path.join(project_root, 'frontend', 'dist'),
    static_url_path=''
)
CORS(app)

register_routes(app)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)