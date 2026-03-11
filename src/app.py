import json
import os
from dotenv import load_dotenv
from flask import Flask

load_dotenv()

from flask_cors import CORS
from models import db, Startup
from routes import register_routes

current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)

app = Flask(
    __name__,
    static_folder=os.path.join(project_root, 'frontend', 'dist'),
    static_url_path=''
)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
register_routes(app)

def init_db():
    with app.app_context():
        db.create_all()

        if Startup.query.count() == 0:
            json_file_path = os.path.join(current_directory, 'init.json')
            with open(json_file_path, 'r') as file:
                data = json.load(file)

                for startup_data in data['startups']:
                    startup = Startup(
                        id=startup_data['id'],
                        name=startup_data['name'],
                        stage=startup_data['stage'],
                        yc_batch=startup_data.get('yc_batch'),
                        industry=startup_data['industry'],
                        location=startup_data.get('location'),
                        description=startup_data['description'],
                        tech_stack=', '.join(startup_data['tech_stack']),
                        roles=', '.join(startup_data['roles']),
                        keywords=', '.join(startup_data['keywords']),
                        url=startup_data.get('url')
                    )
                    db.session.add(startup)

                db.session.commit()
                print("Database initialized with startup data")

init_db()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)