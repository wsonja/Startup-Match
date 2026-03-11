from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Startup(db.Model):
    __tablename__ = 'startups'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    stage = db.Column(db.String(32), nullable=False)          # YC / Series A / Series B
    yc_batch = db.Column(db.String(32), nullable=True)        # e.g. W24, S23
    industry = db.Column(db.String(128), nullable=False)      # AI, fintech, healthtech, etc.
    location = db.Column(db.String(128), nullable=True)
    description = db.Column(db.Text, nullable=False)
    tech_stack = db.Column(db.Text, nullable=False)           # comma-separated for now
    roles = db.Column(db.Text, nullable=False)                # comma-separated for now
    keywords = db.Column(db.Text, nullable=False)             # comma-separated for matching
    url = db.Column(db.String(256), nullable=True)

    def __repr__(self):
        return f'Startup {self.id}: {self.name}'