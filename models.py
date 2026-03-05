from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Startup(db.Model):
    __tablename__ = 'startups'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    stage = db.Column(db.String(32), nullable=False)          # YC / Series A / Series B
    description = db.Column(db.String(4096), nullable=False)
    tags = db.Column(db.String(512), nullable=True)           # comma-separated for MVP
    url = db.Column(db.String(512), nullable=True)

    def __repr__(self):
        return f'Startup {self.id}: {self.name}'