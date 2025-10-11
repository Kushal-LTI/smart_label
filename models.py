# models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    # Relationship: A user can have many curation reports
    reports = db.relationship('CurationReport', backref='curator', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

# --- NEW: Patient Model ---
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    # Relationship: A patient can have many reports
    reports = db.relationship('CurationReport', backref='patient', lazy=True)

    def __repr__(self):
        return f'<Patient {self.name}>'

# --- NEW: Curation Report Model ---
class CurationReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # --- Statistics for the enhanced report ---
    total_images = db.Column(db.Integer, nullable=False)
    auto_labeled_count = db.Column(db.Integer, nullable=False)
    manual_correction_count = db.Column(db.Integer, nullable=False)
    
    # Store the final counts as a JSON string
    final_counts_json = db.Column(db.Text, nullable=False)
    
    # --- Foreign Keys to link tables ---
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

    def __repr__(self):
        return f'<CurationReport {self.id} for Patient {self.patient_id}>'