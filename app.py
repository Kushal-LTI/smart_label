# app.py
import json
import os
import uuid
import shutil
import re
import cv2
import numpy as np
import collections
from datetime import datetime
from PIL import Image
from flask import (Flask, render_template, request, url_for, jsonify,
                   redirect, flash, send_from_directory, Response)
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt

from config import config
from models import CurationReport, Patient, db, User
from grad_cam import GradCAM, get_target_layer
from fpdf import FPDF
from io import BytesIO

# Optional genai (Gemini) integration
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# --- GEMINI SUMMARY FUNCTION WITH NEW, MORE CONCISE PROMPT ---
def generate_gemini_summary(final_counts: dict) -> str:
    """
    Return a short clinical summary based on final_counts.
    Uses Google's Gemini API if available and configured, otherwise falls back to a rule-based summary.
    """
    # Simple rule-based fallback function
    def heuristic_summary(fc: dict) -> str:
        lines = ["Automated clinical impression based on cell counts:"]
        total = sum(fc.values())
        if total == 0:
            return "No cells processed."
        most_common = sorted(fc.items(), key=lambda x: x[1], reverse=True)
        top_type, top_count = most_common[0]
        pct = (top_count / total) * 100
        lines.append(f"  - Total cells processed: {total}.")
        lines.append(f"  - Dominant cell type: {top_type} ({top_count} cells, {pct:.1f}% of sample).")
        hints = []
        if top_type.lower() == 'myeloblast' and pct > 20:
            hints.append('Elevated myeloblasts may indicate acute myeloid leukemia (AML) or a myeloproliferative process; correlate clinically.')
        if top_type.lower() == 'erythroblast' and pct > 30:
            hints.append('High erythroblast counts could indicate erythroid hyperplasia or dyserythropoiesis; correlate with CBC and marrow findings.')
        if top_type.lower() == 'neutrophil' and pct > 60:
            hints.append('Neutrophil predominance often reflects reactive/infectious processes.')
        if hints:
            lines.append('  - Possible interpretations:')
            for h in hints:
                lines.append(f"    - {h}")
        else:
            lines.append('  - No specific malignancy-suggesting pattern detected by heuristic rules; consider expert review.')
        return '\n'.join(lines)

    # Check for Gemini availability and configuration
    if GENAI_AVAILABLE and config.GEMINI_API_KEY and config.GEMINI_MODEL:
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            model = genai.GenerativeModel(config.GEMINI_MODEL)
            
            # --- NEW PROMPT FOR CONCISE OUTPUT ---
            prompt_lines = [
                "You are an expert pathology assistant. Based on the following cell counts, provide a highly concise clinical summary.",
                "The summary must be under 75 words and include:",
                "1. A primary differential diagnosis (one sentence).",
                "2. The single most important next step.",
                "Format the output clearly using a heading and bold text.",
                "\nCell Counts:"
            ]
            for cell_type, count in final_counts.items():
                prompt_lines.append(f"- {cell_type}: {count}")
            prompt = '\n'.join(prompt_lines)

            response = model.generate_content(prompt)
            
            if response.text:
                return response.text
            else:
                print("Gemini response was empty, falling back to heuristic summary.")
                return heuristic_summary(final_counts)

        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            print("Falling back to heuristic summary.")
            return heuristic_summary(final_counts)
    else:
        return heuristic_summary(final_counts)

# --- Helper function to render Markdown in PDF ---
def write_markdown_to_pdf(pdf, text):
    """
    Parses simple Markdown (headings, bold, lists) and writes it to the FPDF2 pdf object.
    """
    original_font_family = pdf.font_family
    original_font_style = pdf.font_style
    original_font_size = pdf.font_size_pt

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            pdf.ln(4)
            continue

        if re.match(r'^### (.*)', line):
            heading_text = line.lstrip('### ').strip()
            pdf.set_font(original_font_family, 'B', original_font_size + 1)
            pdf.multi_cell(0, 6, heading_text)
            pdf.set_font(original_font_family, original_font_style, original_font_size)
        elif re.match(r'^## (.*)', line):
            heading_text = line.lstrip('## ').strip()
            pdf.set_font(original_font_family, 'B', original_font_size + 3)
            pdf.multi_cell(0, 7, heading_text)
            pdf.set_font(original_font_family, original_font_style, original_font_size)
        elif re.match(r'^[*-] (.*)', line):
            bullet_text = line[2:].strip()
            pdf.set_x(pdf.l_margin + 5)
            pdf.multi_cell(0, 6, f"-- {bullet_text}")
            pdf.set_x(pdf.l_margin)
        else:
            # Handle inline bold **text** by splitting the line
            parts = re.split(r'(\*\*.*?\*\*)', line)
            for part in parts:
                if not part: continue
                if part.startswith('**') and part.endswith('**'):
                    text_inside = part[2:-2]
                    pdf.set_font(style='B')
                    pdf.write(5, text_inside)
                    pdf.set_font(style='') # Reset to normal
                else:
                    pdf.write(5, part)
            pdf.ln()

    pdf.set_font(original_font_family, original_font_style, original_font_size)


import torch
import torch.nn as nn
from torchvision import models, transforms
from fpdf.enums import XPos, YPos
from grad_cam import GradCAM, get_target_layer

# --- App Initialization & Configuration ---
app = Flask(__name__, instance_relative_config=True, template_folder=os.path.join('app', 'templates'), static_folder=os.path.join('app', 'static'))
app.config.from_object(config)

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def get_model(model_name, num_classes):
    if model_name.lower() == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.lower() == 'mobilenetv2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else: raise ValueError(f"Model {model_name} not supported.")
    return model

def load_all_models():
    loaded_models, cam_visualizers = {}, {}
    for name, path in config.MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"Warning: Model path not found: {path}")
            continue
        model_arch = 'MobileNetV2' if 'mobile' in name.lower() else 'ResNet18'
        model = get_model(model_arch, config.NUM_CLASSES)
        model.load_state_dict(torch.load(path, map_location=torch.device(config.DEVICE)))
        model.eval().to(config.DEVICE)
        loaded_models[name] = model
        target_layer = get_target_layer(model, model_arch)
        if target_layer:
            cam_visualizers[name] = GradCAM(model=model, target_layer=target_layer)
    preprocess = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return loaded_models, cam_visualizers, preprocess

models_dict, cam_visualizers, preprocess_transform = load_all_models()
SESSIONS = {}
TENSOR_CACHE = {}

def predict_with_ensemble(input_tensor):
    individual_predictions = []
    with torch.no_grad():
        for model_name, model in models_dict.items():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            individual_predictions.append({
                'model_name': model_name,
                'predicted_class': config.CLASS_LABELS[predicted_idx.item()],
                'confidence': float(confidence.item() * 100)
            })
    predicted_classes = [p['predicted_class'] for p in individual_predictions]
    avg_confidence = float(np.mean([p['confidence'] for p in individual_predictions]))
    
    disagreement = len(set(predicted_classes)) > 1
    low_confidence_agreement = not disagreement and avg_confidence < config.AGREEMENT_CONFIDENCE_THRESHOLD
    
    hitl_required = bool(disagreement or low_confidence_agreement)
    hitl_reason = "Model Disagreement" if disagreement else "Low Confidence Agreement" if low_confidence_agreement else "High Confidence Agreement"
    
    final_prediction = collections.Counter(predicted_classes).most_common(1)[0][0]
    return final_prediction, avg_confidence, hitl_required, hitl_reason, individual_predictions

print(f"Flask app ready. Loaded models: {list(models_dict.keys())}.")

# --- Authentication Routes ---
@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('curate'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and bcrypt.check_password_hash(user.password_hash, request.form.get('password')):
            login_user(user, remember=True)
            return redirect(url_for('curate'))
        else:
            flash('Login unsuccessful. Please check username and password.', 'danger')
    return render_template('login.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('curate'))
    if request.method == 'POST':
        username = request.form.get('username')
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'warning')
            return redirect(url_for('register'))
        hashed_password = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- Core Application Routes ---
@app.route('/')
@app.route('/curate')
@login_required
def curate():
    return render_template('curate.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    session_uuid = str(uuid.uuid4())
    SESSIONS[session_uuid] = {'patient_name': request.form.get('patient_name', 'Unknown_Patient'), 'images': {}}
    results_list = []
    
    for file in request.files.getlist('files[]'):
        if file and file.filename:
            temp_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
            temp_path = os.path.join(config.TEMP_UPLOAD_DIR, temp_filename)
            file.save(temp_path)
            image = Image.open(temp_path).convert('RGB')
            input_tensor = preprocess_transform(image).unsqueeze(0).to(config.DEVICE)
            TENSOR_CACHE[temp_filename] = input_tensor
            
            final_pred, avg_conf, hitl, reason, preds = predict_with_ensemble(input_tensor)
            
            SESSIONS[session_uuid]['images'][temp_filename] = {
                'original_path': temp_path, 'current_label': final_pred, 'hitl_required': hitl,
            }
            results_list.append({
                'temp_image_path': url_for('static', filename=f'uploads/temp/{temp_filename}'),
                'temp_filename': temp_filename, 'hitl_required': hitl, 'final_prediction': final_pred,
                'avg_confidence': avg_conf, 'reason': reason, 'individual_preds': preds,
                'all_classes': config.CLASS_LABELS
            })
    return jsonify({'results': results_list, 'session_uuid': session_uuid})

@app.route('/get_explanation', methods=['POST'])
@login_required
def get_explanation():
    data = request.get_json()
    temp_filename = data.get('temp_filename')
    input_tensor = TENSOR_CACHE.get(temp_filename)
    if input_tensor is None: return jsonify({'success': False, 'message': 'Cached tensor not found.'}), 404
        
    try:
        image = transforms.ToPILImage()(input_tensor.squeeze(0).cpu())
        original_image_np = np.array(image.resize((224, 224)))
        grad_cam_urls = {}

        for name, visualizer in cam_visualizers.items():
            cam_image = visualizer.get_cam_image(input_tensor, original_image_np)
            cam_filename = f"cam_{name}_{temp_filename}.png"
            cam_path = os.path.join(config.TEMP_UPLOAD_DIR, cam_filename)
            Image.fromarray(cam_image).save(cam_path)
            grad_cam_urls[name] = url_for('static', filename=f'uploads/temp/{cam_filename}')
            
        return jsonify({'success': True, 'grad_cam_urls': grad_cam_urls})
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return jsonify({'success': False, 'message': 'Failed to generate explanation.'}), 500

@app.route('/correct_label', methods=['POST'])
@login_required
def correct_label():
    data = request.get_json()
    session = SESSIONS.get(data.get('session_uuid'))
    if session and data.get('temp_filename') in session['images']:
        session['images'][data.get('temp_filename')]['current_label'] = data.get('new_label')
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Session or image not found.'}), 404

@app.route('/history')
@login_required
def history():
    reports = CurationReport.query.filter_by(user_id=current_user.id).order_by(CurationReport.report_date.desc()).all()
    return render_template('history.html', reports=reports)

@app.route('/finalize_session', methods=['POST'])
@login_required
def finalize_session():
    session_uuid = request.get_json().get('session_uuid')
    session = SESSIONS.get(session_uuid)
    if not session: return jsonify({'success': False, 'message': 'Session not found.'}), 404

    patient_name = session['patient_name']
    patient = Patient.query.filter_by(name=patient_name).first()
    if not patient:
        patient = Patient(name=patient_name)
        db.session.add(patient)
        db.session.commit()

    summary_counts = collections.Counter(img['current_label'] for img in session['images'].values())
    auto_labeled = sum(1 for img in session['images'].values() if not img['hitl_required'])
    manual_correction = len(session['images']) - auto_labeled

    patient_folder = os.path.join(config.PROCESSED_DATA_DIR, f"{patient.id}_{patient.name.replace(' ', '_')}")
    os.makedirs(patient_folder, exist_ok=True)

    for temp_filename, image_data in session['images'].items():
        shutil.copy(image_data['original_path'], os.path.join(patient_folder, f"{image_data['current_label']}_{uuid.uuid4().hex[:6]}.png"))

    new_report = CurationReport(
        total_images=len(session['images']), auto_labeled_count=auto_labeled,
        manual_correction_count=manual_correction, final_counts_json=json.dumps(dict(summary_counts)),
        user_id=current_user.id, patient_id=patient.id
    )
    db.session.add(new_report)
    db.session.commit()
    
    gemini_summary = generate_gemini_summary(dict(summary_counts))
    with open(os.path.join(patient_folder, 'gemini_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(gemini_summary)

    return jsonify({
        'success': True, 'patient_name': patient_name, 'summary': dict(summary_counts),
        'gemini_summary': gemini_summary, 'report_id': new_report.id
    })

@app.route('/session_summary', methods=['POST'])
@login_required
def session_summary():
    session_uuid = request.get_json().get('session_uuid')
    session = SESSIONS.get(session_uuid)
    if not session: return jsonify({'success': False, 'message': 'Session not found.'}), 404
    summary_counts = collections.Counter(img['current_label'] for img in session['images'].values())
    gemini_summary = generate_gemini_summary(dict(summary_counts))
    return jsonify({'success': True, 'summary': dict(summary_counts), 'gemini_summary': gemini_summary})


# --- RESTORED AND CORRECTED: Report Download Route ---
@app.route('/download_report/<int:report_id>')
@login_required
def download_report(report_id):
    report = CurationReport.query.get_or_404(report_id)
    if report.user_id != current_user.id:
        flash("You are not authorized to view this report.", "danger")
        return redirect(url_for('history'))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 20)
    pdf.cell(0, 10, 'Blood Cell Curation Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 8, f"Patient Name: {report.patient.name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"Curated By: Dr. {report.curator.username}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"Analysis Date: {report.report_date.strftime('%Y-%m-%d')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "AI Performance Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 8, f"  - Total Images Processed: {report.total_images}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"  - Auto-labeled by AI: {report.auto_labeled_count}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 8, f"  - Manually Corrected/Reviewed: {report.manual_correction_count}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if report.total_images > 0:
        accuracy = (report.auto_labeled_count / report.total_images) * 100
        pdf.cell(0, 8, f"  - AI First-Pass Agreement Rate: {accuracy:.2f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)

    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Final Cell Counts", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(95, 10, 'Cell Type', 1, 0, 'C', fill=True)
    pdf.cell(95, 10, 'Count', 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C', fill=True)
    
    pdf.set_font("Helvetica", '', 12)
    final_counts = json.loads(report.final_counts_json)
    for cell_type, count in final_counts.items():
        pdf.cell(95, 10, cell_type.capitalize(), 1, 0)
        pdf.cell(95, 10, str(count), 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(6)

    # --- Use the new Markdown renderer for the AI summary ---
    patient_folder = os.path.join(config.PROCESSED_DATA_DIR, f"{report.patient.id}_{report.patient.name.replace(' ', '_')}")
    summary_file = os.path.join(patient_folder, 'gemini_summary.txt')
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as sf:
            summary_text = sf.read()
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 10, "AI-generated Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", '', 11)
        # Call the new function to parse and write the summary
        write_markdown_to_pdf(pdf, summary_text)

    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    
    return Response(
        pdf_buffer.getvalue(),
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment;filename=report_{report.patient.name}.pdf'}
    )


# --- App Startup ---
with app.app_context():
    db.create_all()
    if os.path.exists(app.config['TEMP_UPLOAD_DIR']):
        shutil.rmtree(app.config['TEMP_UPLOAD_DIR'])
    os.makedirs(app.config['TEMP_UPLOAD_DIR'], exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)