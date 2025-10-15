# Smart Label: Model-Assisted Human-in-the-Loop Labeling

An intelligent web application built with Flask and PyTorch that uses a dual-model ensemble (ResNet-18 & MobileNetV2) to assist hematopathologists in classifying and curating blood cell images. The system enables efficient expert review and streamlines digital curation.

## Features

- **Secure User Authentication:** Multi-user support with registration and login (Flask-Login).
- **Patient-Centric Workflow:** Process entire folders of images for a specific patient.
- **Dual-Model Ensemble:** Uses both ResNet-18 and MobileNetV2 for robust predictions.
- **Intelligent HITL Trigger:** Automatically flags images for expert review based on model disagreement or low-confidence agreement.
- **Batch Curation Tools:** Efficiently apply labels to multiple images at once.
- **PDF Report Generation:** Creates a professional, downloadable PDF summary for each patient session.
- **Persistent History:** Each user can view a history of all patient reports they have curated.

## Tech Stack

- **Backend:** Flask, Flask-SQLAlchemy, Flask-Login
- **Database:** SQLite (can be easily switched to PostgreSQL)
- **Deep Learning:** PyTorch
- **AI Models:** ResNet-18, MobileNetV2
- **Frontend:** JavaScript (AJAX), HTML5, CSS3
- **PDF Generation:** FPDF2
- **Image Processing:** OpenCV, Pillow

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kushal-LTI/smart_label.git
    cd smart_label
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv env
    .\env\Scripts\activate

    # For macOS/Linux
    python3 -m venv env
    # Smart Label: Model-Assisted Human-in-the-Loop Labeling

    A lightweight Flask + PyTorch web app that uses a dual-model ensemble (ResNet-18 & MobileNetV2) and Grad-CAM explanations to help hematopathologists curate blood-cell images.

    This README highlights quick setup, the GenAI (Gemini) summary integration, the key REST endpoints, and developer notes for modifying or extending the system.

    ## Quick features

    - Dual-model ensemble (ResNet-18, MobileNetV2) with rule-based HITL gating
    - On-demand Grad-CAM explanations per model
    - In-memory session workflow with finalize -> copy into `data_processed/`
    - PDF report generation (FPDF2) including AI-generated summary
    - Optional Gemini/GenAI integration for concise clinical summaries

    ## Setup & run (Windows)

    1. Create and activate a virtual environment:

    ```powershell
    python -m venv env
    .\env\Scripts\Activate.ps1
    ```

    2. Install dependencies:

    ```powershell
    pip install -r requirements.txt
    ```

    3. (Optional) Install `genai` and set your Gemini API key. If you set the key the app will call Gemini; otherwise it falls back to a deterministic heuristic.

    ```powershell
    py -m pip install genai
    $env:GEMINI_API_KEY = 'your_gemini_api_key_here'
    # Optional: choose a model
    $env:GEMINI_MODEL = 'gemini-model-name'
    ```

    4. Start the server:

    ```powershell
    python app.py
    ```

    Open: http://127.0.0.1:5000

    ## Key files

    - `app.py` — main Flask app: routes, model loading, Grad-CAM wiring, finalize flow, PDF download
    - `config.py` — constants: `MODEL_PATHS`, `CLASS_LABELS`, `TEMP_UPLOAD_DIR`, `PROCESSED_DATA_DIR`, `GEMINI_API_KEY`, `GEMINI_MODEL`
    - `grad_cam.py` — Grad-CAM helper that returns superimposed images
    - `models/` — model checkpoint files used at runtime
    - `app/static/js/main.js` and `app/templates/curate.html` — front-end interaction and AJAX flows

    ## GenAI (Gemini) integration

    Behavior summary:
    - If `GEMINI_API_KEY` is set and `genai` is installed the server will call Gemini to produce a concise clinical summary for the session.
    - If Gemini is not available, the server uses a short rule-based heuristic fallback.
    - Summaries are saved to `data_processed/<patientid>_<patientname>/gemini_summary.txt` and included in the PDF report.

    Endpoints used for GenAI flows:
    - `POST /session_summary` — Accepts `{ session_uuid }` and returns `{ success, summary, gemini_summary }`. Used by the UI immediately after upload (live summary).
    - `POST /finalize_session` — Finalizes the session, saves images, creates `CurationReport`, and returns `{ success, patient_name, summary, gemini_summary, report_id }`.
    - `GET /download_report/<report_id>` — Returns the PDF with the embedded gemini summary (if present).

    Prompting and token tips (short):
    - The app uses a compact textual prompt and falls back to a short heuristic when needed.
    - To reduce tokens: request JSON-only responses, use short keys (t,c,p,flags,tests,urg,note), limit items (top 1-2) and set temperature to 0.

    Example minimal prompt (the app does this internally):
    ```
    Input counts:
    - myeloblast:24
    - neutrophil:60
    Return only JSON with keys: top, flags, tests, urg, note. Max tokens 120. Temperature 0.
    ```

    ## Quick UI flows

    1. Login / Register
    2. Curate page: enter patient name, upload images or folder
    3. Click "Analyze & Curate" — predictions will appear; the UI will automatically request a live GenAI summary and display it beneath the results.
    4. Make manual corrections as needed.
    5. Click "Finish & Save Report" — finalizes, saves images to `data_processed/`, creates a DB `CurationReport`, and returns a `report_id` you can use to download the PDF.

    ## Developer notes & extension points

    - Adding a new model:
        - Place a checkpoint in `models/` and add/update `config.MODEL_PATHS`.
        - Update `get_model()` in `app.py` to return the correct architecture.
        - Update `get_target_layer()` in `grad_cam.py` to return the right convolutional layer for Grad-CAM.

    - Persisting GenAI summaries in DB (optional):
        - Currently, summaries are saved to a `gemini_summary.txt` file beside the processed images. If you prefer database persistence, add a nullable `gemini_summary` column to `CurationReport` and write the text during `finalize_session`.

    - PDF notes:
        - The PDF generator uses FPDF2 and includes the AI summary when `gemini_summary.txt` is present.
        - The PDF layout uses a fixed-width table; long unbroken tokens may need wrapping. The code ensures left-margin reset before writing multi-line summary text.

    ## Troubleshooting

    - Missing imports in your editor: install dependencies in the project virtualenv and restart the editor's Python interpreter.
    - Models not found: check `config.MODEL_PATHS` to ensure the `.pth` files are in `models/`.
    - GEMINI failures: check that `GEMINI_API_KEY` is set and that `genai` is installed. The app will fall back to heuristics if API calls fail.

    If you'd like, I can:
    - Persist `gemini_summary` to the DB and show it in the `history` view.
    - Change the GenAI prompt to request a structured JSON response (differential, confidence, recommended tests) and parse that into UI fields.

    *** End of README ***
