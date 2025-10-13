## Purpose
Help AI coding agents be immediately productive in the `smart_label` codebase (Flask + PyTorch H.I.T.L. web app).

Keep edits minimal and localized unless asked otherwise. Prefer changes to `app/` templates, `app/static/js/main.js`, or small fixes in `app.py` and `grad_cam.py` rather than large refactors.

## Big picture (what this repo does)
- Single-process Flask webapp that provides a human-in-the-loop labeling UI for blood-cell images.
- Core ML components: two PyTorch models (ResNet-18 and MobileNetV2) in `models/` wired via `config.py` `MODEL_PATHS` and loaded in `app.py`.
- Grad-CAM explanation utility in `grad_cam.py` used on-demand by `/get_explanation`.
- Sessions are temporary, in-memory (see `SESSIONS` in `app.py`) and images are saved under `app/static/uploads/temp` until finalized into `data_processed/`.

## Key files and roles (quick reference)
- `app.py` — main Flask app, routing, auth, model loading, prediction/ensemble logic, Grad-CAM endpoints, and PDF report generation.
- `config.py` — environment paths, model names, class labels, thresholds, and DEVICE selection. Use this file for global constants.
- `models/` — pre-trained model checkpoints used by `app.py` (e.g., `resnet18_blood_cell_best.pth`).
- `grad_cam.py` — lightweight Grad-CAM helper (feature hooks + heatmap compositing).
- `models.py` — SQLAlchemy models: `User`, `Patient`, `CurationReport` and DB init.
- `app/templates/` and `app/static/` — frontend pages and JS that call the REST endpoints (AJAX flows rely on specific JSON shapes returned by `app.py`).
- `requirements.txt` — packages to install. The project expects a Python virtualenv and may run on CPU if CUDA isn't available.

## Developer workflows & commands
- Run locally (Windows PowerShell):
  - Activate venv: `env\Scripts\Activate.ps1` (or `env\Scripts\activate` for cmd)
  - Install deps: `pip install -r requirements.txt`
  - Start server: `python app.py` (dev server, debug=True in `app.py`)
- DB: the SQLite DB file is created automatically; migrations are not present. Use `app.app_context()` blocks in scripts or the running server to access `db`.

## Patterns & conventions the agent should follow
- Small surface fixes only: keep authentication, routing, and DB shapes unchanged unless a user requests schema migrations.
- Session/state: `SESSIONS` is ephemeral — don't persist changes to it unless implementing the `finalize_session` flow (see `app.py`).
- Model handling: use `config.MODEL_PATHS` and `get_model()` in `app.py` when adding support for additional architectures. Register correct target layers via `get_target_layer()` in `grad_cam.py`.
- Grad-CAM: `GradCAM` returns a superimposed BGR image (OpenCV) — endpoints convert to PNG files saved into `app/static/uploads/temp` and return `url_for('static', ...)` paths.

## Important implementation details & examples
- Ensemble prediction: `predict_with_ensemble(input_tensor)` in `app.py` returns (final_prediction, avg_confidence, hitl_required, hitl_reason, individual_predictions). Frontend expects `hitl_required` to decide whether to prompt manual review.
- HITL criteria used:
  - Disagreement across models -> require review.
  - Agreement but average confidence < `config.AGREEMENT_CONFIDENCE_THRESHOLD` -> require review.
- When saving finalized images, `finalize_session` copies files into `data_processed/<patientid>_<patientname>/` and names them like `<label>_N.png`.
- PDF reports use `fpdf2` and expect `CurationReport.final_counts_json` to be a JSON-encoded dict.

## Cross-cutting integration points to watch
- Static upload path: `config.TEMP_UPLOAD_DIR` must be writeable by the Flask process. Endpoints assume the file exists at `image_data['original_path']` when finalizing sessions.
- Torch device: `config.DEVICE` may be `'cpu'` on dev machines; tests and inference must use `map_location` when loading state_dicts (see `load_all_models()` in `app.py`).
- Hooks: `GradCAM` registers forward and backward hooks on the target layer — confirm target_layer selection when changing architectures.

## Typical small tasks an agent will be asked to do (and how to do them)
- Add a new model: update `models/` with a checkpoint, add a key to `config.MODEL_PATHS`, and update `get_model()` and `get_target_layer()` to support the architecture.
- Fix Grad-CAM sizing or color order bug: operate in `grad_cam.py` (note code converts RGB->BGR before compositing).
- Change class labels: edit `config.CLASS_LABELS` and ensure any DB stored label strings remain compatible or migrate `data_processed/` naming.
- Improve error messages: prefer returning JSON with `{'success': False, 'message': '...'} ` as existing endpoints do.

## Test & verification suggestions (fast)
- Unit tests are not present. For quick verification:
  - Start the server and use the UI to upload a single image to the Curation page.
  - Verify the `/upload` JSON shape (contains `results`, `session_uuid`).
  - Use `/get_explanation` with the returned `temp_filename` to ensure Grad-CAM images are created.

## Do / Don't (agent guardrails)
- Do: Make minimal, reversible edits and prefer adding feature flags or config toggles.
- Don't: Introduce asynchronous server changes, new background workers, or migrate DB schema without an explicit request.
- Don't: Commit secrets — `config.SECRET_KEY` is intentionally a placeholder.

## Where to look for context or more details
- Read `app.py` for request/response shapes and session lifecycle.
- `config.py` for feature flags and thresholds.
- `grad_cam.py` to understand explanation generation.
- `app/templates/` and `app/static/js/main.js` to see the expected AJAX contract.

If anything in this file is unclear or you want it extended with more examples (API payloads, common bug fixes, or a short checklist for PR reviewers), tell me what to add and I'll iterate.
