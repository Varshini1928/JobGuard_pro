"""
app.py — Flask Backend for JobGuard Pro
Integrates: TF-IDF + Logistic Regression model, OCR (Tesseract), and Frontend HTML
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
import re
import pytesseract
from PIL import Image
import io
import base64
import traceback

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Model paths
MODEL_PATH      = 'fake_job_detector.pkl'
VECTORIZER_PATH = 'text_vectorizer.pkl'

# ─────────────────────────────────────────────────────────────
# LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────────────────────

model     = None
vectorizer = None
MODEL_READY = False

def load_model():
    global model, vectorizer, MODEL_READY
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"[WARN] Model not found at '{MODEL_PATH}'.")
            print("[INFO] Run your training script first to generate the model.")
            return

        if not os.path.exists(VECTORIZER_PATH):
            print(f"[WARN] Vectorizer not found at '{VECTORIZER_PATH}'.")
            return

        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        MODEL_READY = True
        print("[INFO] Model and vectorizer loaded successfully.")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        MODEL_READY = False

load_model()


# ─────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────

# Fake and real word signals for indicator counting
FAKE_SIGNALS = [
    "upfront payment", "registration fee", "work from home",
    "no experience needed", "earn money fast", "get rich quick",
    "guaranteed income", "send money", "bank details", "urgently hiring",
    "immediate start", "limited time", "payment required", "cash transfer",
    "pay to start", "one-time fee", "investment required", "quick cash",
    "western union", "money transfer", "paypal payment", "high earnings",
    "no qualifications", "simple tasks", "easy money", "big money",
    "part time", "work online", "online job", "daily pay"
]

REAL_SIGNALS = [
    "competitive salary", "benefits package", "health insurance",
    "401k", "professional development", "required qualifications",
    "experience needed", "background check", "interview process",
    "team collaboration", "degree required", "responsibilities include",
    "performance review", "training provided", "career growth",
    "technical skills", "collaborate with", "mentor", "code review",
    "annual bonus", "paid time off", "retirement plan", "dental insurance",
    "remote work", "company culture", "full-time position"
]


def clean_text(text: str) -> str:
    """Clean and normalize text for model input."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_signals(text: str) -> dict:
    """Count fake and real indicator words in text."""
    text_lower = text.lower()
    fake_count = sum(1 for signal in FAKE_SIGNALS if signal in text_lower)
    real_count = sum(1 for signal in REAL_SIGNALS if signal in text_lower)
    return {"fake": fake_count, "real": real_count}


def extract_title(text: str) -> str:
    """Extract job title from first line of text."""
    first_line = text.strip().split('\n')[0].strip()
    return first_line[:60] + ('...' if len(first_line) > 60 else '') or "Analyzed Job Posting"


def predict_text(text: str) -> dict:
    """
    Run prediction on cleaned job text.
    Returns label, confidence, indicators, and details.
    """
    if not MODEL_READY:
        raise RuntimeError("Model not loaded. Please train the model first.")

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Map prediction to label
    label    = "Fake" if prediction == 1 else "Real"
    label_id = int(prediction)

    # Confidence = probability of predicted class
    confidence = round(float(max(probabilities)) * 100, 1)
    real_prob  = round(float(probabilities[0]) * 100, 1)
    fake_prob  = round(float(probabilities[1]) * 100, 1)

    # Signal analysis
    signals = count_signals(text)

    # Build details string
    if label == "Fake":
        details = (
            f"Detected {signals['fake']} potential scam indicators. "
            "This job posting shows characteristics commonly associated with fraudulent opportunities."
        )
    else:
        details = (
            f"Detected {signals['real']} legitimate job indicators. "
            "This job posting appears to be from a genuine employer."
        )

    return {
        "classification": label,
        "label_id":       label_id,
        "confidence":     confidence,
        "probabilities":  {"real": real_prob, "fake": fake_prob},
        "indicators":     signals,
        "details":        details,
        "title":          extract_title(text),
        "model":          "TF-IDF + Logistic Regression",
    }


# ─────────────────────────────────────────────────────────────
# ROUTES — SERVE FRONTEND
# ─────────────────────────────────────────────────────────────

@app.route('/')
def serve_frontend():
    """Serve the main HTML frontend."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, images)."""
    full_path = os.path.join(app.static_folder, path)
    if os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


# ─────────────────────────────────────────────────────────────
# ROUTES — API
# ─────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    """Health check — tells frontend if model is ready."""
    return jsonify({
        "status":       "ok",
        "model_ready":  MODEL_READY,
        "model_type":   "TF-IDF + Logistic Regression",
        "message":      "JobGuard Pro API is running",
    })


# ── PREDICT FROM TEXT ─────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict if a job posting is real or fake from text input.

    Accepts:
      { "text": "Full job posting..." }
    OR structured:
      { "title": "...", "description": "...", "requirements": "...", ... }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # Build text from raw or structured input
    if "text" in data and data["text"].strip():
        text = data["text"].strip()
    else:
        fields = ["title", "company", "description", "requirements", "location", "salary", "benefits"]
        parts  = [str(data.get(f, "")).strip() for f in fields if data.get(f, "").strip()]
        text   = " ".join(parts).strip()

    if not text:
        return jsonify({"error": "No text content provided"}), 400

    if len(text) > 10000:
        return jsonify({"error": "Text too long. Maximum 10,000 characters."}), 400

    if len(text) < 30:
        return jsonify({"error": "Text too short. Please provide more details."}), 400

    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Run training script first."}), 503

    try:
        result = predict_text(text)
        result["mode"] = data.get("mode", "text")
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── PREDICT FROM IMAGE (OCR) ──────────────────────────────────
@app.route('/api/predict/image', methods=['POST'])
def predict_image():
    """
    Extract text from uploaded image using Tesseract OCR, then predict.

    Accepts multipart/form-data with 'image' file field.
    OR JSON with base64-encoded image: { "image_base64": "..." }
    """
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Run training script first."}), 503

    extracted_text = ""

    try:
        # ── Handle file upload ──
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename == '':
                return jsonify({"error": "No image file selected"}), 400

            image = Image.open(image_file.stream)

        # ── Handle base64 image ──
        elif request.is_json and 'image_base64' in request.get_json():
            data       = request.get_json()
            image_data = base64.b64decode(data['image_base64'])
            image      = Image.open(io.BytesIO(image_data))

        else:
            return jsonify({"error": "No image provided. Send 'image' file or 'image_base64' field."}), 400

        # ── OCR Extraction ──
        extracted_text = pytesseract.image_to_string(image, lang='eng')
        extracted_text = extracted_text.strip()

        if not extracted_text or len(extracted_text) < 20:
            return jsonify({
                "error":          "Could not extract meaningful text from image.",
                "extracted_text": extracted_text,
                "tip":            "Try a clearer image with higher resolution and visible text.",
            }), 422

        # ── Run Prediction ──
        result = predict_text(extracted_text)
        result["extracted_text"] = extracted_text
        result["mode"]           = "image"
        return jsonify(result)

    except pytesseract.TesseractNotFoundError:
        return jsonify({
            "error": "Tesseract OCR is not installed or path is incorrect.",
            "fix":   "Install Tesseract and update 'tesseract_cmd' path in app.py",
        }), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500


# ── PREDICT COMBINED (TEXT + IMAGE) ───────────────────────────
@app.route('/api/predict/combined', methods=['POST'])
def predict_combined():
    """
    Predict using combined text input AND image OCR.

    Accepts multipart/form-data:
      - 'text' field: manual text
      - 'image' file: job poster image (optional)
    """
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Run training script first."}), 503

    manual_text    = request.form.get('text', '').strip()
    extracted_text = ""

    # ── OCR if image provided ──
    if 'image' in request.files:
        image_file = request.files['image']
        if image_file.filename != '':
            try:
                image          = Image.open(image_file.stream)
                extracted_text = pytesseract.image_to_string(image, lang='eng').strip()
            except Exception as e:
                print(f"[WARN] OCR failed: {e}")

    # ── Combine texts ──
    combined_text = " ".join(filter(None, [manual_text, extracted_text])).strip()

    if not combined_text:
        return jsonify({"error": "No text content found. Please provide text or a readable image."}), 400

    if len(combined_text) < 30:
        return jsonify({"error": "Combined text too short. Add more details."}), 400

    try:
        result = predict_text(combined_text)
        result["mode"]           = "both"
        result["manual_text"]    = manual_text
        result["extracted_text"] = extracted_text
        result["combined_text"]  = combined_text
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ── BATCH PREDICT ──────────────────────────────────────────────
@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple job postings at once.

    Accepts: { "postings": [ { "text": "..." }, ... ] }
    Maximum 20 postings per request.
    """
    data = request.get_json()
    if not data or "postings" not in data:
        return jsonify({"error": "Expected JSON: { 'postings': [...] }"}), 400

    postings = data["postings"]
    if len(postings) > 20:
        return jsonify({"error": "Maximum 20 postings per batch request"}), 400

    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Run training script first."}), 503

    results = []
    for i, posting in enumerate(postings):
        text = posting.get("text", "").strip()
        if not text:
            results.append({"index": i, "error": "Empty text"})
            continue
        try:
            r          = predict_text(text)
            r["index"] = i
            results.append(r)
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return jsonify({"results": results, "total": len(results)})


# ── OCR ONLY (extract text, no prediction) ────────────────────
@app.route('/api/ocr', methods=['POST'])
def ocr_only():
    """
    Extract text from image using Tesseract OCR without prediction.
    Useful for the frontend to preview extracted text before scanning.

    Accepts multipart/form-data with 'image' file.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        image          = Image.open(image_file.stream)
        extracted_text = pytesseract.image_to_string(image, lang='eng').strip()

        if not extracted_text:
            return jsonify({
                "extracted_text": "",
                "warning": "No text detected. Try a clearer image.",
            })

        return jsonify({
            "extracted_text":  extracted_text,
            "character_count": len(extracted_text),
            "word_count":      len(extracted_text.split()),
        })

    except pytesseract.TesseractNotFoundError:
        return jsonify({
            "error": "Tesseract is not installed or path is wrong.",
            "fix":   "Set correct tesseract_cmd path in app.py",
        }), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500


# ── MODEL INFO ────────────────────────────────────────────────
@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Return information about the loaded model."""
    info = {
        "model_ready":  MODEL_READY,
        "model_type":   "TF-IDF Vectorizer + Logistic Regression",
        "model_file":   MODEL_PATH,
        "vectorizer":   VECTORIZER_PATH,
        "max_features": 10000,
        "ngram_range":  "(1, 2)",
        "classes":      ["Real (0)", "Fake (1)"],
    }

    if MODEL_READY and vectorizer is not None:
        info["vocabulary_size"] = len(vectorizer.vocabulary_)

    return jsonify(info)


# ── SAMPLE EXAMPLES ───────────────────────────────────────────
@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Return sample job postings for frontend demo."""
    return jsonify([
        {
            "id":    1,
            "type":  "real",
            "title": "Senior Software Engineer",
            "text":  (
                "We are hiring a Senior Software Engineer with 5+ years of experience in Java and Spring Boot. "
                "Responsibilities: Develop scalable microservices, collaborate with cross-functional teams, "
                "mentor junior developers, and conduct code reviews. "
                "Requirements: Bachelor degree in Computer Science, 5+ years software development experience, "
                "strong knowledge of Java Spring Boot REST APIs, experience with AWS Azure and containerization. "
                "Benefits: Competitive salary with performance bonuses, comprehensive health insurance, "
                "401k matching program, flexible work hours and remote work options, professional development budget. "
                "Interview process: Phone screening, technical assessment, on-site interviews, reference checks."
            )
        },
        {
            "id":    2,
            "type":  "fake",
            "title": "Work From Home - Earn $5000/Week!!!",
            "text":  (
                "URGENT HIRING! WORK FROM HOME - NO EXPERIENCE NEEDED! "
                "Earn $5000 per month working just 2 to 3 hours daily from your home! "
                "No previous experience required, we provide full training! "
                "IMMEDIATE START AVAILABLE! Limited positions, apply NOW! "
                "TO GET STARTED: Send your resume to hiring@quickcashjobs.com "
                "Pay the ONE-TIME registration fee of $49.99 to receive your training materials. "
                "Payment methods accepted: PayPal, Western Union, Bank Transfer. "
                "DON'T MISS THIS LIMITED TIME OPPORTUNITY! Earn money FAST with minimal effort!"
            )
        },
        {
            "id":    3,
            "type":  "real",
            "title": "Data Analyst",
            "text":  (
                "We are looking for a Data Analyst to join our business intelligence team. "
                "You will analyze large datasets, build dashboards in Tableau and Power BI, "
                "and provide actionable insights to stakeholders. "
                "Requirements: Bachelor degree in Statistics or Computer Science, "
                "2 plus years experience with SQL and Python, strong communication skills. "
                "We provide health insurance, paid time off, annual bonus, and career development programs. "
                "Salary range 75000 to 95000. Background check required. Interview process involved."
            )
        },
        {
            "id":    4,
            "type":  "fake",
            "title": "Part-time Data Entry Clerk",
            "text":  (
                "Make easy money from home doing simple data entry! "
                "Earn $200 per day with no skills required. Part time flexible hours. "
                "We will send you a check, just deposit it and send back a portion via wire transfer. "
                "No interview needed. Immediate start. "
                "Send your SSN and credit card number for background verification. "
                "Guaranteed daily pay! Investment required to get started. Quick cash opportunity!"
            )
        }
    ])


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  JobGuard Pro — Flask API")
    print("=" * 60)
    print(f"  Model Ready  : {MODEL_READY}")
    print(f"  Model File   : {MODEL_PATH}")
    print(f"  Frontend     : ../frontend/index.html")
    print(f"  API Base URL : http://localhost:5000/api/")
    print("=" * 60)

    if not MODEL_READY:
        print("\n  [!] WARNING: Model not loaded.")
        print("  [!] Run your training script first:")
        print("      python train_model.py")
        print()

    app.run(debug=True, port=5000, host='0.0.0.0')
