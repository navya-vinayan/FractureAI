
import os
import json
import uuid
import sqlite3
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv

# Load environment variables before importing chatbot module
load_dotenv()

from rag.chatbot import ask_chatbot

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, send_from_directory, jsonify
)

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename


# ---------------------------------------------------------
# 1. PROJECT PATHS AND FLASK CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "FractureAI_Final.keras")
THRESHOLD_PATH = os.path.join(BASE_DIR, "model", "FractureAI_threshold.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DATABASE_PATH = os.path.join(BASE_DIR, "fractureai.db")

HOSPITAL_DATA_PATH = os.path.join(
    BASE_DIR, "kerala_hospitals.json"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "change-this-development-secret-before-deployment"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# ---------------------------------------------------------
# 2. LOAD THE TRAINED FRACTURE CLASSIFICATION MODEL
# ---------------------------------------------------------

model = None
model_error = None
threshold = 0.43699625

try:

    if os.path.exists(THRESHOLD_PATH):

        with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
            threshold_data = json.load(file)

        threshold = float(
            threshold_data.get("threshold", threshold)
        )

    if os.path.exists(MODEL_PATH):

        model = tf.keras.models.load_model(MODEL_PATH)

        print("FractureAI model loaded successfully.")

    else:

        model_error = f"Model file not found: {MODEL_PATH}"
        print(model_error)

except Exception as exc:

    model_error = str(exc)
    print("Model loading error:", model_error)


# ---------------------------------------------------------
# 3. DATABASE FUNCTIONS
# ---------------------------------------------------------

def get_db():

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    with get_db() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_name TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL,
                threshold REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                doctor_name TEXT NOT NULL,
                appointment_date TEXT NOT NULL,
                appointment_time TEXT NOT NULL,
                reason TEXT,
                status TEXT NOT NULL DEFAULT 'Request received',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)


init_db()


# ---------------------------------------------------------
# 4. LOGIN PROTECTION
# ---------------------------------------------------------

def login_required(view):

    @wraps(view)
    def wrapped(*args, **kwargs):

        if "user_id" not in session:

            flash("Please log in to continue.", "warning")

            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped


# ---------------------------------------------------------
# 5. FILE VALIDATION
# ---------------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ---------------------------------------------------------
# 6. GRAD-CAM VISUALIZATION
# ---------------------------------------------------------

def gradcam_overlay(image_array, original_image, output_path):

    """
    Generate an illustrative Grad-CAM overlay.
    It is not an exact fracture location map.
    """

    if model is None:
        raise RuntimeError("The classification model is not loaded.")

    base_model = model.get_layer("efficientnetb0")

    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer("top_conv").output,
            base_model.output
        ]
    )

    base_index = model.layers.index(base_model)
    head_layers = model.layers[base_index + 1:]

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, features = feature_model(
            image_tensor,
            training=False
        )

        tape.watch(conv_outputs)

        x = features

        for layer in head_layers:
            x = layer(x, training=False)

        score = x[:, 0]

    gradients = tape.gradient(score, conv_outputs)

    if gradients is None:

        raise RuntimeError(
            "Could not calculate Grad-CAM gradients."
        )

    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_map = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_map * weights,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = cv2.resize(
        heatmap.numpy(),
        (original_image.width, original_image.height)
    )

    heatmap_u8 = np.uint8(255 * heatmap)

    colored = cv2.applyColorMap(
        heatmap_u8,
        cv2.COLORMAP_JET
    )

    original = np.array(
        original_image.convert("RGB")
    )

    overlay = cv2.addWeighted(
        original,
        0.60,
        colored,
        0.40,
        0
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )


# ---------------------------------------------------------
# 7. HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def index():

    return render_template("index.html")


# ---------------------------------------------------------
# 8. USER REGISTRATION
# ---------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or len(password) < 8:

            flash(
                "Enter your name, a valid email and a password "
                "with at least 8 characters.",
                "danger"
            )

            return render_template("register.html")

        try:

            with get_db() as conn:

                conn.execute(
                    """
                    INSERT INTO users (name, email, password_hash)
                    VALUES (?, ?, ?)
                    """,
                    (
                        name,
                        email,
                        generate_password_hash(password)
                    )
                )

            flash(
                "Account created. Please log in.",
                "success"
            )

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            flash(
                "An account with that email already exists.",
                "danger"
            )

    return render_template("register.html")


# ---------------------------------------------------------
# 9. USER LOGIN
# ---------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with get_db() as conn:

            user = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            ).fetchone()

        if user and check_password_hash(
            user["password_hash"],
            password
        ):

            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]

            return redirect(url_for("dashboard"))

        flash(
            "Email or password is incorrect.",
            "danger"
        )

    return render_template("login.html")


# ---------------------------------------------------------
# 10. USER LOGOUT
# ---------------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "info"
    )

    return redirect(url_for("index"))


# ---------------------------------------------------------
# 11. USER DASHBOARD
# ---------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():

    with get_db() as conn:

        analyses = conn.execute(
            """
            SELECT * FROM analyses
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 5
            """,
            (session["user_id"],)
        ).fetchall()

        appointments = conn.execute(
            """
            SELECT * FROM appointments
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 5
            """,
            (session["user_id"],)
        ).fetchall()

    return render_template(
        "dashboard.html",
        analyses=analyses,
        appointments=appointments,
        model_ready=(model is not None),
        model_error=model_error
    )


# ---------------------------------------------------------
# 12. X-RAY ANALYSIS
# ---------------------------------------------------------

@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():

    if request.method == "POST":

        if model is None:

            flash(
                "The AI model is not available. Check the model file.",
                "danger"
            )

            return redirect(url_for("analyze"))

        file = request.files.get("xray")

        if not file or file.filename == "":

            flash(
                "Please choose an X-ray image.",
                "warning"
            )

            return redirect(url_for("analyze"))

        if not allowed_file(file.filename):

            flash(
                "Please upload a PNG, JPG or JPEG image.",
                "danger"
            )

            return redirect(url_for("analyze"))

        safe_name = secure_filename(file.filename)

        unique_name = f"{uuid.uuid4().hex}_{safe_name}"

        image_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )

        file.save(image_path)

        try:

            original = Image.open(image_path).convert("RGB")
            resized = original.resize((224, 224))

            image_array = np.expand_dims(
                np.asarray(resized, dtype=np.float32),
                axis=0
            )

            probability = float(
                model.predict(
                    image_array,
                    verbose=0
                )[0][0]
            )

            prediction = (
                "Fractured"
                if probability >= threshold
                else "Non-Fractured"
            )

            overlay_name = (
                f"gradcam_{unique_name.rsplit('.', 1)[0]}.jpg"
            )

            overlay_path = os.path.join(
                UPLOAD_FOLDER,
                overlay_name
            )

            gradcam_message = None

            try:

                gradcam_overlay(
                    image_array,
                    original,
                    overlay_path
                )

            except Exception as exc:

                gradcam_message = (
                    f"Grad-CAM could not be generated: {exc}"
                )

                overlay_name = None

            with get_db() as conn:

                cursor = conn.execute(
                    """
                    INSERT INTO analyses (
                        user_id,
                        image_name,
                        prediction,
                        probability,
                        threshold,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session["user_id"],
                        unique_name,
                        prediction,
                        probability,
                        threshold,
                        datetime.now().strftime("%Y-%m-%d %H:%M")
                    )
                )

                analysis_id = cursor.lastrowid

            return render_template(
                "analysis.html",
                prediction=prediction,
                probability=probability,
                threshold=threshold,
                image_name=unique_name,
                overlay_name=overlay_name,
                analysis_id=analysis_id,
                gradcam_message=gradcam_message
            )

        except Exception as exc:

            flash(
                f"Could not process this image: {exc}",
                "danger"
            )

            return redirect(url_for("analyze"))

    return render_template(
        "analyze.html",
        model_ready=(model is not None),
        model_error=model_error
    )


# ---------------------------------------------------------
# 13. DISPLAY UPLOADED IMAGES
# ---------------------------------------------------------

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ---------------------------------------------------------
# 14. DOCTOR APPOINTMENT REQUESTS
# ---------------------------------------------------------

@app.route("/appointments", methods=["GET", "POST"])
@login_required
def appointments():

    if request.method == "POST":

        doctor = request.form.get("doctor", "").strip()
        appointment_date = request.form.get("date", "").strip()
        appointment_time = request.form.get("time", "").strip()
        reason = request.form.get("reason", "").strip()

        if not doctor or not appointment_date or not appointment_time:

            flash(
                "Please complete the doctor, date and time fields.",
                "warning"
            )

        else:

            with get_db() as conn:

                conn.execute(
                    """
                    INSERT INTO appointments (
                        user_id,
                        doctor_name,
                        appointment_date,
                        appointment_time,
                        reason,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session["user_id"],
                        doctor,
                        appointment_date,
                        appointment_time,
                        reason,
                        datetime.now().strftime("%Y-%m-%d %H:%M")
                    )
                )

            flash(
                "Appointment request submitted. "
                "This is a request, not a confirmed booking.",
                "success"
            )

            return redirect(url_for("appointments"))

    with get_db() as conn:

        items = conn.execute(
            """
            SELECT * FROM appointments
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (session["user_id"],)
        ).fetchall()

    return render_template(
        "appointments.html",
        appointments=items
    )


# ---------------------------------------------------------
# 15. FRACTUREAI CHATBOT - QWEN + RAG
# ---------------------------------------------------------

@app.route("/chat", methods=["GET", "POST"])
@login_required
def chat():

    if request.method == "GET":

        return render_template("chat.html")

    data = request.get_json(silent=True) or {}

    message = data.get("message", "")

    if not isinstance(message, str):

        return jsonify({
            "reply": "Please enter your question as text."
        }), 400

    message = message.strip()

    if not message:

        return jsonify({
            "reply": "Please enter a question."
        }), 400

    try:

        result = ask_chatbot(message)

        answer = result.get(
            "answer",
            "Sorry, I could not generate an answer."
        )

        sources = result.get("sources", [])

        reply = answer

        if sources:

            reply += "\n\nSources:\n"

            reply += "\n".join(
                f"- {source}" for source in sources
            )

        return jsonify({"reply": reply})

    except Exception:

        app.logger.exception(
            "FractureAI Qwen + RAG chatbot error"
        )

        return jsonify({
            "reply": (
                "The FractureAI AI chatbot is temporarily unavailable. "
                "Please try again later."
            )
        }), 503


# ---------------------------------------------------------
# 16. LOAD ALL KERALA HOSPITALS FROM JSON
# ---------------------------------------------------------

def load_kerala_hospitals():

    if not os.path.exists(HOSPITAL_DATA_PATH):

        raise FileNotFoundError(
            "kerala_hospitals.json was not found beside app.py."
        )

    with open(
        HOSPITAL_DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    hospital_records = []

    ownership_labels = {
        "government": "Government Hospital",
        "private": "Private Hospital",
        "trust": "Trust / Charitable Hospital",
        "trust/charitable": "Trust / Charitable Hospital",
        "co-operative": "Co-operative Hospital",
        "cooperative": "Co-operative Hospital"
    }

    for item in data.get("hospitals", []):

        ownership = str(
            item.get("o") or "not specified"
        ).strip()

        category = str(
            item.get("k") or "Hospital"
        ).strip()

        ownership_label = ownership_labels.get(
            ownership.lower(),
            ownership.replace("_", " ").title()
        )

        departments = item.get("dep") or []

        if isinstance(departments, str):
            departments = [departments]

        orthopaedics_found = any(
            "orthop" in str(department).lower()
            for department in departments
        )

        if orthopaedics_found:

            orthopaedics = "Orthopaedics listed in source"

        else:

            orthopaedics = (
                "Not listed in source; contact hospital to confirm"
            )

        if item.get("e") is True:

            emergency = "24-hour emergency listed in source"

        else:

            emergency = (
                "Not confirmed in source; contact hospital"
            )

        address_parts = [
            item.get("t"),
            item.get("d"),
            "Kerala",
            "India"
        ]

        address = ", ".join(
            str(part)
            for part in address_parts
            if part
        )

        hospital = {

            "id": item.get("id"),

            "name": item.get("n") or "Hospital name unavailable",

            "district": item.get("d") or "Not specified",

            "town": item.get("t") or "",

            "type": ownership_label + " - " + category,

            "category": category,

            "ownership": ownership,

            "address": address,

            "phone": item.get("ph") or "",

            "website": item.get("w") or "",

            "orthopaedics": orthopaedics,

            "op_time": item.get("op_hours") or "Not listed in source",

            "emergency": emergency,

            "emergency_phone": item.get("eph") or "",

            "emergency_24x7": item.get("e"),

            "source": "Adserr Kerala hospital directory",

            "image": "images/hospitals/default_hospital.jpg",

            "departments": departments,

            "beds": item.get("b"),

            "maps_url": item.get("m") or "",

            "pincode": item.get("p") or ""

        }

        hospital_records.append(hospital)

    return hospital_records


# ---------------------------------------------------------
# 17. KERALA HOSPITAL DIRECTORY
# ---------------------------------------------------------

@app.route("/hospitals")
def hospitals():

    try:

        all_hospitals = load_kerala_hospitals()

    except FileNotFoundError as exc:

        app.logger.error(str(exc))

        return (
            "Hospital dataset not found. "
            "Please place kerala_hospitals.json in the same folder as app.py."
        ), 500

    except (json.JSONDecodeError, KeyError, TypeError) as exc:

        app.logger.exception("Hospital dataset loading error")

        return f"Could not read the hospital dataset: {exc}", 500

    selected_district = request.args.get(
        "district", ""
    ).strip()

    search_text = request.args.get(
        "search", ""
    ).strip().lower()

    selected_type = request.args.get(
        "type", ""
    ).strip()

    selected_ownership = request.args.get(
        "ownership", ""
    ).strip().lower()

    # Build district list from the entire dataset
    districts = sorted({
        hospital["district"]
        for hospital in all_hospitals
        if hospital["district"] != "Not specified"
    })

    # Build hospital category list
    hospital_types = sorted({
        hospital["category"]
        for hospital in all_hospitals
        if hospital["category"]
    })

    # Build ownership list
    ownership_types = sorted({
        hospital["ownership"]
        for hospital in all_hospitals
        if hospital["ownership"]
    })

    filtered_hospitals = []

    for hospital in all_hospitals:

        # District filter
        district_match = (
            not selected_district
            or hospital["district"].lower() == selected_district.lower()
        )

        # Category filter
        type_match = (
            not selected_type
            or hospital["category"].lower() == selected_type.lower()
        )

        # Ownership filter
        ownership_match = (
            not selected_ownership
            or hospital["ownership"].lower() == selected_ownership
        )

        # Search by name, town, district, ownership, category or department
        searchable_text = " ".join([
            hospital["name"],
            hospital["district"],
            hospital["town"],
            hospital["address"],
            hospital["type"],
            hospital["ownership"],
            hospital["category"],
            " ".join(hospital["departments"])
        ]).lower()

        search_match = (
            not search_text
            or search_text in searchable_text
        )

        if (
            district_match
            and type_match
            and ownership_match
            and search_match
        ):

            filtered_hospitals.append(hospital)

    return render_template(
        "hospitals.html",
        hospitals=filtered_hospitals,
        districts=districts,
        hospital_types=hospital_types,
        ownership_types=ownership_types,
        selected_district=selected_district,
        selected_type=selected_type,
        selected_ownership=selected_ownership,
        search_text=search_text,
        total_hospitals=len(all_hospitals)
    )


# ---------------------------------------------------------
# 18. HEALTH CHECK
# ---------------------------------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "FractureAI Flask application is running",
        "model_loaded": model is not None,
        "model_error": model_error
    })


# ---------------------------------------------------------
# 19. START THE FLASK APPLICATION
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
