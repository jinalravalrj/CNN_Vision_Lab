"""
CNN Vision Lab - Flask Backend
Main application file: handles routing, model inference, prediction history.
"""

import os
import io
import time
import json
import sqlite3
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── App config ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DB_PATH    = BASE_DIR / "database" / "predictions.db"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ── Model state ───────────────────────────────────────────────────────────────
MODEL       = None
MODEL_NAME  = "MobileNetV2"
MODEL_TYPE  = "Pretrained – ImageNet"
IMG_SIZE    = (224, 224)
FEAT_MODEL  = None   # intermediate feature-extraction model
MODEL_ERROR = None

EVAL_RESULTS_PATH = BASE_DIR / "models" / "eval_results.json"

# ── Load MobileNetV2 at startup ───────────────────────────────────────────────
def load_model():
    global MODEL, FEAT_MODEL, MODEL_ERROR
    try:
        import tensorflow as tf
        import importlib
        # Support both tf.keras (TF ≤ 2.15) and standalone keras (TF ≥ 2.16)
        keras_app = None
        for mod_name in ("keras.applications.mobilenet_v2", "tensorflow.keras.applications.mobilenet_v2"):
            try:
                keras_app = importlib.import_module(mod_name)
                break
            except ImportError:
                continue

        keras_models = None
        for mod_name in ("keras", "keras.models", "tensorflow.keras.models"):
            try:
                keras_models = importlib.import_module(mod_name)
                if hasattr(keras_models, "Model"):
                    break
            except ImportError:
                continue

        if keras_app is None or not hasattr(keras_app, "MobileNetV2"):
            raise ImportError("Could not locate MobileNetV2 in keras or tensorflow.keras")

        MobileNetV2 = getattr(keras_app, "MobileNetV2")
        KModel = getattr(keras_models, "Model", None)

        log.info("Loading MobileNetV2 with ImageNet weights …")
        MODEL = MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
        MODEL.trainable = False

        # Feature-extraction sub-model: outputs from first conv block
        # Layer name varies slightly by TF version; try multiple candidates.
        feat_layer_name = None
        for candidate in ("Conv1", "Conv1_pad", "expanded_conv_depthwise"):
            try:
                MODEL.get_layer(candidate)
                feat_layer_name = candidate
                break
            except ValueError:
                continue

        if feat_layer_name:
            FEAT_MODEL = KModel(inputs=MODEL.input, outputs=MODEL.get_layer(feat_layer_name).output)
            log.info("Feature-extraction model ready (layer: %s).", feat_layer_name)
        else:
            log.warning("Could not find a suitable feature layer; feature maps disabled.")

        log.info("MobileNetV2 loaded successfully (%d layers).", len(MODEL.layers))
    except Exception as exc:
        MODEL_ERROR = str(exc)
        log.error("Model load failed: %s", exc)


# ── Database helpers ──────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                filename      TEXT    NOT NULL,
                predicted     TEXT    NOT NULL,
                confidence    REAL    NOT NULL,
                top3          TEXT    NOT NULL,
                model_name    TEXT    NOT NULL,
                infer_ms      REAL,
                timestamp     TEXT    NOT NULL
            )
        """)
        conn.commit()
    log.info("Database ready at %s", DB_PATH)


# ── Utility helpers ───────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _keras_app_module():
    """Return the mobilenet_v2 applications module, supporting tf.keras and standalone keras."""
    import importlib
    for mod_name in ("keras.applications.mobilenet_v2", "tensorflow.keras.applications.mobilenet_v2"):
        try:
            return importlib.import_module(mod_name)
        except ImportError:
            continue
    raise ImportError("Could not import mobilenet_v2 module")


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Resize to 224×224, convert to RGB, apply MobileNetV2 preprocessing."""
    mob = _keras_app_module()
    img = pil_img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return mob.preprocess_input(arr)


def decode_predictions_custom(preds: np.ndarray, top: int = 5):
    """Decode ImageNet predictions using Keras's built-in decoder."""
    mob = _keras_app_module()
    decoded = mob.decode_predictions(preds, top=top)[0]
    # Returns list of (class_id, human_label, score)
    return [(label, float(score)) for (_, label, score) in decoded]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# Serve sample images
@app.route("/static/images/samples/<path:filename>")
def sample_image(filename):
    return send_from_directory(str(BASE_DIR / "static" / "images" / "samples"), filename)


# ── API: health ───────────────────────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_name": MODEL_NAME,
        "model_error": MODEL_ERROR,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── API: model info ───────────────────────────────────────────────────────────
@app.route("/api/model-info")
def api_model_info():
    info = {
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "input_shape": list(IMG_SIZE) + [3],
        "output_classes": 1000,
        "total_params": None,
        "trainable_params": None,
        "eval_results": None,
        "eval_source": None,
    }

    if MODEL is not None:
        try:
            total    = int(MODEL.count_params())
            trainable = int(sum(np.prod(v.shape) for v in MODEL.trainable_variables))
            info["total_params"]    = total
            info["trainable_params"] = trainable
        except Exception:
            pass

    if EVAL_RESULTS_PATH.exists():
        try:
            with open(EVAL_RESULTS_PATH) as f:
                info["eval_results"] = json.load(f)
            info["eval_source"] = "custom_training"
        except Exception:
            pass

    return jsonify(info)


# ── API: predict ──────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODEL is None:
        return jsonify({"success": False, "error": "Model not loaded. Check server logs."}), 503

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file type. Please upload JPG, JPEG, or PNG.",
        }), 400

    try:
        img_bytes = file.read()
        if len(img_bytes) > MAX_CONTENT_LENGTH:
            return jsonify({"success": False, "error": "File exceeds 10 MB limit."}), 413

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"success": False, "error": "Could not read image file."}), 400

    try:
        arr = preprocess_image(pil_img)

        t0    = time.perf_counter()
        preds = MODEL.predict(arr, verbose=0)
        infer_ms = round((time.perf_counter() - t0) * 1000, 1)

        top5 = decode_predictions_custom(preds, top=5)
        top3 = [{"label": lbl, "confidence": round(conf * 100, 2)} for lbl, conf in top5[:3]]
        best_label = top3[0]["label"].replace("_", " ")
        best_conf  = top3[0]["confidence"]

        safe_name = secure_filename(file.filename)

        # Save to DB
        ts = datetime.now(timezone.utc).isoformat()
        with get_db() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (filename, predicted, confidence, top3, model_name, infer_ms, timestamp)
                   VALUES (?,?,?,?,?,?,?)""",
                (safe_name, best_label, best_conf, json.dumps(top3), MODEL_NAME, infer_ms, ts),
            )
            conn.commit()

        return jsonify({
            "success":    True,
            "predicted":  best_label,
            "confidence": best_conf,
            "top3":       top3,
            "model_name": MODEL_NAME,
            "infer_ms":   infer_ms,
            "timestamp":  ts,
        })

    except Exception as exc:
        log.error("Prediction error: %s\n%s", exc, traceback.format_exc())
        return jsonify({"success": False, "error": "Prediction failed. See server log."}), 500


# ── API: feature maps ─────────────────────────────────────────────────────────
@app.route("/api/feature-maps", methods=["POST"])
def api_feature_maps():
    """Return base64-encoded activation maps from the first conv layer."""
    if MODEL is None or FEAT_MODEL is None:
        return jsonify({"success": False, "error": "Model not loaded."}), 503

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Unsupported file type."}), 400

    try:
        import base64
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
        arr = preprocess_image(pil_img)

        activations = FEAT_MODEL.predict(arr, verbose=0)   # shape (1, H, W, filters)
        acts = activations[0]                               # (H, W, filters)

        maps = []
        n_show = min(16, acts.shape[-1])
        for i in range(n_show):
            channel = acts[:, :, i]
            mn, mx = channel.min(), channel.max()
            if mx - mn > 1e-8:
                norm = ((channel - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                norm = np.zeros_like(channel, dtype=np.uint8)
            pil_map = Image.fromarray(norm, mode="L").resize((96, 96), Image.NEAREST)
            buf = io.BytesIO()
            pil_map.save(buf, format="PNG")
            maps.append(base64.b64encode(buf.getvalue()).decode())

        return jsonify({
            "success": True,
            "layer_name": "Conv1",
            "shape": list(acts.shape),
            "maps": maps,
        })
    except Exception as exc:
        log.error("Feature-map error: %s", exc)
        return jsonify({"success": False, "error": "Feature extraction failed."}), 500


# ── API: history ──────────────────────────────────────────────────────────────
@app.route("/api/history")
def api_history():
    limit  = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        records = []
        for r in rows:
            rec = dict(r)
            rec["top3"] = json.loads(rec["top3"])
            records.append(rec)
        return jsonify({"success": True, "records": records, "total": total})
    except Exception as exc:
        log.error("History fetch error: %s", exc)
        return jsonify({"success": False, "error": "Could not fetch history."}), 500


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def api_delete_record(record_id: int):
    try:
        with get_db() as conn:
            cur = conn.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
            conn.commit()
        if cur.rowcount == 0:
            return jsonify({"success": False, "error": "Record not found."}), 404
        return jsonify({"success": True, "deleted_id": record_id})
    except Exception as exc:
        log.error("Delete error: %s", exc)
        return jsonify({"success": False, "error": "Could not delete record."}), 500


@app.route("/api/history", methods=["DELETE"])
def api_clear_history():
    confirm = request.args.get("confirm", "").lower()
    if confirm != "yes":
        return jsonify({
            "success": False,
            "error": "Pass ?confirm=yes to clear all history.",
        }), 400
    try:
        with get_db() as conn:
            conn.execute("DELETE FROM predictions")
            conn.commit()
        return jsonify({"success": True, "message": "All history cleared."})
    except Exception as exc:
        log.error("Clear history error: %s", exc)
        return jsonify({"success": False, "error": "Could not clear history."}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    load_model()
    log.info("Starting CNN Vision Lab on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
