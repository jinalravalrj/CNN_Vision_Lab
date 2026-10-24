# CNN Vision Lab

> **Interactive CNN Image Classification & Architecture Visualization System**  
> Full-Stack Computer Vision Application demonstrating Convolutional Neural Networks, Deep Learning Inference, Feature Map Activations, and Architecture Analytics.

[![GitHub Repository](https://img.shields.io/badge/GitHub-jinalravalrj%2FCNN__Vision__Lab-181717?style=flat&logo=github)](https://github.com/jinalravalrj/CNN_Vision_Lab)
[![Python](https://img.shields.io/badge/Python-3.10%20--%203.14-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%20%2F%202.22-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

**CNN Vision Lab** is an interactive, full-stack educational and practical deep learning workbench designed to make Convolutional Neural Networks accessible, interpretable, and visual.

The application allows users to upload any real-world image or select built-in sample images to run real-time inference using a **MobileNetV2** deep convolutional neural network, inspect intermediate convolution layer feature map activations, explore layer-by-layer architectural flow diagrams, review past inference history saved in an SQLite database, and test their conceptual knowledge via integrated learning modules and quizzes.

### Key Capabilities
- **Real-Time Image Classification**: Predicts across 1,000 ImageNet categories with top-3 probabilistic confidence scores and response latency profiling.
- **Intermediate Feature-Map Visualization**: Extracts activations from the model's early convolutional blocks (`Conv1`) and renders filter response heatmaps.
- **Interactive CNN Architecture Viewer**: Interactive diagram mapping out input tensor shapes, convolutional filters, batch normalization, inverted residual bottlenecks, global average pooling, and softmax dense classification layers.
- **Prediction Analytics & History**: Persistent logging of previous predictions, timestamps, confidence scores, and filenames with SQLite persistence and RESTful management (view, delete, bulk-clear).
- **Custom CNN Training Pipeline**: Dedicated modular training script (`training/train_cnn.py`) supporting user-provided categorical image datasets (e.g., Cats vs. Dogs subset), generating accuracy/loss curves and test evaluation reports.
- **Interactive Learning & Viva Q&A**: 10 educational modules covering core neural network topics and an interactive assessment quiz.

---

## 🏗️ CNN Architecture & Inference Workflow

### MobileNetV2 Architecture Summary
| Property | Specification |
|---|---|
| **Base Model** | MobileNetV2 (Sandler et al., Google Research) |
| **Pretrained Weights** | ImageNet-1k (1,000 classes) |
| **Input Tensor** | $224 \times 224 \times 3$ (RGB) |
| **Output Layer** | 1,000-dimensional Dense layer with Softmax activation |
| **Total Parameters** | $\approx 3.53\text{M}$ parameters (100% inference mode) |
| **Top-1 / Top-5 Benchmark** | 72.0% Top-1 / 90.6% Top-5 |

### Execution Workflow
1. **Input Preprocessing**: User uploads an image (`.jpg`, `.jpeg`, `.png`, max 10 MB). The image is verified with Pillow, resized to $224 \times 224 \times 3$, converted to float32 NumPy arrays, and preprocessed using MobileNetV2 scaling (values scaled to $[-1, 1]$).
2. **Forward Inference**: The input passes through MobileNetV2 forward propagation.
3. **Softmax Output**: ImageNet class index predictions are decoded into human-readable label names and percentage probabilities.
4. **Feature-Map Extraction**: A sub-model extracts activation maps from the primary convolutional layer (`Conv1`), normalizes activation outputs, and returns them as encoded image matrices.
5. **History Logging**: Prediction details, top class name, top confidence, processing time, and timestamp are committed to `database/predictions.db`.

```
[ User Image ] 
      │
      ▼
[ Validation & Preprocessing (224×224×3, RGB) ]
      │
      ├───────────────────────────────────────────┐
      ▼                                           ▼
[ MobileNetV2 Backbone ]                 [ Intermediate Conv1 Layer ]
      │                                           │
      ▼                                           ▼
[ Global Avg Pooling & Softmax ]         [ 16 Feature Activation Maps ]
      │                                           │
      ▼                                           ▼
[ Top-3 Predictions & Confidence ]       [ Frontend Visual Activation Grid ]
      │
      ▼
[ SQLite predictions.db Record ]
```

---

## 💻 Tech Stack & Dependencies

- **Backend**: Python 3.10 – 3.14, Flask 3.0.3, Flask-CORS 4.0.1, Werkzeug 3.0.3, SQLite3
- **Machine Learning**: TensorFlow (v2.16.1 for Python 3.10–3.12, v2.22.0rc0 for Python 3.14), Keras
- **Image Processing & Math**: Pillow 11.2.1, NumPy >= 1.26.4, Matplotlib >= 3.8.0
- **Frontend**: HTML5, Vanilla JavaScript (ES6+), Vanilla CSS3 (Custom Dark Teal Glassmorphic UI)

---

## 📂 Project Structure

```
cnn-vision-lab/
├── .gitignore                    # Excludes venv, pycache, temporary uploads & caches
├── app.py                        # Main Flask server & inference API endpoints
├── generate_samples.py           # Sample test image generator (cats, dogs, cars, etc.)
├── requirements.txt              # Project Python dependencies
├── README.md                     # Comprehensive project documentation
│
├── database/
│   └── predictions.db            # SQLite database file (created/maintained on startup)
│
├── models/
│   ├── .gitkeep                  # Preserves directory in version control
│   ├── eval_results.json         # Evaluation metrics from custom training
│   └── custom_cnn/               # SavedModel output directory for custom training
│
├── static/
│   ├── css/
│   │   └── style.css             # Responsive theme, typography, and layout styles
│   ├── js/
│   │   └── app.js                # Frontend SPA controller and API client
│   ├── images/
│   │   └── samples/              # Built-in demo sample images
│   └── uploads/
│       └── .gitkeep              # Runtime directory for temporary upload processing
│
├── templates/
│   └── index.html                # Single-page user interface template
│
├── tests/
│   └── test_api.py               # REST API integration and smoke test suite
│
└── training/
    ├── train_cnn.py              # Optional standalone CNN training script
    ├── data/                     # Custom dataset folder (train/val/test splits)
    ├── history.json              # Training history metric logs
    └── plots/                    # Loss and accuracy curve charts
```

---

## ⚡ Installation & Setup Instructions (Windows)

### Prerequisites
- Windows 10 or 11
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed with `Add Python to PATH` checked.
- Git for Windows installed.

### Step 1 — Clone the Repository

```powershell
git clone https://github.com/jinalravalrj/CNN_Vision_Lab.git
cd CNN_Vision_Lab
```

### Step 2 — Create and Activate a Virtual Environment

```powershell
python -m venv venv
venv\Scripts\activate
```

*(You should see `(venv)` preceding your terminal command prompt).*

### Step 3 — Install Dependencies

For Python 3.14:
```powershell
pip install -r requirements.txt --pre
```

For Python 3.10 – 3.12:
```powershell
pip install flask==3.0.3 flask-cors==4.0.1 tensorflow==2.16.1 pillow==11.2.1 numpy werkzeug==3.0.3 matplotlib
```

### Step 4 — Generate Test Sample Images

```powershell
python generate_samples.py
```

### Step 5 — Launch the Application

```powershell
python app.py
```

The console will indicate that MobileNetV2 is loaded and the server is running:
```text
2026-10-24 20:25:00  INFO      Loading MobileNetV2 with ImageNet weights …
2026-10-24 20:25:05  INFO      Feature-extraction model ready (layer: Conv1).
2026-10-24 20:25:05  INFO      Database ready at database\predictions.db
2026-10-24 20:25:05  INFO      Starting CNN Vision Lab on http://127.0.0.1:5000
```

### Step 6 — Open in Browser
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## 🧪 Optional Custom CNN Training

To train a custom CNN on custom categorical data (such as binary Cats vs Dogs):

1. Organize your dataset under `training/data/`:
   ```
   training/data/
     train/
       class_a/
       class_b/
     val/
       class_a/
       class_b/
     test/
       class_a/
       class_b/
   ```
2. Run the training script:
   ```powershell
   python training/train_cnn.py
   ```
3. Training generates:
   - Model weights saved to `models/custom_cnn/`
   - Evaluation metrics in `models/eval_results.json`
   - Loss and accuracy curve records in `training/history.json` and `training/plots/`

---

## 🔌 API Reference

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/health` | Health check, status of model loading and database |
| `POST` | `/api/predict` | Upload image (`multipart/form-data`) → Top-3 classification results |
| `POST` | `/api/feature-maps` | Upload image → First convolution block feature activations |
| `GET` | `/api/model-info` | Metadata, parameters, architecture specs, and custom training metrics |
| `GET` | `/api/history` | Paginated list of recent prediction records |
| `DELETE` | `/api/history/<id>` | Remove specific prediction entry |
| `DELETE` | `/api/history?confirm=yes` | Wipe all history records |

---

## 🔧 Troubleshooting

- **`git: The term 'git' is not recognized`**:
  Install Git for Windows from [https://git-scm.com/download/win](https://git-scm.com/download/win) and restart your terminal.
- **Port 5000 in use**:
  Modify the port number in the last line of `app.py`: `app.run(..., port=5001)`.
- **First model loading delay**:
  On first boot, MobileNetV2 pretrained weights (~14 MB) are fetched from Google Cloud storage and cached under `~/.keras/models/`. Subsequent startups load instantly from the local cache.

---

## 👤 Author & Repository

- **Repository**: [https://github.com/jinalravalrj/CNN_Vision_Lab](https://github.com/jinalravalrj/CNN_Vision_Lab)
- **Author**: Jinal Raval
- **Project**: CNN Vision Lab — Deep Learning & Computer Vision Educational Workbench
