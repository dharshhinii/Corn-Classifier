# Corn Subspecies Classifier Dashboard 🌽

A production-ready Web Application and REST API that classifies maize (corn) image samples into three subspecies: **Chulpi Cancha**, **Indurata**, and **Rugosa**. Built with a high-performance **FastAPI backend** and an ultra-modern, dark-mode **React (Vite) dashboard**.

---

## 📂 Project Architecture

```text
Three_image_classification/
├── app/                    # FastAPI backend codebase
│   ├── config.py           # Configuration loader & validation
│   ├── inference.py        # Image preprocessing & inference pipeline (Keras & TFLite)
│   ├── main.py             # Route definitions & static frontend serving
│   └── schemas.py          # Pydantic schema validation
├── frontend/               # React + Vite dashboard SPA
│   ├── src/                # React source code (App.jsx, style configs)
│   └── dist/               # Built static production bundle
├── models/                 # DL model files (Keras & Quantized TFLite formats)
│   ├── baseline_best.keras
│   ├── mobilenetv2_finetuned_best.keras
│   └── corn_classifier_dynamic_quant.tflite
├── notebooks/              # Jupyter Notebooks for training & analysis
│   └── corn_3class_image_classification_pipeline.ipynb
├── run.py                  # CLI backend server launcher script
├── run_ui.py               # Development frontend launcher script
├── config.json             # Global active model configuration
├── requirements.txt        # Backend dependencies
└── .gitignore              # Repository exclusion rules
```

---

## ✨ Features

- **Dynamic Model Swapping:** Toggle live between three different model architectures (Baseline CNN, MobileNetV2 Finetuned, and Quantized TFLite) directly from the UI without restarting the backend.
- **Inference Timing Analytics:** Displays fine-grained performance latency for preprocessing, model inference execution, and overall total duration in milliseconds.
- **Dual Inference Modes:** Predict via simple image upload (drag-and-drop) or by entering a server-side absolute file path.
- **Image Content Heuristics:** Employs HSV color space heuristics to detect non-plant/non-corn images (such as screenshots or documents) and warns the user if the input looks invalid.
- **Single-Origin Serve:** In production, the React frontend is served directly by the FastAPI backend on a single port for simple deployment.

---

## 🚀 Installation & Setup

### Prerequisites
* **Python:** Version 3.9 to 3.12
* **Node.js:** Version 18.0+

### Installation Steps

1. **Clone the repository** and navigate to the project root.
2. **Install Backend Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Frontend Dependencies:**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

---

## 💻 Running the Application

### 1. Local Development (Dual-Server Mode)
Run both backend and frontend servers in watch/development mode for live-reloading.

* **Start the Backend (Port 8000):**
  ```bash
  python run.py --reload
  ```
  *Swagger interactive docs will be available at:* `http://127.0.0.1:8000/docs`

* **Start the Frontend (Port 8502):**
  ```bash
  python run_ui.py
  ```
  *Access the development dashboard at:* `http://localhost:8502`

---

### 2. Production Deployment (Single-Server Mode)
FastAPI hosts the compiled React SPA directly from `frontend/dist/`. You only need to run one process in production!

1. **Build the production assets:**
   ```bash
   cd frontend
   npm run build
   cd ..
   ```
2. **Launch the unified production server:**
   ```bash
   python run.py
   ```
   *Access the full app dashboard and API at:* `http://localhost:8000`

*(Note: Keep `--workers 1` or `workers=1` for MVP execution as each worker loads a full copy of the TensorFlow models into RAM).*

---

## 🔌 API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Get API health status and currently loaded model details |
| `/models` | `GET` | List all available configured model IDs |
| `/model` | `GET` | Retrieve metadata of the currently active model (size, paths, inputs) |
| `/model/select` | `POST` | Hot-swap the active model (Query Param: `model_name`) |
| `/predict-image` | `POST` | Perform inference using a multipart file upload |
| `/predict` | `POST` | Perform inference using an absolute local image file path |

### Example Prediction Request `/predict-image`
**Request:**
Attach an image file in the `file` field of a `multipart/form-data` request.

```bash
curl -X POST http://127.0.0.1:8000/predict-image \
  -F "file=@/path/to/local/sample.jpg"
```

**Response:**
```json
{
  "model": "mobilenet_finetuned",
  "model_type": "keras",
  "image_path": "D:\\Three_image_classification\\temp_uploads\\sample.jpg",
  "predicted_class": "Zea_mays_Rugosa",
  "confidence": 0.999673,
  "probabilities": {
    "Zea_mays_Chulpi_Cancha": 0.000102,
    "Zea_mays_Indurata": 0.000225,
    "Zea_mays_Rugosa": 0.999673
  },
  "timing_ms": {
    "preprocessing": 4.15,
    "inference": 122.48,
    "total": 126.83
  },
  "is_corn_image": true
}
```

### Example Prediction Request `/predict`
**Request:**
Pass the absolute server-side image path in the JSON payload:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_path":"d:/Three_image_classification/sample.jpg"}'
```
