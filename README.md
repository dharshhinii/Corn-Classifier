# Image Classification FastAPI MVP with React UI

A three-class deep learning image classification service for corn seed/kernel varieties (*Zea mays Chulpi Cancha*, *Zea mays Indurata*, *Zea mays Rugosa*). Includes a FastAPI backend with model switching, edge TFLite quantization, and an ultra-modern React frontend.

---

## 📁 Model Files in `models/`

All model files are organized inside the `models/` directory:

| Model Name | Model Type | Path | Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `baseline` | Keras | `models/mobilenetv2_head_best.keras` | ~11.6 MB | Head-trained MobileNetV2 baseline classifier |
| `mobilenet_finetuned` | Keras | `models/mobilenetv2_finetuned_best.keras` | ~23.7 MB | Full fine-tuned MobileNetV2 architecture (Default) |
| `light` | TFLite | `models/corn_classifier_dynamic_quant.tflite` | ~2.67 MB | Dynamic int8 quantized TFLite edge model |

The `class_names` in `config.json` match the exact training label order:
```json
["Zea_mays_Chulpi_Cancha", "Zea_mays_Indurata", "Zea_mays_Rugosa"]
```

---

## 🚀 Installation & Setup

### 1. Python Environment Setup

```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## ⚡ Running the Application

### Run with Specific Models:

- **Baseline model:**
  ```bash
  python run.py --model baseline
  ```

- **MobileNetV2 Fine-Tuned model:**
  ```bash
  python run.py --model mobilenet_finetuned
  ```

- **Light TFLite model:**
  ```bash
  python run.py --model light
  ```

- **Default Model (from `config.json`):**
  ```bash
  python run.py
  ```

---

## 🖥️ React UI & Swagger OpenAPI Documentation

- **React Web Application:**  
  Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. Features live model switching, drag-and-drop image upload, sample gallery, probability breakdown charts, and timing latency metrics.

- **Interactive Swagger Documentation:**  
  Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

*(Note: Keep `workers=1` for MVP execution as each worker loads a full copy of the neural model into memory).*

---

## 📡 API Prediction Endpoint Examples

### 1. Predict via Local Server Image Path (`/predict`)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_path":"c:/project_files/Sessions/test_images/chulpi_cancha 3.jpg"}'
```

**Example JSON Response:**
```json
{
  "model": "mobilenet_finetuned",
  "model_type": "keras",
  "image_path": "C:\\project_files\\Sessions\\test_images\\chulpi_cancha 3.jpg",
  "predicted_class": "Zea_mays_Chulpi_Cancha",
  "confidence": 0.999993,
  "probabilities": {
    "Zea_mays_Chulpi_Cancha": 0.999993,
    "Zea_mays_Indurata": 0.0,
    "Zea_mays_Rugosa": 0.000007
  },
  "timing_ms": {
    "preprocessing": 4.12,
    "inference": 48.51,
    "total": 52.63
  }
}
```

### 2. Predict via Direct Image File Upload (`/predict/upload`)

```bash
curl -X POST http://127.0.0.1:8000/predict/upload \
  -F "file=@/path/to/local/sample.jpg"
```

---

## 🛠️ Frontend Development (Optional Vite Dev Server)

To run the React frontend in hot-reloading development mode:

```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.
