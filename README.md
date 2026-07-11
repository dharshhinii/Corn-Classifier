# Image Classification FastAPI MVP

## Put model files in `models/`

Update `config.json` so filenames and class names match your files.

The `class_names` order must be exactly the same order used during training.

## Install

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Run

```bash
python run.py --model baseline
python run.py --model mobilenet_finetuned
python run.py --model light
```

Use the default model from `config.json`:

```bash
python run.py
```

## Predict

```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"image_path":"/absolute/path/test.jpg"}'
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

The image path is local to the API server or container. It is not a path
from a remote caller's computer.

Keep `workers=1` for the MVP. Every Uvicorn worker loads another copy of
the model.
