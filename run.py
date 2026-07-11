import argparse
import os

import uvicorn


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["baseline", "mobilenet_finetuned", "light"],
)
parser.add_argument("--config", default="config.json")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--reload", action="store_true")
args = parser.parse_args()

os.environ["MODEL_CONFIG"] = args.config
if args.model:
    os.environ["MODEL_NAME"] = args.model

uvicorn.run(
    "app.main:app",
    host=args.host,
    port=args.port,
    workers=args.workers,
    reload=args.reload,
)
