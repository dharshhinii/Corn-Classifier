import os
import subprocess
import sys

os.environ.setdefault("API_URL", "http://127.0.0.1:8000")
os.environ.setdefault("SHARED_IMAGE_DIR", "./test_images")

subprocess.run(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.address=0.0.0.0",
        "--server.port=8502",
    ],
    check=True,
)
