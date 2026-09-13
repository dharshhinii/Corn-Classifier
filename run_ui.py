import os
import subprocess
import sys

# Navigate to the frontend directory relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(script_dir, "frontend")

if not os.path.exists(frontend_dir):
    print(f"Error: Frontend directory {frontend_dir} does not exist.")
    sys.exit(1)

os.chdir(frontend_dir)

print("Starting React (Vite) development server on http://localhost:8502...")
try:
    subprocess.run(
        "npm run dev -- --port 8502 --host 0.0.0.0",
        shell=True,
        check=True,
    )
except KeyboardInterrupt:
    print("\nStopping React development server.")

