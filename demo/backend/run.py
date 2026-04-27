import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

subprocess.run([
    sys.executable, '-m', 'uvicorn',
    'routes:app',
    '--host', '0.0.0.0',
    '--port', '8000',
    '--reload',
])
