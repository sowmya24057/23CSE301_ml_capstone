"""
Root entrypoint for Streamlit Community Cloud and local CLI.
Runs the main ML Studio application located in app/app.py.
"""
import os
import sys
from pathlib import Path

# Execute app/app.py directly in the current scope
app_path = Path(__file__).resolve().parent / "app" / "app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, {"__name__": "__main__", "__file__": str(app_path)})
