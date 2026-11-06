from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass


DATA_ROOT = Path(os.getenv("PATIENT_AGENT_SIMPLE_DATA", PROJECT_ROOT / "data"))
SKILLS_ROOT = Path(os.getenv("PATIENT_AGENT_SKILLS_ROOT", PROJECT_ROOT / "skills" / "departments"))
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_MODEL_URL")
DEFAULT_API_MODE = os.getenv("OPENAI_API_MODE", "auto").lower()
MAX_DEPARTMENT_SWITCHES = int(os.getenv("PATIENT_AGENT_MAX_SWITCHES", "2"))
MAX_SPECIALTY_TURNS = int(os.getenv("PATIENT_AGENT_MAX_SPECIALTY_TURNS", "4"))

