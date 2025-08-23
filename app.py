# app.py
# Study Mode (Slides Auto): Two activities + auto-LOs from ./slides
# -----------------------------------------------------------------------------
# Features:
# - Scans ./slides for text-based slide exports (.txt, .md, .html). Optional .pdf if PyPDF2 available.
# - Extracts Learning Objectives (LOs) with simple heuristics; no student entry.
# - Generates two activities: (1) Fill-in-the-Blank (lenient), (2) Drag-the-Words.
# - Instructor mode toggles answer keys and enables JSON export.
# - Fully offline; no external calls.
#
# Run:
#   pip install streamlit
#   streamlit run app.py
# -----------------------------------------------------------------------------

import streamlit as st
import os, re, uuid, datetime, json, pathlib

st.set_page_config(page_title="Study Mode (Slides Auto)", page_icon="🗂️", layout="wide")

# -------------------------- Config -------------------------------------------
SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm"}
PDF_ENABLED = False
try:
    import PyPDF2  # optional
    PDF_ENABLED = True
except Exception:
    PDF_ENABLED = False

# -------------------------- Utilities ----------------------------------------
def new_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            return ""

def read_pdf_file(path: str) -> str:
    if not PDF_ENABLED:
        return ""
    try:
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    text.append(page.extract_text() or "")
                except Exception:
                    pass
        return "\n".join(text)
    except Exception:
        return ""

def load_slides_corpus(slides_dir: str):
    corpus = {}  # {relative_path: text}
    if not os.path.isdir(slides_dir):
        return corpus
    base = pathlib.Path(slides_dir)
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        rel = str(p.relative_to(base))
        text = ""
        if ext in SUPPORTED_TEXT_EXTS:
            text = read_text_file(str(p))
        elif ext == ".pdf":
            text = rea
