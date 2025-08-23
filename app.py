# app.py
# Study Mode (Slides from GitHub repo, PDF-friendly)
# -------------------------------------------------------------------
# Two activities only:
#   1) Fill-in-the-Blank (lenient; critical thinking)
#   2) Drag-the-Words (word bank -> blanks)
#
# Automatically uses slides from ./slides (PDF, txt, md, html).
# Extracts Learning Objectives (LOs) from slide text.
#
# Run:
#   pip install streamlit PyPDF2   # or pip install pypdf
#   streamlit run app.py
# -------------------------------------------------------------------

import streamlit as st
import os, re, uuid, datetime, json, pathlib

st.set_page_config(page_title="Study Mode (Repo Slides)", page_icon="🗂️", layout="wide")

# -------------------------- Config -------------------------------------------
SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm"}

PDF_BACKEND = None
PDF_INFO = ""
try:
    import PyPDF2
    PDF_BACKEND = "PyPDF2"
    PDF_INFO = "PDF extraction via PyPDF2"
except Exception:
    try:
        import pypdf
        PDF_BACKEND = "pypdf"
        PDF_INFO = "PDF extraction via pypdf"
    except Exception:
        PDF_BACKEND = None
        PDF_INFO = "No PDF extractor installed (install PyPDF2 or pypdf)"

# -------------------------- Utilities ----------------------------------------
def new_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_text_file(path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""

def read_pdf_file(path: str) -> str:
    if PDF_BACKEND == "PyPDF2":
        try:
            text = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    try:
                        text.append(page.extract_text() or "")
                    except Exception:
                        text.append("")
            return "\n".join(text)
        except Exception:
            return ""
    if PDF_BACKEND == "pypdf":
        try:
            text = []
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    try:
                        text.append(page.extract_text() or "")
                    except Exception:
                        text.append("")
            return "\n".join(text)
        except Exception:
            return ""
    return ""

def load_slides_corpus(slides_dir: str):
    corpus = {}
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
            text = read_pdf_file(str(p))
        if text and len(text.strip()) > 20:
            corpus[rel] =
