# app.py
# Prompt + Slides → (1) 4× Fill-in-the-Blank (prediction) + (2) Drag-into-Bins (true drag)
# -----------------------------------------------------------------------------------------
# What students see
# - Type a topic prompt. The app searches ./slides and generates:
#   (1) Four warm, prediction-style fill-in-the-blank items (lenient grading)
#   (2) TRUE drag-and-drop: draggable bank → titled bins (no extra steps for students)
#
# Notes
# - Works with PDFs (via PyPDF2 or pypdf) and .txt/.md/.html in ./slides
# - Avoids _repr_html_ crash by only emitting plain strings/components
# - Auto-installs streamlit-sortables at runtime (one-time) so drag just works
#
# Run:
#   streamlit run app.py
# -----------------------------------------------------------------------------------------

import os, re, pathlib, sys, subprocess
import streamlit as st

# ---------- Auto-enable TRUE drag without student CLI ----------
def _ensure_sortables():
    try:
        from streamlit_sortables import sort_multiple  # noqa
        return True
    except Exception:
        try:
            # silent install attempt (server-side, one time)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit-sortables"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            from streamlit_sortables import sort_multiple  # noqa
            return True
        except Exception:
            return False

DRAG_OK = _ensure_sortables()
if DRAG_OK:
    from streamlit_sortables import sort_multiple

# ---------- PDF backends ----------
PDF_BACKEND = None
try:
    import PyPDF2
    PDF_BACKEND = "PyPDF2"
except Exception:
    try:
        import pypdf
        PDF_BACKEND = "pypdf"
    except Exception:
        PDF_BACKEND = None

# ---------- App config ----------
st.set_page_config(page_title="Study Mode", page_icon="📘", layout="wide")
SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm"}

# ---------- IO ----------
def _read_text(path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""

def _read_pdf(path: str) -> str:
    if PDF_BACKEND == "PyPDF2":
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""
    if PDF_BACKEND == "pypdf":
        try:
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                return "\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""
    return ""

def load_corpus(slides_dir: str):
    corpus = {}
    if not os.path.isdir(slides_dir): return corpus
    base = pathlib.Path(slides_dir)
    for p in base.rglob("*"):
        if not p.is_file(): continue
        ext = p.suffix.lower()
        rel = str(p.relative_to(base))
        text = ""
        if ext in SUPPORTED_TEXT_EXTS:
            text = _read_text(str(p))
        elif ext == ".pdf":
            text = _read_pdf(str(p))
        if text and len(text.strip()) > 20:
            corpus[rel] = text
    return corpus

# ---------- Search ----------
def _tok(s: str):
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())

def _score(text: str, q_tokens):
    toks = _tok(text)
    if not toks: return 0
    freq = {}
    for t in toks: freq[t] = freq.get(t, 0) + 1
    return sum(freq.get(q, 0) for q in q_tokens)

# ---------- Topic detection from prompt ----------
def detect_topic(prompt: str):
    t = (prompt or "").lower()
    if any(k in t for k in ["transcription","rna pol","promoter","enhancer","tfiid","tbp","splice","mrna"]):
        return "transcription"
    if any(k in t for k in ["translation","ribosome","trna","elongation factor","eftu","ef-tu","efg","ef-g","release factor"]):
        return "translation"
    if any(k in t for k in ["replication","helicase","primase","ligase","okazaki","leading strand","lagging strand"]):
        return "replication"
    if any(k in t for k in ["microscope","resolution","diffraction","na","numerical aperture"]):
        return "microscopy"
    if any(k in t for k in ["rtk","gpcr","erk","mapk","pi3k","akt","ras","raf"]):
        return "signaling"
    if any(k in t for k in ["cell cycle","cdk","cyclin","checkpoint","apc/c","p53"]):
        return "cell_cycle"
    return "generic"

# ---------- FITB (prediction style, warm language) ----------
UP = {"increase","increases","increased","up","higher","activates","activation","faster","more","↑"}
DOWN = {"decrease","decreases","decreased","down","lower","inhibits","inhibition","slower","less","↓"}
NOCHANGE = {"no change","unchanged","same","neutral","nc","~"}
ACCUM = {"accumulate","accumulates","accumulated","builds up","pile u
