# app.py
# Study Mode (Slides from GitHub repo, PDF-friendly)
# -------------------------------------------------------------------
# Two activities only:
#   1) Fill-in-the-Blank (lenient; critical thinking)
#   2) Drag-the-Words (word bank -> blanks)
#
# Uses slides from ./slides (PDF, txt, md, html).
# Extracts Learning Objectives (LOs) from slide text.
#
# Run:
#   pip install streamlit PyPDF2
#   streamlit run app.py
# -------------------------------------------------------------------

import streamlit as st
import os, re, uuid, datetime, json, pathlib

st.set_page_config(page_title="Study Mode", page_icon="🗂️", layout="wide")

SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".html"}

# Try PyPDF2 (default), fallback to pypdf
PDF_BACKEND = None
try:
    import PyPDF2
    PDF_BACKEND = "PyPDF2"
except ImportError:
    try:
        import pypdf
        PDF_BACKEND = "pypdf"
    except ImportError:
        PDF_BACKEND = None

def read_text_file(path: str) -> str:
    try:
        return open(path, "r", encoding="utf-8").read()
    except:
        return ""

def read_pdf_file(path: str) -> str:
    if PDF_BACKEND == "PyPDF2":
        try:
            text = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        except:
            return ""
    if PDF_BACKEND == "pypdf":
        try:
            text = []
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        except:
            return ""
    return ""

def load_slides(slides_dir: str):
    corpus = {}
    if not os.path.isdir(slides_dir): return corpus
    base = pathlib.Path(slides_dir)
    for p in base.rglob("*"):
        if not p.is_file(): continue
        ext = p.suffix.lower()
        rel = str(p.relative_to(base))
        if ext in SUPPORTED_TEXT_EXTS:
            text = read_text_file(str(p))
        elif ext == ".pdf":
            text = read_pdf_file(str(p))
        else:
            text = ""
        if text.strip():
            corpus[rel] = text
    return corpus

# --- Learning Objective extraction ---
VERBS = ["explain","predict","justify","compare","contrast","design","interpret","evaluate","propose","identify","analyze","apply"]
def extract_los(text: str):
    lines = [ln.strip() for ln in text.splitlines()]
    los = []
    for ln in lines:
        low = ln.lower()
        bullet = low.startswith(("-","•","*")) or re.match(r"^\d", low)
        starts_verb = any(low.startswith(v) for v in VERBS)
        if bullet or starts_verb or "objective" in low:
            cleaned = re.sub(r"^[-•*\d\).\s]+","",ln)
            if len(cleaned) > 8: los.append(cleaned)
    return list(dict.fromkeys(los))  # dedupe

def detect_topic(text: str):
    t=text.lower()
    if "replication" in t: return "dna_replication"
    if "glycolysis" in t or "mitochond" in t: return "metabolism"
    if "actin" in t or "microtubule" in t: return "cytoskeleton"
    if "cell cycle" in t or "cdk" in t: return "cell_cycle"
    if "rtk" in t or "gpcr" in t: return "signaling"
    return "generic"

DNA_KEY_TERMS=["helicase","primase","leading strand","lagging strand","Okazaki fragments","DNA ligase"]

def make_fitb(lo, topic):
    answer = next((t for t in DNA_KEY_TERMS if t in lo.lower()), "mechanism")
    return {"prompt": f"{lo} — What is the key factor/mechanism?", "answer": answer, "accepted":[answer,answer.lower()]}

def make_dragwords(topic):
    if topic=="dna_replication":
        return {
            "sentence":"At the fork, ______ unwinds DNA while ______ makes primers. Continuous synthesis is on the ______, while discontinuous synthesis makes ______ sealed by ______.",
            "answers":["helicase","primase","leading strand","Okazaki fragments","DNA ligase"],
            "bank":["helicase","primase","leading strand","Okazaki fragments","DNA ligase","topoisomerase","polymerase"]
        }
    else:
        return {
            "sentence":"In this pathway, ______ starts the process, ______ is rate-limiting, feedback via ______ stabilizes, recovery depends on ______.",
            "answers":["initiator","rate-limiting","feedback","recovery"],
            "bank":["initiator","rate-limiting","feedback","recovery","mechanism","inhibitor"]
        }

def normalize(s): return re.sub(r"\s+"," ",(s or "").strip().lower())

def check_fitb(ans,accepted): return normalize(ans) in [normalize(a) for a in accepted]

# ---------------- UI -----------------
st.title("🗂️ Study Mode")
st.write("Slides loaded from `/slides`")

corpus=load_slides(SLIDES_DIR)
if not corpus:
    st.error("No slides found or unable to read PDFs. Install `PyPDF2` or `pypdf`.")
else:
    st.success(f"Loaded {len(corpus)} slide file(s).")

files=list(corpus.keys())
choice=st.selectbox("Choose slide file", files) if files else None

if choice:
    text=corpus[choice]
    topic=detect_topic(text)
    los=extract_los(text)[:4]
    st.info(f"Topic: {topic} | {len(los)} LOs found")

    if st.button("Generate Activities"):
        fitbs=[make_fitb(lo,topic) for lo in los[:2]]
        drag=make_dragwords(topic)
        st.session_state["gen"]={"fitb":fitbs,"drag":drag}

data=st.session_state.get("gen")
if data:
    st.header("Fill in the Blank")
    for i,it in enumerate(data["fitb"],1):
        st.subheader(f"Q{i}")
        st.write(it["prompt"])
        ans=st.text_input("Your answer", key=f"fitb_{i}")
        if ans and st.button(f"Check {i}"):
            st.write("✅ Correct" if check_fitb(ans,it["accepted"]) else "❌ Try again")

    st.header("Drag the Words")
    d=data["drag"]
    st.write(d["sentence"])
    st.write("Word bank:", ", ".join(d["bank"]))
    st.write("Answer key:", ", ".join(d["answers"]))
