# app.py
# Classic Study Mode — Prompt-based, slides-aware (minimal UI)
# -----------------------------------------------------------------------------
# 1) User enters a topic prompt.
# 2) App searches ./slides (PDF/txt/md/html) for relevant content.
# 3) App extracts learning-objective-like lines to drive question generation.
# 4) App outputs exactly two activities in this order:
#       (1) Fill-in-the-Blank (lenient)
#       (2) Drag-and-Drop (click-to-fill word bank)
#
# Run:
#   pip install streamlit
#   # For PDFs, install one of:
#   pip install PyPDF2
#   # or
#   pip install pypdf
#   streamlit run app.py
# -----------------------------------------------------------------------------

import streamlit as st
import os, re, uuid, pathlib

st.set_page_config(page_title="Study Mode (Prompt + Slides)", page_icon="📘", layout="wide")

# -------------------- Config --------------------
SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm"}

# Try lightweight PDF backends (optional)
PDF_BACKEND = None
try:
    import PyPDF2  # common
    PDF_BACKEND = "PyPDF2"
except Exception:
    try:
        import pypdf  # alternative
        PDF_BACKEND = "pypdf"
    except Exception:
        PDF_BACKEND = None

# -------------------- IO Helpers ----------------
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
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        except Exception:
            return ""
    if PDF_BACKEND == "pypdf":
        try:
            text = []
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        except Exception:
            return ""
    return ""

def load_corpus(slides_dir: str):
    corpus = {}  # {rel_path: text}
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
            corpus[rel] = text
    return corpus

# -------------------- Search & LO extraction ----------------
def tokenize(s: str):
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())

def simple_match_score(text: str, query_tokens):
    tks = tokenize(text)
    if not tks: return 0
    score = 0
    bag = {}
    for tk in tks:
        bag[tk] = bag.get(tk, 0) + 1
    for q in query_tokens:
        score += bag.get(q, 0)
    return score

VERBS = [
    "explain","predict","justify","compare","contrast","distinguish","design","interpret",
    "evaluate","diagnose","infer","propose","calculate","model","identify",
    "classify","analyze","synthesize","outline","formulate","apply","assess"
]
def extract_los(text: str):
    # Prefer LO-like bullets or verb-led lines; fallback to long sentences
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    los = []
    for ln in lines:
        low = ln.lower()
        if not ln or len(ln) < 6: 
            continue
        bullet = low.startswith(("-","•","—","*")) or re.match(r"^\d+[\).\]]\s", low)
        starts_verb = any(re.match(rf"^\s*({v})\b", low) for v in VERBS)
        mentions_lo = any(k in low for k in ["learning objective", "objective:", "outcome:"])
        if bullet or starts_verb or mentions_lo:
            cleaned = re.sub(r"^(\d+[\).\]]\s*|[-•—*]\s*)", "", ln).strip()
            if len(cleaned) >= 10:
                los.append(cleaned)
    if not los:
        # Fallback: long-ish sentences as pseudo-LOs
        chunks = re.split(r"(?<=[\.\?\!])\s+", text)
        los = [c.strip() for c in chunks if len(c.strip()) > 50][:6]
    # dedupe
    seen = set(); uniq = []
    for x in los:
        k = x.lower()
        if k not in seen:
            uniq.append(x); seen.add(k)
    return uniq[:8]

# -------------------- Generation (exact two activities) ----------------
DNA_TERMS = ["helicase","primase","leading strand","lagging strand","okazaki fragments","dna ligase","topoisomerase","ssb","rnase h","dna polymerase iii"]

def detect_topic_from_text(text: str):
    t = text.lower()
    if any(k in t for k in ["replication","okazaki","helicase","primase","leading strand","lagging strand","ligase"]):
        return "dna_replication"
    if any(k in t for k in ["glycolysis","tca","oxidative phosphorylation","electron transport","mitochond"]):
        return "metabolism"
    if any(k in t for k in ["cell cycle","cdk","cyclin","p53","meiosis","mitosis"]):
        return "cell_cycle"
    if any(k in t for k in ["actin","microtubule","kinesin","dynein","focal adhesion"]):
        return "cytoskeleton"
    if any(k in t for k in ["rtk","gpcr","pi3k","akt","mapk"]):
        return "signaling"
    return "generic"

def make_fitb(lo: str, topic: str):
    # Lean, specific, lenient answer check; prefer DNA terms if detected.
    ans = "mechanism"
    if topic == "dna_replication":
        low = lo.lower()
        for term in DNA_TERMS:
            if term in low:
                ans = term
                break
        if ans == "mechanism":
            ans = "dna ligase"
    prompt = f"{lo} — What is the **single most specific factor/mechanism** involved?"
    return {"prompt": prompt, "answer": ans, "accepted": [ans, ans.lower(), ans.title()]}

def make_dragwords(topic: str):
    if topic == "dna_replication":
        sentence = ("At the replication fork, ______ unwinds DNA while ______ lays down primers. "
                    "Continuous synthesis is on the ______, whereas discontinuous synthesis forms ______ "
                    "that are sealed by ______.")
        answers = ["helicase","primase","leading strand","okazaki fragments","dna ligase"]
        bank = list(set(answers + ["topoisomerase","dna polymerase iii","lagging strand","rnase h","proofreading"]))
    else:
        sentence = ("In this pathway, ______ initiates the process, ______ is rate-limiting, "
                    "feedback via ______ stabilizes output, and recovery depends on ______.")
        answers = ["initiator","rate-limiting","feedback","recovery"]
        bank = list(set(answers + ["mechanism","inhibitor","activator"]))
    return {"sentence": sentence, "answers": answers, "bank": bank}

def normalize(s): return re.sub(r"\s+"," ", (s or "").strip().lower())
def check_fitb(user, accepted): return normalize(user) in [normalize(a) for a in accepted]

# -------------------- UI (minimal; exact order) ----------------
st.title("📘 Study Mode — Prompt + Slides")
st.caption("Enter a topic prompt. The app searches your existing slides and generates two activities in order: Fill-in-the-Blank, then Drag-and-Drop.")

prompt = st.text_input("Enter a topic prompt (e.g., 'DNA Replication', 'RTK signaling', 'Cell cycle checkpoints')", value="DNA Replication")
go = st.button("Generate")

# Load once per session
if "corpus" not in st.session_state:
    st.session_state.corpus = load_corpus(SLIDES_DIR)

if go:
    corpus = st.session_state.corpus
    if not corpus:
        st.error("No slides found or unable to read from `./slides`. Add slides or install a PDF parser (PyPDF2/pypdf).")
    else:
        q_toks = tokenize(prompt)
        ranked = sorted(corpus.items(), key=lambda kv: simple_match_score(kv[1], q_toks), reverse=True)
        best_text = ranked[0][1] if ranked else ""
        topic = detect_topic_from_text(best_text or prompt)
        los = extract_los(best_text or prompt)
        # choose first LO for FITB; if missing, create a generic one
        fitb_source = los[0] if los else f"Explain the main constraint in {prompt}."
        fitb_item = make_fitb(fitb_source, topic)
        drag_item = make_dragwords(topic)
        st.session_state["activities"] = {"fitb": fitb_item, "drag": drag_item}

data = st.session_state.get("activities")
if data:
    # 1) Fill-in-the-Blank (first)
    st.markdown("## Activity 1: Fill-in-the-Blank")
    st.write(data["fitb"]["prompt"])
    user_ans = st.text_input("Your answer:", key="fitb_ans")
    if st.button("Check answer"):
        st.success("✅ Correct!") if check_fitb(user_ans, data["fitb"]["accepted"]) else st.error("❌ Not quite. Aim for the most specific term.")

    # 2) Drag-and-Drop (second)
    st.markdown("---")
    st.markdown("## Activity 2: Drag-and-Drop")
    d = data["drag"]
    # Keep state for blanks/bank
    if "dw_bank" not in st.session_state:
        st.session_state.dw_bank = list(d["bank"])
    if "dw_blanks" not in st.session_state:
        st.session_state.dw_blanks = [None]*len(d["answers"])
    bank = st.session_state.dw_bank
    blanks = st.session_state.dw_blanks

    parts = d["sentence"].split("______")
    disp = []
    for idx in range(len(parts)+len(blanks)):
        if idx % 2 == 0:
            disp.append(parts[idx//2])
        else:
            bi = (idx-1)//2
            label = blanks[bi] if blanks[bi] else "______"
            if st.button(label, key=f"blank_{bi}"):
                if blanks[bi]:
                    bank.append(blanks[bi])
                    blanks[bi] = None
                st.session_state.dw_blanks = blanks
    st.write("".join(disp))

    st.write("**Word bank:**")
    cols = st.columns(4)
    for idx, w in enumerate(list(bank)):
        col = cols[idx % 4]
        if col.button(w, key=f"bank_{idx}"):
            for bi in range(len(blanks)):
                if blanks[bi] is None:
                    blanks[bi] = w
                    break
            bank.remove(w)
            st.session_state.dw_bank = bank
            st.session_state.dw_blanks = blanks

    if st.button("Check sentence"):
        correct = all((blanks[i] or "").lower() == d["answers"][i].lower() for i in range(len(d["answers"])))
        if correct:
            st.success("✅ All blanks correct!")
        else:
            good = sum(1 for i in range(len(d["answers"])) if (blanks[i] or '').lower() == d['answers'][i].lower())
            st.warning(f"{good}/{len(d['answers'])} correct. Keep refining.")
        st.caption("Answer key: " + ", ".join(d["answers"]))
