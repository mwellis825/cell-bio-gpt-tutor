# app.py
# Prompt + Slides → (1) 4× Fill-in-the-Blank (prediction) + (2) Drag-into-Bins
# ---------------------------------------------------------------------------------
# How it works
# - User types a topic prompt (no defaults).
# - App searches ./slides (PDF / txt / md / html) for relevant content.
# - Extracts topical terms and builds:
#     (1) FOUR prediction-style fill-in-the-blank items (lenient grading; friendly language)
#     (2) TRUE drag-and-drop: draggable bank → titled bins (requires streamlit-sortables)
#
# Install
#   pip install streamlit
#   # for PDFs:
#   pip install PyPDF2    # or: pip install pypdf
#   # for true drag & drop:
#   pip install streamlit-sortables
#
# Run
#   streamlit run app.py
# ---------------------------------------------------------------------------------

import os, re, pathlib
import streamlit as st

# -------------------- Config --------------------
SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm"}

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

# Optional drag component (required for true drag & drop)
DRAG_OK = False
try:
    from streamlit_sortables import sort_multiple
    DRAG_OK = True
except Exception:
    DRAG_OK = False

# -------------------- File IO --------------------
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

# -------------------- Search helpers --------------------
def _tok(s: str):
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())

def _score(text: str, q_tokens):
    toks = _tok(text)
    if not toks: return 0
    freq = {}
    for t in toks: freq[t] = freq.get(t, 0) + 1
    return sum(freq.get(q, 0) for q in q_tokens)

# -------------------- Term mining (proteins/processes) --------------------
ENTITY_PATTERNS = [
    r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b",      # Cyclin, Ras, Myosin
    r"\b[AA-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*kinase\b",
    r"\bdna polymerase(?:\s*[ivx0-9]+)?\b|\brna polymerase(?:\s*[ivx0-9]+)?\b",
    r"\bhelicase\b|\bprimase\b|\bligase\b|\btopoisomerase\b|\bssb\b|\bgyrase\b",
    r"\bribosome\b|\btRNA\b|\brRNA\b|\bEF[- ]?Tu\b|\bEF[- ]?G\b|\brelease factor\b",
    r"\bcyclin ?[A-E]\b|\bcdk\d?\b|\bp53\b|\brb\b|\bapc/c\b|\bmcm\b",
    r"\breceptor tyrosine kinase\b|\brtk\b|\bgpcr\b|\bpi3k\b|\bakt\b|\bmapk\b|\bmek\b|\berk\b|\bras\b|\braf\b|\bmtor\b",
]
PROCESS_PATTERNS = [
    r"\binitiation\b|\belongation\b|\btermination\b|\bfidelity\b|\bproofreading\b",
    r"\bleading strand\b|\blagging strand\b|\bokazaki fragments\b|\borigin of replication\b|\breplication fork\b",
    r"\boxidative phosphorylation\b|\belectron transport\b|\btca cycle\b|\bglycolysis\b",
    r"\bspindle assembly checkpoint\b|\bg1/s\b|\bg2/m\b|\bmetaphase\b|\banaphase\b",
]

def mine_terms(text: str, max_terms=12):
    found = []
    for pat in ENTITY_PATTERNS + PROCESS_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            term = re.sub(r"\s+", " ", m.group(0).strip())
            if len(term) >= 3:
                found.append(term)
    seen = set(); clean = []
    for t in found:
        key = t.lower()
        if key not in seen:
            clean.append(t)
            seen.add(key)
    return clean[:max_terms]

# -------------------- FITB (prediction style) --------------------
UP = {"increase","increases","increased","up","higher","activates","activation","faster","more","↑"}
DOWN = {"decrease","decreases","decreased","down","lower","inhibits","inhibition","slower","less","↓"}
NOCHANGE = {"no change","unchanged","same","neutral","nc","~"}
ACCUM = {"accumulate","accumulates","accumulated","builds up","pile up"}

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+"," ", s)
    s = s.replace("’","'")
    return s

def match_pred(user_text: str, canonical: str, noun: str = "") -> bool:
    u = _norm(user_text)
    c = canonical
    if c == "increase":   return any(x in u for x in UP)
    if c == "decrease":   return any(x in u for x in DOWN)
    if c == "no_change":  return any(x in u for x in NOCHANGE)
    if c == "accumulate": return (any(x in u for x in ACCUM) or (noun and noun in u))
    return c in u

def fitb_from_terms(terms):
    """Return 4 friendly, prediction-style items: stem (string) + key ('increase'/'decrease'/...)"""
    tset = {t.lower() for t in terms}
    items = []

    if "helicase" in tset:
        items.append({
            "stem": "When helicase activity is inhibited, the rate of replication-fork progression would ______.",
            "key": "decrease", "noun": ""
        })
    if "primase" in tset:
        items.append({
            "stem": "If primase activity drops sharply, the number of new lagging-strand starts would ______.",
            "key": "decrease", "noun": ""
        })
    if "dna ligase" in tset or "ligase" in tset:
        items.append({
            "stem": "If DNA ligase is inactivated, unjoined Okazaki fragments would ______.",
            "key": "accumulate", "noun": "okazaki"
        })
    if {"rtk","erk","mapk","ras","raf"}.intersection(tset):
        items.append({
            "stem": "If an RTK is locked ON, the level of ERK phosphorylation would ______.",
            "key": "increase", "noun": ""
        })
    if not items:
        items = [
            {"stem": "When a rate-limiting enzyme is overexpressed, the amount of product formation would ______.", "key": "increase", "noun": ""},
            {"stem": "If a competitive inhibitor occupies the active site, the reaction rate would ______.", "key": "decrease", "noun": ""},
            {"stem": "If a feedback inhibitor is removed, the steady-state level of the controlled metabolite would ______.", "key": "increase", "noun": ""},
            {"stem": "If assembly chaperones are depleted, the efficiency of complex assembly would ______.", "key": "decrease", "noun": ""},
        ]
    return items[:4]

# -------------------- Drag into bins --------------------
def build_bins_and_bank(terms):
    """Return bins (labels), bank (draggable items), and answer map for scoring."""
    labels = ["Increase", "Decrease", "No change", "Accumulates"]
    tset = {t.lower() for t in terms}

    statements = []
    if "helicase" in tset:
        statements.append(("Replication-fork progression rate", "Decrease"))
    if "primase" in tset:
        statements.append(("Lagging-strand initiation events", "Decrease"))
    if "dna ligase" in tset or "ligase" in tset:
        statements.append(("Okazaki fragments", "Accumulates"))
    if {"rtk","erk","mapk","ras","raf"}.intersection(tset):
        statements.append(("ERK phosphorylation", "Increase"))

    if not statements:
        statements = [
            ("Product formation rate", "Increase"),
            ("Reaction rate with active-site inhibitor", "Decrease"),
            ("Controlled metabolite level without feedback", "Increase"),
            ("Off-pathway byproducts (immediate)", "No change"),
        ]

    bank = [s for (s, _) in statements]
    answer = {s:k for (s,k) in statements}
    return labels, bank, answer

# -------------------- UI --------------------
st.set_page_config(layout="wide")
st.title("📘 Prompt → Critical-Thinking Activities")

# No default topic
prompt = st.text_input("Enter a topic (e.g., translation fidelity, RTK→ERK, replication fork dynamics)", value="", placeholder="Type your topic…")

if st.button("Generate"):
    # Load slides once
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)

    corpus = st.session_state.corpus
    if not corpus:
        st.error("No slides found or PDFs not parsable. Add slide files to ./slides and/or install PyPDF2/pypdf.")
    else:
        q = _tok(prompt)
        ranked = sorted(corpus.items(), key=lambda kv: _score(kv[1], q), reverse=True)
        best_text = ranked[0][1] if ranked else ""
        terms = mine_terms((best_text + " " + prompt) if best_text else prompt, max_terms=12)

        st.session_state.fitb = fitb_from_terms(terms)
        st.session_state.labels, st.session_state.bank, st.session_state.answer = build_bins_and_bank(terms)
        # initialize drag state (multi-list order: [bank] + each bin list)
        st.session_state.drag_lists = [list(st.session_state.bank)] + [[] for _ in st.session_state.labels]
        st.success("Activities generated.")

# -------------------- Activity 1: 4× FITB (prediction) --------------------
if "fitb" in st.session_state:
    st.markdown("## Activity 1 — Predict the immediate effect")
    st.caption("Answer with simple words like **increase**, **decrease**, **no change**, or **accumulates**. I’ll grade leniently.")
    for i, item in enumerate(st.session_state.fitb, start=1):
        stem = item["stem"]
        key  = item["key"]
        noun = item.get("noun","")
        user = st.text_input(f"{i}. {stem}", key=f"fitb_{i}")
        if st.button(f"Check {i}", key=f"check_{i}"):
            if match_pred(user, key, noun):
                st.success("Correct.")
            else:
                st.error("Not quite — think about the **immediate** effect of that perturbation.")

# -------------------- Activity 2: TRUE Drag-and-Drop --------------------
if all(k in st.session_state for k in ["labels","bank","answer","drag_lists"]):
    st.markdown("---")
    st.markdown("## Activity 2 — Drag statements into the correct bin")
    st.caption("Bins: **Increase** • **Decrease** • **No change** • **Accumulates**")

    if DRAG_OK:
        labels = ["Bank"] + st.session_state.labels
        # keep list of lists stable across reruns
        lists = st.session_state.drag_lists
        updated = sort_multiple(lists, labels=labels, key="drag-multi")
        st.session_state.drag_lists = updated

        if st.button("Check bins"):
            # updated[0] is bank; bins start at index 1
            total = 0; correct = 0
            for bin_idx, bin_label in enumerate(st.session_state.labels, start=1):
                for item in updated[bin_idx]:
                    total += 1
                    want = st.session_state.answer.get(item, "No change")
                    got  = bin_label
                    if want == got:
                        correct += 1
            if total == 0:
                st.warning("Drag items from the bank into the bins first.")
            elif correct == total:
                st.success("All bins correct! 🎉")
            else:
                st.warning(f"{correct}/{total} correct — adjust and try again.")
    else:
        st.info("To enable **true drag-and-drop**, install the optional component:\n\n`pip install streamlit-sortables`\n\nThen restart the app.")
