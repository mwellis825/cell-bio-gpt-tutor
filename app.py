# app.py
# Prompt + Slides → (1) 4× Fill-in-the-Blank (prediction) + (2) Drag-into-Bins
# ---------------------------------------------------------------------------------
# What it does
# - User types a topic prompt (no defaults).
# - App searches ./slides (PDF/txt/md/html) for relevant content.
# - Extracts candidate entities (proteins/enzymes/factors/process words).
# - Builds 4 SHORT prediction prompts (perturbation → predicted change), graded leniently.
# - Shows Drag-and-Drop with labeled bins. If 'streamlit-sortables' is installed, uses real
#   drag across multiple lists. Otherwise, falls back to a click-to-assign UI.
#
# Install
#   pip install streamlit
#   # For PDFs:
#   pip install PyPDF2    # or: pip install pypdf
#   # Optional for true drag:
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

# Optional drag component
DRAG_AVAILABLE = False
try:
    from streamlit_sortables import sort_items, sort_multiple
    DRAG_AVAILABLE = True
except Exception:
    DRAG_AVAILABLE = False

# -------------------- File IO --------------------
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
            text = read_text_file(str(p))
        elif ext == ".pdf":
            text = read_pdf_file(str(p))
        if text and len(text.strip()) > 20:
            corpus[rel] = text
    return corpus

# -------------------- Search helpers --------------------
def tokenize(s: str):
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())

def score(text: str, q_tokens):
    toks = tokenize(text)
    if not toks: return 0
    freq = {}
    for t in toks: freq[t] = freq.get(t, 0) + 1
    return sum(freq.get(q, 0) for q in q_tokens)

# -------------------- Entity / process mining --------------------
ENTITY_HINTS = [
    r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?",             # Proper-like (Cyclin, Ras, Myosin)
    r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*kinase",  # *kinase
    r"[A-Za-z0-9]+ polymerase(?:\s*[IVX0-9]+)?", # DNA polymerase III, RNA polymerase II
    r"helicase|primase|ligase|topoisomerase|ssb|gyrase|clamp loader|cohesin|separase",
    r"ribosome|ribosomal subunit|tRNA|rRNA|EF[- ]?Tu|EF[- ]?G|release factor",
    r"cyclin ?[A-E]|cdk\d?|p53|rb|apc/c|mcm|origin recognition complex",
    r"receptor tyrosine kinase|rtk|gpcr|pi3k|akt|mapk|mek|erk|ras|raf|mTOR",
]

PROCESS_HINTS = [
    r"initiation|elongation|termination|fidelity|proofreading",
    r"leading strand|lagging strand|okazaki fragments|origin of replication|replication fork",
    r"oxidative phosphorylation|electron transport|tca cycle|glycolysis",
    r"spindle assembly checkpoint|g1/s|g2/m|metaphase/anaphase",
]

def mine_terms(text: str, max_terms=10):
    found = []
    for pat in ENTITY_HINTS + PROCESS_HINTS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            term = m.group(0).strip()
            if len(term) < 3: continue
            found.append(term)
    # de-dupe and normalize whitespace/case style
    clean = []
    seen = set()
    for t in found:
        tt = re.sub(r"\s+", " ", t).strip()
        key = tt.lower()
        if key not in seen:
            clean.append(tt)
            seen.add(key)
    return clean[:max_terms]

# -------------------- Prompt builders (prediction, not trivia) --------------------
UP = {"↑","increase","increases","increased","up","higher","activates","activation","faster","more"}
DOWN = {"↓","decrease","decreases","decreased","down","lower","inhibits","inhibition","slower","less"}
NOCHANGE = {"nc","no change","unchanged","same","neutral","~"}
ACCUM = {"accumulate","accumulates","accumulated","builds up","pile up","↑ [x]","more [x]"}

def normalize(s: str):
    s = (s or "").strip().lower()
    s = re.sub(r"\s+"," ", s)
    s = s.replace("’","'")
    return s

def matches_pred(user_text: str, canonical: str):
    u = normalize(user_text)
    if canonical == "increase":
        return any(tok in u for tok in UP)
    if canonical == "decrease":
        return any(tok in u for tok in DOWN)
    if canonical == "no_change":
        return any(tok in u for tok in NOCHANGE)
    if canonical.startswith("accumulate"):
        # accept "accumulate", "builds up", or name mention
        base = canonical.replace("accumulate:", "").strip()
        return (any(tok in u for tok in ACCUM) and (base=="" or base in u)) or (base and base in u)
    # fallback: substring
    return canonical in u

def build_fitb_predictions(terms):
    """
    Build 4 perturbation→prediction items from discovered terms/processes.
    Each item = {stem, answer_key('increase'/'decrease'/'no_change'/'accumulate:XYZ')}
    """
    items = []
    # Heuristic pairings for common topics
    tset = {t.lower() for t in terms}
    # 1 helicase/primase/ligase/EF-Tu/EF-G etc.
    if any(k in tset for k in ["helicase","primase","dna ligase","ligase","ef-tu","ef tu","ef-g","ef g","release factor"]):
        if "helicase" in tset:
            items.append({
                "stem": "You add a potent helicase inhibitor. Predict replication fork progression rate.",
                "key": "decrease"
            })
        if "primase" in tset:
            items.append({
                "stem": "Primase activity drops 80% due to nucleotide scarcity. Predict lagging-strand initiation frequency.",
                "key": "decrease"
            })
        if "dna ligase" in tset or "ligase" in tset:
            items.append({
                "stem": "DNA ligase is temperature-sensitive and becomes inactive. Predict the immediate fate of Okazaki fragments.",
                "key": "accumulate:okazaki fragments"
            })
        if "release factor" in tset:
            items.append({
                "stem": "Release factor is depleted. Predict effect on termination-time per protein.",
                "key": "increase"
            })
    # 2 translation factors
    if any(k in tset for k in ["ribosome","ef-tu","ef g","ef-g","trna"]):
        items.append({
            "stem": "EF-Tu binding to aminoacyl-tRNA is blocked. Predict elongation rate.",
            "key": "decrease"
        })
    # 3 signaling generic
    if any(k in tset for k in ["rtk","pi3k","akt","mapk","erk","ras","raf","mtor"]):
        items.append({
            "stem": "A small-molecule locks the RTK in its active dimer state (ligand-independent). Predict downstream ERK phosphorylation.",
            "key": "increase"
        })
    # 4 cell-cycle
    if any(k in tset for k in ["cyclin", "cdk", "p53", "apc/c", "apc c"]):
        items.append({
            "stem": "APC/C activity is inhibited by a spindle checkpoint signal. Predict separase activation at metaphase.",
            "key": "decrease"
        })

    # Backfill if we have <4
    generic_pool = [
        {"stem": "You overexpress a rate-limiting enzyme in this pathway. Predict pathway flux.", "key": "increase"},
        {"stem": "A competitive inhibitor binds the active site of the key catalyst. Predict product formation rate.", "key": "decrease"},
        {"stem": "A feedback inhibitor is deleted. Predict steady-state level of the regulated metabolite.", "key": "increase"},
        {"stem": "Chaperone assisting complex assembly is removed. Predict assembly efficiency.", "key": "decrease"},
        {"stem": "Signal is present but receptor is nonfunctional. Predict downstream target activation.", "key": "decrease"},
    ]
    i = 0
    while len(items) < 4 and i < len(generic_pool):
        items.append(generic_pool[i]); i += 1
    return items[:4]

def build_drag_bins(terms):
    """
    Build labeled bins plus a bank of statements to classify.
    Prefer bins: Increase / Decrease / No change / Accumulates.
    """
    bins = [("Increase", "increase"), ("Decrease", "decrease"), ("No change", "no_change"), ("Accumulates", "accumulate")]
    # make 6–8 statements influenced by terms
    statements = []
    tset = {t.lower() for t in terms}
    if "helicase" in tset:
        statements.append(("Fork progression rate", "decrease"))
    if "primase" in tset:
        statements.append(("Lagging-strand initiation events", "decrease"))
    if "dna ligase" in tset or "ligase" in tset:
        statements.append(("Okazaki fragments", "accumulate"))
    if "ribosome" in tset or "ef-tu" in tset or "ef g" in tset or "ef-g" in tset:
        statements.append(("Elongation speed", "decrease"))
        statements.append(("Ribosome stalling frequency", "increase"))
    if any(k in tset for k in ["rtk","pi3k","akt","mapk","erk","ras","raf"]):
        statements.append(("ERK phosphorylation", "increase"))
        statements.append(("Pro-survival signaling", "increase"))
    if not statements:
        statements = [
            ("Pathway flux", "increase"),
            ("Product formation rate", "decrease"),
            ("Feedback-regulated metabolite", "increase"),
            ("Assembly efficiency", "decrease"),
            ("Off-pathway byproducts", "no_change"),
        ]
    # bank is just the left column words
    bank = [s for (s, _) in statements]
    answer_map = {s:k for (s,k) in statements}
    return bins, bank, answer_map

# -------------------- UI --------------------
st.set_page_config(layout="wide")
st.title("📘 Prompt → Critical-Thinking Activities")

prompt = st.text_input("Enter a topic prompt (e.g., 'translation fidelity', 'RTK to ERK', 'replication fork dynamics')", value="", placeholder="Type your topic…")
if st.button("Generate"):
    # load slides once
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)

    corpus = st.session_state.corpus
    if not corpus:
        st.error("No slides found or PDFs unreadable. Add slides to ./slides and/or install PyPDF2 or pypdf.")
    else:
        q = tokenize(prompt)
        ranked = sorted(corpus.items(), key=lambda kv: score(kv[1], q), reverse=True)
        best_text = ranked[0][1] if ranked else ""
        # mine terms from the best match + prompt
        terms = mine_terms((best_text + " " + prompt) if best_text else prompt, max_terms=12)
        if not terms:
            st.warning("Didn’t detect many topic terms; generating generic prediction activities.")
        # Build activities
        fitb_items = build_fitb_predictions(terms)
        bins, bank, ansmap = build_drag_bins(terms)

        # store in session
        st.session_state["fitb_items"] = fitb_items
        st.session_state["bins"] = bins
        st.session_state["bank"] = bank
        st.session_state["ansmap"] = ansmap
        st.session_state["drag_state"] = {"Increase": [], "Decrease": [], "No change": [], "Accumulates": []}

# ----- Render Activity 1: 4× Fill-in-the-Blank (prediction) -----
if "fitb_items" in st.session_state:
    st.markdown("## Activity 1 — Predict the immediate effect")
    st.caption("Answer with terms like: increase / decrease / no change / accumulate X. Lenient matching accepted.")
    for i, item in enumerate(st.session_state["fitb_items"], start=1):
        stem = item["stem"]
        key = item["key"]           # canonical
        u = st.text_input(f"{i}. {stem}", key=f"fitb_{i}")
        if st.button(f"Check {i}", key=f"check_{i}"):
            ok = matches_pred(u, key)
            st.success("Correct.") if ok else st.error("Not quite — think causally about immediate effects.")

# ----- Render Activity 2: Drag into labeled bins -----
if all(k in st.session_state for k in ["bins","bank","ansmap","drag_state"]):
    st.markdown("---")
    st.markdown("## Activity 2 — Drag statements into the correct bin")
    st.caption("Bins: Increase / Decrease / No change / Accumulates")

    labels = [b[0] for b in st.session_state["bins"]]
    # If true drag available, show multi-list drag; else fallback to click-assign UI
    if DRAG_AVAILABLE:
        left_col, right_col = st.columns([1, 2])
        with left_col:
            st.subheader("Word bank")
            bank = sort_items(st.session_state["bank"], key="bank")
            st.session_state["bank"] = bank

        with right_col:
            st.subheader("Bins")
            lists = [st.session_state["drag_state"].get(lbl, []) for lbl in labels]
            updated = sort_multiple(lists, labels=labels, key="bins_multi")
            for lbl, lst in zip(labels, updated):
                st.session_state["drag_state"][lbl] = lst

        if st.button("Check bins"):
            correct = 0; total = 0
            for lbl, lst in st.session_state["drag_state"].items():
                for s in lst:
                    total += 1
                    want = st.session_state["ansmap"].get(s, "no_change")
                    got = "accumulate" if lbl.lower().startswith("accum") else lbl.lower()
                    if got == want:
                        correct += 1
            if total == 0:
                st.warning("Drag items into bins first.")
            elif correct == total:
                st.success("All bins correct! 🎉")
            else:
                st.warning(f"{correct}/{total} correct — adjust and try again.")

    else:
        # Click-to-assign fallback (stable)
        left, right = st.columns([1, 2])
        with left:
            st.subheader("Word bank")
            chosen = st.session_state.get("chosen_item")
            for idx, s in enumerate(st.session_state["bank"]):
                if st.button(("🔘 " if chosen == s else "⚪ ") + s, key=f"bank_{idx}"):
                    st.session_state["chosen_item"] = s
            if st.button("Reset"):
                # put everything back
                st.session_state["bank"] = list(st.session_state["ansmap"].keys())
                st.session_state["drag_state"] = {lbl: [] for lbl in labels}
                st.session_state["chosen_item"] = None

        with right:
            st.subheader("Bins")
            cols = st.columns(2)
            for i, lbl in enumerate(labels):
                col = cols[i % 2]
                with col:
                    st.write(f"**{lbl}**")
                    # placement button
                    if st.button("Place here", key=f"place_{lbl}"):
                        sel = st.session_state.get("chosen_item")
                        if sel:
                            # remove from bank
                            st.session_state["bank"] = [x for x in st.session_state["bank"] if x != sel]
                            # place in bin
                            st.session_state["drag_state"][lbl].append(sel)
                            st.session_state["chosen_item"] = None
                    # show contents + remove buttons
                    for j, s in enumerate(list(st.session_state["drag_state"][lbl])):
                        cols2 = st.columns([3,1])
                        cols2[0].write(f"- {s}")
                        if cols2[1].button("✖", key=f"rm_{lbl}_{j}"):
                            # remove and return to bank
                            st.session_state["drag_state"][lbl].pop(j)
                            st.session_state["bank"].append(s)

            if st.button("Check bins"):
                correct = 0; total = 0
                for lbl, lst in st.session_state["drag_state"].items():
                    for s in lst:
                        total += 1
                        want = st.session_state["ansmap"].get(s, "no_change")
                        got = "accumulate" if lbl.lower().startswith("accum") else lbl.lower()
                        if got == want:
                            correct += 1
                if total == 0:
                    st.warning("Assign items to bins first.")
                elif correct == total:
                    st.success("All bins correct! 🎉")
                else:
                    st.warning(f"{correct}/{total} correct — adjust and try again.")
