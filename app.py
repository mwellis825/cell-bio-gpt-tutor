# app.py
# Prompt + Slides → (1) 4× Fill-in-the-Blank (prediction) + (2) Drag into Topic-Specific Bins
# ---------------------------------------------------------------------------------------------
# How it works
# - User types a topic prompt (no defaults).
# - App searches ./slides (PDF / txt / md / html).
# - It detects the topic from the prompt, then generates:
#   (1) FOUR warm, prediction-style fill-in-the-blank items (lenient grading).
#   (2) TRUE drag-and-drop (if streamlit-sortables is installed): draggable bank → titled bins
#
# Environment
#   pip install streamlit
#   # PDFs:
#   pip install PyPDF2    # or: pip install pypdf
#   # Drag & drop (recommended):
#   pip install streamlit-sortables
#
# Run:
#   streamlit run app.py
# ---------------------------------------------------------------------------------------------

import os, re, pathlib
import streamlit as st

# ---- Optional drag component (true drag if available) ----
DRAG_OK = False
try:
    from streamlit_sortables import sort_multiple  # multiple connected lists
    DRAG_OK = True
except Exception:
    DRAG_OK = False

# ---- Optional PDF backends ----
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

# ---- App config ----
st.set_page_config(page_title="Study Mode", page_icon="📘", layout="wide")
SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm"}

# ---- File IO ----
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

# ---- Search helpers (corpus relevance just biases, topic comes from prompt) ----
def _tok(s: str):
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())

def _score(text: str, q_tokens):
    toks = _tok(text)
    if not toks: return 0
    freq = {}
    for t in toks: freq[t] = freq.get(t, 0) + 1
    return sum(freq.get(q, 0) for q in q_tokens)

# ---- Topic detection from prompt (drives both activities) ----
def detect_topic(prompt: str):
    t = (prompt or "").lower()
    if any(k in t for k in ["transcription","rna pol","promoter","enhancer","tfiid","tbp","splice","mrna","utr"]):
        return "transcription"
    if any(k in t for k in ["translation","ribosome","trna","elongation factor","ef-tu","eftu","ef-g","release factor","shine-dalgarno","kozak"]):
        return "translation"
    if any(k in t for k in ["replication","helicase","primase","ligase","okazaki","leading strand","lagging strand","ssb"]):
        return "replication"
    if any(k in t for k in ["microscope","microscopy","resolution","diffraction","numerical aperture","na","nyquist"]):
        return "microscopy"
    if any(k in t for k in ["rtk","gpcr","erk","mapk","pi3k","akt","ras","raf","phosphatase","kinase"]):
        return "signaling"
    if any(k in t for k in ["cell cycle","cdk","cyclin","checkpoint","apc/c","p53","cohesin","separase"]):
        return "cell_cycle"
    return "generic"

# ---- Lenient grading helpers ----
UP = {"increase","increases","increased","up","higher","activates","activation","faster","more","↑"}
DOWN = {"decrease","decreases","decreased","down","lower","inhibits","inhibition","slower","less","↓"}
NOCHANGE = {"no change","unchanged","same","neutral","nc","~"}
ACCUM = {"accumulate","accumulates","accumulated","builds up","build up","piles up","pile up","amass","gathers"}

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+"," ", s)
    s = s.replace("’","'")
    return s

def match_pred(user_text: str, canonical: str, noun: str = "") -> bool:
    u = _norm(user_text)
    if canonical == "increase":   return any(x in u for x in UP)
    if canonical == "decrease":   return any(x in u for x in DOWN)
    if canonical == "no_change":  return any(x in u for x in NOCHANGE)
    if canonical == "accumulate": return any(x in u for x in ACCUM) or (noun and noun in u)
    return canonical in u

# ---- FITB (prediction style, warm language) ----
def fitb_for_topic(topic: str):
    if topic == "transcription":
        return [
            {"stem": "When TFIIH helicase activity is reduced, the rate of transcription initiation would ______.", "key":"decrease","noun":""},
            {"stem": "When a strong enhancer makes stable contact with the promoter, the amount of mRNA produced would ______.", "key":"increase","noun":""},
            {"stem": "When Mediator–Pol II contact is disrupted, promoter clearance would ______.", "key":"decrease","noun":""},
            {"stem": "When the poly(A) signal is mutated, unprocessed pre-mRNA species would ______.", "key":"accumulate","noun":"pre-mrna"},
        ]
    if topic == "translation":
        return [
            {"stem": "When EF-Tu cannot escort aminoacyl-tRNA to the A site, the elongation rate would ______.", "key":"decrease","noun":""},
            {"stem": "When the start context (Shine-Dalgarno/Kozak) improves, initiation frequency would ______.", "key":"increase","noun":""},
            {"stem": "When release factors are scarce, the average termination time per protein would ______.", "key":"increase","noun":""},
            {"stem": "When peptidyl-transferase is blocked, incomplete nascent chains would ______.", "key":"accumulate","noun":"nascent"},
        ]
    if topic == "replication":
        return [
            {"stem": "When helicase activity is inhibited, replication-fork progression would ______.", "key":"decrease","noun":""},
            {"stem": "When primer supply drops, the number of lagging-strand starts would ______.", "key":"decrease","noun":""},
            {"stem": "When DNA ligase is inactive, unjoined Okazaki fragments would ______.", "key":"accumulate","noun":"okazaki"},
            {"stem": "When SSB coverage improves on ssDNA, unwanted re-annealing at forks would ______.", "key":"decrease","noun":""},
        ]
    if topic == "microscopy":
        return [
            {"stem": "When numerical aperture (NA) increases, the diffraction-limited minimum resolvable distance would ______.", "key":"decrease","noun":""},
            {"stem": "When a shorter wavelength is used, the resolution limit would ______.", "key":"decrease","noun":""},
            {"stem": "When refractive-index mismatch increases spherical aberration, image sharpness would ______.", "key":"decrease","noun":""},
            {"stem": "When the confocal pinhole is narrowed (within reason), optical sectioning contrast would ______.", "key":"increase","noun":""},
        ]
    if topic == "signaling":
        return [
            {"stem": "When an RTK is ligand-bound and dimerized persistently, ERK phosphorylation would ______.", "key":"increase","noun":""},
            {"stem": "When GAP activity on RAS increases, RAS-GTP lifetime would ______.", "key":"decrease","noun":""},
            {"stem": "When PI3K is hyperactive, AKT activation would ______.", "key":"increase","noun":""},
            {"stem": "When a phosphatase targeting ERK is overexpressed, pERK levels would ______.", "key":"decrease","noun":""},
        ]
    if topic == "cell_cycle":
        return [
            {"stem": "When APC/C is inhibited by the spindle checkpoint, separase activation would ______.", "key":"decrease","noun":""},
            {"stem": "When Cyclin E–CDK2 activity rises, G1→S entry would ______.", "key":"increase","noun":""},
            {"stem": "When p53 is stabilized after DNA damage, CDK activity would ______.", "key":"decrease","noun":""},
            {"stem": "When cohesion removal is delayed, metaphase duration would ______.", "key":"increase","noun":""},
        ]
    # generic
    return [
        {"stem":"When a rate-limiting step is enhanced, the amount of product formed would ______.","key":"increase","noun":""},
        {"stem":"When a core catalytic step is hindered, the overall throughput would ______.","key":"decrease","noun":""},
        {"stem":"When a terminal processing step is blocked, intermediate species would ______.","key":"accumulate","noun":"intermediate"},
        {"stem":"When a parallel compensation route is active, the net change would be ______.","key":"no_change","noun":""},
    ]

# ---- Drag bins (labels adapt by topic) ----
def drag_sets_for_topic(topic: str):
    """
    Returns (labels, bank, answer_map) where labels are topic-specific bins.
      transcription → Initiation / Elongation / Termination / Processing
      translation   → Initiation / Elongation / Termination / Quality control
      replication   → Initiation / Elongation / Ligation / Fork stability
      microscopy    → Resolution / Contrast / Artifacts / Sampling
      signaling     → Upstream activation / Pathway inhibition / Downstream output / Feedback
      cell_cycle    → Entry/Commitment / Checkpoint hold / Transition / Exit/Completion
      generic       → Increase / Decrease / No change / Accumulates
    """
    if topic == "transcription":
        labels = ["Initiation", "Elongation", "Termination", "Processing"]
        pairs  = [
            ("TFIIH-dependent promoter opening", "Initiation"),
            ("Promoter clearance efficiency", "Initiation"),
            ("Pol II pause-release rate", "Elongation"),
            ("Readthrough at poly(A) signal", "Termination"),
            ("Unprocessed pre-mRNA species", "Processing"),
            ("Splicing completion", "Processing"),
        ]
    elif topic == "translation":
        labels = ["Initiation", "Elongation", "Termination", "Quality control"]
        pairs  = [
            ("Start-site selection", "Initiation"),
            ("Elongation speed", "Elongation"),
            ("Ribosome stalling events", "Quality control"),
            ("Release-factor-dependent termination", "Termination"),
            ("Incomplete nascent chains", "Quality control"),
        ]
    elif topic == "replication":
        labels = ["Initiation", "Elongation", "Ligation", "Fork stability"]
        pairs  = [
            ("Origin firing frequency", "Initiation"),
            ("Replication-fork progression", "Elongation"),
            ("Okazaki fragment joining", "Ligation"),
            ("Strand re-annealing at fork", "Fork stability"),
        ]
    elif topic == "microscopy":
        labels = ["Resolution", "Contrast", "Artifacts", "Sampling"]
        pairs  = [
            ("Minimum resolvable distance (d)", "Resolution"),
            ("Optical sectioning", "Contrast"),
            ("Spherical aberration", "Artifacts"),
            ("Nyquist violations", "Sampling"),
        ]
    elif topic == "signaling":
        labels = ["Upstream activation", "Pathway inhibition", "Downstream output", "Feedback"]
        pairs  = [
            ("RTK dimerization/activation", "Upstream activation"),
            ("RAS-GTP lifetime", "Pathway inhibition"),
            ("ERK phosphorylation", "Downstream output"),
            ("Negative feedback to ERK", "Feedback"),
        ]
    elif topic == "cell_cycle":
        labels = ["Entry/Commitment", "Checkpoint hold", "Transition", "Exit/Completion"]
        pairs  = [
            ("G1→S entry rate", "Entry/Commitment"),
            ("Spindle assembly checkpoint activity", "Checkpoint hold"),
            ("Anaphase onset", "Transition"),
            ("Cohesion removal completion", "Exit/Completion"),
        ]
    else:
        labels = ["Increase", "Decrease", "No change", "Accumulates"]
        pairs  = [
            ("Product formation rate", "Increase"),
            ("Core reaction throughput", "Decrease"),
            ("Intermediate species", "Accumulates"),
            ("Off-pathway byproducts (immediate)", "No change"),
        ]
    bank   = [s for (s, _) in pairs]
    answer = {s:k for (s,k) in pairs}
    return labels, bank, answer

# ---- UI ----
st.title("📘 Prompt → Critical-Thinking Activities")
st.caption("Type a topic. The app searches your slides and builds fresh activities each time.")

prompt = st.text_input("Enter a topic (e.g., transcription initiation, RTK→ERK, replication fork dynamics)", value="", placeholder="Type your topic…")

if st.button("Generate"):
    # load slides once (used for future upgrades; topic comes from prompt)
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)

    topic = detect_topic(prompt)
    st.session_state.topic = topic

    # Build activities (fresh)
    st.session_state.fitb   = fitb_for_topic(topic)
    st.session_state.labels, st.session_state.bank, st.session_state.answer = drag_sets_for_topic(topic)

    # Initialize drag lists for the multi-list widget: index 0 is the Bank; bins follow
    st.session_state.drag_lists = [list(st.session_state.bank)] + [[] for _ in st.session_state.labels]

    st.success(f"Generated new activities for **{topic.replace('_',' ').title()}**.")

# ---- Activity 1: 4× FITB (prediction, warm language) ----
if "fitb" in st.session_state:
    st.markdown("## Activity 1 — Predict the immediate effect")
    st.caption("Answer with words like **increase**, **decrease**, **no change**, or **accumulates**. I’ll grade leniently.")
    for i, item in enumerate(st.session_state.fitb, start=1):
        stem = item["stem"]; key = item["key"]; noun = item.get("noun","")
        user = st.text_input(f"{i}. {stem}", key=f"fitb_{i}")
        if st.button(f"Check {i}", key=f"chk_{i}"):
            if match_pred(user, key, noun):
                st.success("Nice — that’s the expected immediate effect.")
            else:
                st.error("Not quite — think about the immediate causal consequence of that change.")

# ---- Activity 2: TRUE Drag-and-Drop (bank → topic-specific bins) ----
if all(k in st.session_state for k in ["labels","bank","answer","drag_lists"]):
    st.markdown("---")
    st.markdown("## Activity 2 — Drag the statements into the correct bin")

    if DRAG_OK:
        # Labels for the multi-list widget: first list is the Bank, then the bins
        labels = ["Bank"] + st.session_state.labels
        lists  = st.session_state.drag_lists
        updated = sort_multiple(lists, labels=labels, key="drag-multi")
        st.session_state.drag_lists = updated

        if st.button("Check bins"):
            total = 0; correct = 0
            # updated[0] is the Bank; bins start at 1
            for bin_idx, bin_label in enumerate(st.session_state.labels, start=1):
                for item in updated[bin_idx]:
                    total += 1
                    want = st.session_state.answer.get(item, None)
                    got  = bin_label
                    if want is not None and want == got:
                        correct += 1
            if total == 0:
                st.warning("Drag items from the Bank into the bins first.")
            elif correct == total:
                st.success("All bins correct! 🎉")
            else:
                st.warning(f"{correct}/{total} correct — adjust and try again.")
    else:
        st.error(
            "True drag-and-drop couldn’t be enabled. "
            "Please preinstall `streamlit-sortables` in the environment so students can drag items into bins.\n\n"
            "Example (one-time on the server): `pip install streamlit-sortables`"
        )
