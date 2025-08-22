import base64, io, json, random, time, zipfile, pathlib, os
from typing import List, Dict
import streamlit as st
from openai import OpenAI

# -------------------------------
# Config & paths
# -------------------------------
st.set_page_config(page_title="Cell Bio Tutor", layout="centered")
st.title("🧬 Cell Bio Tutor")

REPO_ROOT = pathlib.Path(__file__).parent
FIB_MASTER = REPO_ROOT / "templates" / "rtk_fill_in_blanks_FIXED_blocks.h5p"
DND_MASTER = REPO_ROOT / "templates" / "cellular_respiration_aligned_course_style.h5p"

# OpenAI client (optional LLM mode)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", None))
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------------
# Session state
# -------------------------------
def init_state():
    defaults = dict(
        topic="",
        difficulty="easy",      # easy / medium / hard
        accuracy_window=[],     # last 5 MCQ results
        last_fib_bytes=None,
        last_dnd_bytes=None,
        last_fib_lines=[],
        use_ai=True,            # toggle LLM authoring
        n_items=3,              # items per FIB activity
    )
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v
init_state()

# -------------------------------
# Utilities
# -------------------------------
def render_h5p_inline(h5p_bytes: bytes, height: int = 560):
    """Render a .h5p package inline using h5p-standalone (CDN)."""
    b64 = base64.b64encode(h5p_bytes).decode("utf-8")
    html = f"""
    <div id="h5p-container" style="border:0; margin:0; padding:0;"></div>
    <script src="https://unpkg.com/h5p-standalone@1.3.0/dist/main.bundle.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/h5p-standalone@1.3.0/dist/styles/h5p.css">
    <script>
      H5PStandalone.display('#h5p-container', {{
        h5pContent: "data:application/zip;base64,{b64}",
        frameJs: "https://unpkg.com/h5p-standalone@1.3.0/dist/frame.bundle.js",
        frameCss: "https://unpkg.com/h5p-standalone@1.3.0/dist/styles/h5p.css"
      }});
    </script>
    """
    st.components.v1.html(html, height=height, scrolling=True)

def record_result(correct: bool):
    st.session_state.accuracy_window.append(int(correct))
    if len(st.session_state.accuracy_window) > 5:
        st.session_state.accuracy_window = st.session_state.accuracy_window[-5:]
    acc = sum(st.session_state.accuracy_window) / max(1, len(st.session_state.accuracy_window))
    st.session_state.difficulty = "hard" if acc>=0.8 else "medium" if acc>=0.5 else "easy"

# -------------------------------
# LLM-authored critical-thinking lines (FIB)
# -------------------------------
def generate_ct_fib_lines_llm(topic: str, difficulty: str, n_items: int = 3) -> List[str]:
    """
    Returns concise, critical-thinking lines with exactly ONE blank using H5P Blanks syntax:
    Surround acceptable answers with asterisks, and variants with slashes: *increase/increased*.
    """
    if not client:
        raise RuntimeError("OPENAI_API_KEY missing")

    sys = (
        "You are a college Cell Biology tutor. "
        "Generate short, critical-thinking prompts that ask students to PREDICT downstream effects of perturbations. "
        "Each item must be a single sentence with exactly one blank using H5P Fill-in-the-Blanks syntax: "
        "surround acceptable answers with asterisks, and variants with slashes, e.g., *increase/increased*. "
        "Keep cognitive load low and avoid jargon bloat."
    )
    user = (
        f"Topic: {topic}\n"
        f"Difficulty: {difficulty}\n"
        f"Create {n_items} items. Each one sentence with one blank. "
        f"Prefer 'increase/decrease' predictions about downstream components.\n"
        f"Examples of style:\n"
        f"- If Complex I activity rises, proton motive force will *increase/increased*.\n"
        f"- If RTK can't bind ligand, MAPK phosphorylation will *decrease/decreased*.\n"
        f"Return as a numbered list of plain lines only."
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.3,
    )
    text = resp.choices[0].message.content
    # Parse numbered lines
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        # strip leading numbers like "1) " or "1. "
        s = s.lstrip("0123456789). ").strip()
        if "*" in s:
            lines.append(s)
    return lines[:n_items] if lines else [
        "If the inner membrane becomes leaky to H+, ATP synthase output will *decrease/decreased*."
    ]

# -------------------------------
# Rule-based fallback CT lines
# -------------------------------
def ct_fib_lines_rule(topic: str, n_items: int = 3) -> List[str]:
    t = (topic or "").lower().strip()
    bank = []
    if "electron transport" in t or "etc" in t:
        bank = [
            "If Complex I activity increases, proton pumping and Δp will *increase/increased*.",
            "If the inner membrane becomes leaky to H+, ATP synthase output will *decrease/decreased*.",
            "If oxygen is unavailable, electron flow through Complex IV will *decrease/decreased*.",
        ]
    elif "rtk" in t or "receptor tyrosine" in t:
        bank = [
            "If RTK is constitutively active, MAPK phosphorylation will *increase/increased*.",
            "If Grb2 cannot bind RTK, Ras activation will *decrease/decreased*.",
            "If PI3K is hyperactive, AKT activity will *increase/increased*.",
        ]
    elif "glycolysis" in t:
        bank = [
            "If ATP binds PFK-1’s regulatory site, glycolytic flux will *decrease/decreased*.",
            "If AMP rises in the cell, PFK-1 activity will *increase/increased*.",
            "If pyruvate kinase is inhibited, pyruvate output will *decrease/decreased*.",
        ]
    else:
        bank = [
            "If lysosomal pH rises, hydrolase activity will *decrease/decreased*.",
            "If microtubules depolymerize, kinesin-based transport will *decrease/decreased*.",
            "If mitochondrial Δψ increases, ATP output will *increase/increased*.",
        ]
    return bank[:n_items]

# -------------------------------
# Build H5P: FIB from your MASTER (DO NOT change master metadata)
# - content.json must keep your master shape: text (instructions) + questions (Text blocks)
# -------------------------------
def build_fib_from_master(master_path: pathlib.Path, instructions: str, lines: List[str]) -> bytes:
    with zipfile.ZipFile(master_path, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    # Your master expects 'content/content.json' with keys:
    #   text: string (instructions)
    #   questions: list of HTML strings, each with *answers/variants*
    content = json.loads(files["content/content.json"].decode("utf-8"))
    content["text"] = instructions
    content["questions"] = [f"<p>{ln}</p>" for ln in lines]

    # Make answers lenient where supported
    beh = content.get("behaviour", {})
    beh["caseSensitive"] = False
    beh["ignorePunctuation"] = True
    beh["acceptSpellingErrors"] = True
    content["behaviour"] = beh

    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            if name == "content/content.json":
                zout.writestr(name, json.dumps(content, ensure_ascii=False, indent=2))
            else:
                zout.writestr(name, data)
    return out_buf.getvalue()

# -------------------------------
# Build H5P: DnD from your MASTER (compact style)
# Assumes your master stores geometry under question.task.elements / dropZones
# -------------------------------
def build_dnd_from_master(master_path: pathlib.Path, elements: List[Dict], zones: List[Dict]) -> bytes:
    with zipfile.ZipFile(master_path, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    content = json.loads(files["content/content.json"].decode("utf-8"))
    # Map into the master structure
    content["question"]["task"]["elements"] = [{
        "x": e["x"], "y": e["y"], "width": e["w"], "height": e["h"],
        "dropZones": [str(i) for i in range(len(zones))],
        "backgroundOpacity": 0,
        "type": {"library": "H5P.AdvancedText 1.1", "params": {"text": f"<span>{e['text']}</span>"}}
    } for e in elements]

    content["question"]["task"]["dropZones"] = [{
        "x": zc["x"], "y": zc["y"], "width": zc["w"], "height": zc["h"],
        "label": f"<div style='text-align:center'>{zc['label']}</div>",
        "showLabel": True, "backgroundOpacity": 0,
        "autoAlign": True, "single": False,
        "correctElements": [str(zc["correct_idx"])]
    } for zc in zones]

    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            if name == "content/content.json":
                zout.writestr(name, json.dumps(content, ensure_ascii=False, indent=2))
            else:
                zout.writestr(name, data)
    return out_buf.getvalue()

def default_dnd_geometry():
    # Use your compact, left/right geometry
    elements = [
        {"text": "Glycolysis", "x": 1.0, "y": 6.5, "w": 8.5, "h": 3.2},
        {"text": "Pyruvate oxidation", "x": 0.8, "y": 24.5, "w": 9.0, "h": 3.2},
        {"text": "Citric acid cycle", "x": 0.8, "y": 42.5, "w": 9.0, "h": 3.2},
        {"text": "ETC", "x": 0.8, "y": 60.5, "w": 6.0, "h": 3.2},
        {"text": "ATP synthase", "x": 3.0, "y": 78.5, "w": 7.5, "h": 3.2},
    ]
    zones = [
        {"label": "Cytoplasm • 2 ATP + 2 NADH", "x": 45.8, "y": 13.9, "w": 16.2, "h": 1.9, "correct_idx": 0},
        {"label": "Mito. matrix • Acetyl-CoA + NADH", "x": 43.5, "y": 30.6, "w": 18.1, "h": 1.9, "correct_idx": 1},
        {"label": "Mito. matrix • NADH, FADH₂, CO₂, 2 ATP", "x": 40.3, "y": 47.2, "w": 20.6, "h": 1.9, "correct_idx": 2},
        {"label": "Inner mem. • Proton gradient (NADH/FADH₂ → e⁻)", "x": 33.9, "y": 63.9, "w": 25.6, "h": 1.9, "correct_idx": 3},
        {"label": "Inner mem. • ~34 ATP via gradient", "x": 43.5, "y": 80.6, "w": 17.5, "h": 1.9, "correct_idx": 4},
    ]
    return elements, zones

# -------------------------------
# Adaptive MCQs (short & targeted)
# -------------------------------
MCQ_BANK = {
    "etc": {
        "easy": [
            ("The terminal electron acceptor of the ETC is:", ["Oxygen (O2)", "NAD+", "FAD", "CO2"], 0),
            ("Protons are pumped across the:", ["Inner mitochondrial membrane", "Outer membrane", "Matrix", "Cytosol"], 0),
        ],
        "medium": [
            ("If Complex III is inhibited first, which decreases immediately?",
             ["Δp (proton motive force)", "ATP synthase rotation", "NADH oxidation at CI", "Citrate synthase"], 2),
        ],
        "hard": [
            ("Uncouplers primarily reduce:", ["Proton gradient (Δp)", "Electron transfer rates",
                                             "Substrate-level phosphorylation", "mtDNA replication"], 0),
        ]
    },
    "rtk": {
        "easy": [
            ("Ligand binding to RTK typically causes:", ["Dimerization", "Endocytosis only", "Gα activation", "MAP phosphatase activation"], 0),
        ],
        "medium": [
            ("Grb2 binds RTK via:", ["SH2 domain", "PH domain", "PTB domain", "EF hand"], 0),
        ],
        "hard": [
            ("Constitutively active RTK most directly elevates:", ["MAPK phosphorylation", "RNA splicing", "Proteasome degradation", "β-oxidation"], 0),
        ]
    }
}
def topic_key(topic: str) -> str:
    t = (topic or "").lower()
    if "electron transport" in t or "etc" in t: return "etc"
    if "rtk" in t or "receptor tyrosine" in t: return "rtk"
    return "etc"
def pick_mcqs(topic: str, difficulty: str):
    key = topic_key(topic)
    pool = MCQ_BANK[key].get(difficulty, MCQ_BANK[key]["easy"])
    return random.sample(pool, k=min(2, len(pool)))
def ask_mcqs(topic: str):
    st.subheader("Quick Check")
    for i, (q, options, correct_idx) in enumerate(pick_mcqs(topic, st.session_state.difficulty)):
        key = f"mcq_{time.time()}_{i}"
        sel = st.radio(q, options, key=key, index=None)
        if sel is not None:
            correct = (options.index(sel) == correct_idx)
            record_result(correct)
            st.markdown("✅ Correct!" if correct else "❌ Not quite — review the concept.")
            st.divider()

# -------------------------------
# UI: student prompt -> activities
# -------------------------------
with st.sidebar:
    st.header("Authoring options")
    st.session_state.use_ai = st.toggle("Use AI to generate items", value=True if client else False,
                                        help="If off (or no API key), uses built-in rule-based items.")
    st.session_state.n_items = st.slider("Items per activity", 2, 6, 3, 1)

topic = st.text_input("What do you want help with? (e.g., “electron transport chain”, “RTK”):", value=st.session_state.topic)
colA, colB = st.columns([1,1])
with colA:
    gen_fib = st.button("Generate Fill-in-the-Blanks")
with colB:
    gen_dnd = st.button("Generate Drag-and-Drop")

# FIB flow
if gen_fib:
    st.session_state.topic = topic
    try:
        lines = generate_ct_fib_lines_llm(topic, st.session_state.difficulty, st.session_state.n_items) \
                if (st.session_state.use_ai and client) else \
                ct_fib_lines_rule(topic, st.session_state.n_items)
    except Exception as e:
        st.warning(f"AI generation failed, using fallback. ({e})")
        lines = ct_fib_lines_rule(topic, st.session_state.n_items)

    st.session_state.last_fib_lines = lines
    instructions = "Predict whether the following situations would increase or decrease protein activity:"
    fib_bytes = build_fib_from_master(FIB_MASTER, instructions, lines)
    st.session_state.last_fib_bytes = fib_bytes

    st.success("Your Fill-in-the-Blanks activity is ready below ⤵")
    render_h5p_inline(fib_bytes, height=560)

    st.subheader("Critical Thinking (preview)")
    st.caption("Same content embedded in the H5P above.")
    for i, ln in enumerate(lines, 1):
        st.write(f"{i}) {ln}")

    ask_mcqs(topic)
    st.info(f"Difficulty now set to **{st.session_state.difficulty}** based on your recent answers.")

# DnD flow (optional; keeps your master’s compact layout)
if gen_dnd:
    elements, zones = default_dnd_geometry()
    dnd_bytes = build_dnd_from_master(DND_MASTER, elements, zones)
    st.session_state.last_dnd_bytes = dnd_bytes
    st.success("Your Drag-and-Drop activity is ready below ⤵")
    render_h5p_inline(dnd_bytes, height=520)
