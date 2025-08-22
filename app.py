import base64, io, json, random, time, zipfile, pathlib, os
from typing import List, Dict, Optional
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="Cell Bio Tutor", layout="centered")
st.title("🧬 Cell Bio Tutor")

REPO_ROOT = pathlib.Path(__file__).parent
FIB_MASTER = REPO_ROOT / "templates" / "rtk_fill_in_blanks_FIXED_blocks.h5p"
DND_MASTER = REPO_ROOT / "templates" / "cellular_respiration_aligned_course_style.h5p"

# ---------- OpenAI (robust init) ----------
def get_openai_client() -> Optional[object]:
    """Initialize OpenAI client safely. If anything fails, return None and show a warning."""
    try:
        from openai import OpenAI  # requires openai >= 1.x
    except Exception as e:
        st.warning(f"OpenAI library import failed. Using rule-based items. ({e})")
        return None

    api_key = os.getenv("OPENAI_API_KEY", None)
    if not api_key:
        # Streamlit Cloud: secrets live in st.secrets
        api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    if not api_key:
        st.info("No OPENAI_API_KEY found — continuing with rule-based items.")
        return None

    try:
        # IMPORTANT: don’t pass custom base_url unless you know you need it
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error("Couldn’t initialize OpenAI. Falling back to rule-based items.")
        st.caption(f"(Details for admin: {e})")
        return None

client = get_openai_client()

# ---------- Session ----------
def init_state():
    defaults = dict(
        topic="",
        difficulty="easy",
        accuracy_window=[],
        n_items=3,
        use_ai=True if client else False,
        last_fib_bytes=None,
        last_dnd_bytes=None,
    )
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v
init_state()

# ---------- Inline H5P ----------
def render_h5p_inline(h5p_bytes: bytes, height: int = 560):
    if not h5p_bytes:
        return
    b64 = base64.b64encode(h5p_bytes).decode("utf-8")
    html = f"""
    <div id="h5p-container"></div>
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

# ---------- Adaptivity ----------
def record_result(correct: bool):
    st.session_state.accuracy_window.append(int(correct))
    if len(st.session_state.accuracy_window) > 5:
        st.session_state.accuracy_window = st.session_state.accuracy_window[-5:]
    acc = sum(st.session_state.accuracy_window)/max(1,len(st.session_state.accuracy_window))
    st.session_state.difficulty = "hard" if acc>=0.8 else "medium" if acc>=0.5 else "easy"

# ---------- LLM & Fallback item generation ----------
def generate_ct_fib_lines_llm(client, topic: str, difficulty: str, n_items: int) -> List[str]:
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
        f"Create {n_items} items. "
        f"Prefer 'increase/decrease' predictions about downstream components.\n"
        f"Examples:\n"
        f"- If Complex I activity rises, proton motive force will *increase/increased*.\n"
        f"- If RTK can't bind ligand, MAPK phosphorylation will *decrease/decreased*.\n"
        f"Return as a numbered list of plain lines only."
    )
    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0.3,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    text = resp.choices[0].message.content
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        s = s.lstrip("0123456789). ").strip()
        if "*" in s:
            lines.append(s)
    return lines[:n_items] if lines else [
        "If the inner membrane becomes leaky to H+, ATP synthase output will *decrease/decreased*."
    ]

def ct_fib_lines_rule(topic: str, n_items: int) -> List[str]:
    t = (topic or "").lower()
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
    else:
        bank = [
            "If lysosomal pH rises, hydrolase activity will *decrease/decreased*.",
            "If microtubules depolymerize, kinesin-based transport will *decrease/decreased*.",
            "If mitochondrial Δψ increases, ATP output will *increase/increased*.",
        ]
    return bank[:n_items]

# ---------- Build FIB from your MASTER ----------
def build_fib_from_master(master_path: pathlib.Path, instructions: str, lines: List[str]) -> bytes:
    with zipfile.ZipFile(master_path, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}
    # Your master uses content/content.json with "text" and "questions"
    content = json.loads(files["content/content.json"].decode("utf-8"))
    content["text"] = instructions
    content["questions"] = [f"<p>{ln}</p>" for ln in lines]
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

# ---------- Build DnD from your MASTER ----------
def default_dnd_geometry():
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

def build_dnd_from_master(master_path: pathlib.Path, elements: List[Dict], zones: List[Dict]) -> bytes:
    with zipfile.ZipFile(master_path, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}
    content = json.loads(files["content/content.json"].decode("utf-8"))
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

# ---------- MCQs ----------
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

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Authoring options")
    st.session_state.n_items = st.slider("Items per activity", 2, 6, 3, 1)
    if client:
        st.session_state.use_ai = st.toggle("Use AI to generate items", value=True)
    else:
        st.session_state.use_ai = False

# ---------- UI ----------
topic = st.text_input("What do you want help with? (e.g., “electron transport chain”, “RTK”):", value=st.session_state.topic)
generate = st.button("Generate Activity")

if generate:
    st.session_state.topic = topic

    # 1) Build items
    try:
        if st.session_state.use_ai and client:
            lines = generate_ct_fib_lines_llm(client, topic, st.session_state.difficulty, st.session_state.n_items)
        else:
            lines = ct_fib_lines_rule(topic, st.session_state.n_items)
    except Exception as e:
        st.warning(f"AI generation failed, using fallback. ({e})")
        lines = ct_fib_lines_rule(topic, st.session_state.n_items)

    # 2) Build FIB H5P from your MASTER
    if not FIB_MASTER.exists():
        st.error("FIB master not found in templates/. Make sure rtk_fill_in_blanks_FIXED_blocks.h5p is committed.")
    else:
        instructions = "Predict whether the following situations would increase or decrease protein activity:"
        try:
            fib_bytes = build_fib_from_master(FIB_MASTER, instructions, lines)
            st.success("Fill-in-the-Blanks activity ready below ⤵")
            render_h5p_inline(fib_bytes, height=560)
            st.caption("Tip: if you don’t see the H5P, your network may be blocking the CDN (unpkg).")
        except Exception as e:
            st.error("Couldn’t build the FIB H5P from master.")
            st.caption(f"(Details for admin: {e})")

    # 3) Quick MCQs + adapt
    ask_mcqs(topic)
    st.info(f"Difficulty now set to **{st.session_state.difficulty}** based on recent answers.")
