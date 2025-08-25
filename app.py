
import os, re, json, pathlib, random, time, sys
import streamlit as st
from typing import List, Tuple, Dict

# ------------------ Page & Globals ------------------
st.set_page_config(page_title="Let's Practice Biology!", page_icon="🎓", layout="wide")
SLIDES_DIR = os.path.join(os.getcwd(), "slides")

def new_seed() -> int:
    return int(time.time() * 1000) ^ random.randint(0, 1_000_000)

# ------------------ LLM hint (optional) ------------------
def _try_llm(prompt: str) -> str:
    api = os.environ.get("OPENAI_API_KEY")
    if not api:
        raise RuntimeError("No API key")
    import openai  # type: ignore
    openai.api_key = api
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a Socratic college-level cell biology tutor. Ask ONE concise guiding question (< 20 words) that nudges the student to the answer without revealing it."},
            {"role":"user","content":prompt},
        ],
        temperature=0.3,
        max_tokens=60,
    )
    return resp["choices"][0]["message"]["content"].strip()

def llm_hint_for_fitb(stem: str, key: str, topic: str, fallback: str) -> str:
    try:
        return _try_llm(f"Topic: {topic}\nFITB stem: {stem}\nTarget concept (label): {key}\nWrite one guiding question.")
    except Exception:
        return fallback

def llm_hint_for_dnd(item: str, target_label: str, labels: list, topic: str, fallback: str) -> str:
    try:
        return _try_llm(
            "You will write a hint for a drag-and-drop classification.\n"
            f"Topic: {topic}\n"
            f"Draggable item: {item}\n"
            f"Bins: {', '.join(labels)}\n"
            f"Correct bin: {target_label}\n"
            "Write ONE targeted, non-revealing hint that helps the student distinguish the correct bin from the others."
        )
    except Exception:
        return fallback

# ------------------ Simple text extraction from slides ------------------
PDF_BACKEND = None
try:
    import PyPDF2  # type: ignore
    PDF_BACKEND = "PyPDF2"
except Exception:
    try:
        import pypdf  # type: ignore
        PDF_BACKEND = "pypdf"
    except Exception:
        PDF_BACKEND = None

def read_pdf(path: str) -> str:
    if PDF_BACKEND == "PyPDF2":
        try:
            with open(path,"rb") as f:
                reader = PyPDF2.PdfReader(f)  # type: ignore
                return "\\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""
    if PDF_BACKEND == "pypdf":
        try:
            with open(path,"rb") as f:
                reader = pypdf.PdfReader(f)  # type: ignore
                return "\\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""
    return ""

def read_text(path: str) -> str:
    for enc in ("utf-8","latin-1"):
        try:
            with open(path,"r",encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""

def load_corpus(slides_dir: str):
    texts = []
    if not os.path.isdir(slides_dir): return texts
    base = pathlib.Path(slides_dir)
    for p in base.rglob("*"):
        if not p.is_file(): continue
        ext = p.suffix.lower()
        if ext == ".pdf":
            t = read_pdf(str(p))
        elif ext in {".txt",".md",".mdx",".html",".htm"}:
            t = read_text(str(p))
        else:
            t = ""
        if t and len(t.strip()) > 20:
            texts.append(t)
    return texts

# ------------------ Topic recognition ------------------
def classify_topic(prompt: str) -> str:
    p = (prompt or "").lower()
    if "organelle" in p or "organelles" in p: return "organelle function"
    if "replicat" in p: return "dna replication"
    if "transcription" in p: return "transcription"
    if "translation" in p: return "translation"
    if "glycolysis" in p: return "glycolysis"
    if "membrane" in p or "transport" in p: return "membrane transport"
    if "protein sorting" in p or "signal sequence" in p or "er signal" in p: return "protein sorting"
    if "cell cycle" in p or "mitosis" in p: return "cell cycle"
    return (p.split(",")[0].split(";")[0] or "this topic").strip()

# ------------------ FITB helpers ------------------
STOP = {"the","and","for","that","with","this","from","into","are","was","were","has","have","had","can","will","would","could","should",
        "a","an","of","in","on","to","by","as","at","or","be","is","it","its","their","our","your","if","when","then","than","but",
        "we","you","they","which","these","those","there","here","such","may","might","also","very","much","many","most","more","less"}
META = {"lecture","slide","slides","figure","fig","table","exam","objective","objectives","learning","homework","quiz","next","previous",
        "today","describe","how","identify","define","overview","summary"}

def tokenize(s: str):
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())

def tokens_nostop(s: str):
    return [t for t in tokenize(s) if t not in STOP and len(t) > 2]

def split_sentences(text: str):
    parts = re.split(r"(?<=[\\.\\!\\?])\\s+|\\n+", text)
    return [re.sub(r"\\s+"," ",p).strip() for p in parts if p and len(p.strip()) > 30]

def relevance(sent: str, q_tokens):
    bag = {}
    for tk in tokenize(sent):
        bag[tk] = bag.get(tk,0)+1
    score = sum(bag.get(q,0) for q in q_tokens)
    s_low = sent.lower()
    if len(q_tokens) >= 2 and " ".join(q_tokens[:2]) in s_low:
        score += 2
    return score

def collect_prompt_matched(corpus, prompt: str, top_docs=6, max_sents=800):
    q = tokens_nostop(prompt)
    if not q: return []
    doc_scores = []
    for doc in corpus:
        sents = split_sentences(doc)
        sc = sum(relevance(s, q) for s in sents[:max_sents])
        if sc > 0:
            doc_scores.append((sc, sents))
    doc_scores.sort(reverse=True, key=lambda x: x[0])
    matched = []
    for _, sents in doc_scores[:top_docs]:
        for s in sents:
            if relevance(s, q) > 0:
                matched.append(s)
    return matched

# Acceptance categories
UP = {"increase","increases","increased","up","higher","stabilizes","stabilize","stabilized","faster","more","↑","improves","greater","accumulate","accumulates","builds up","build up"}
DOWN = {"decrease","decreases","decreased","down","lower","destabilizes","destabilize","destabilized","slower","less","↓","reduces","reduced","loss"}
NOCH = {"no change","unchanged","same","neutral","nc","~"}
TRUNC = {"truncated","shorter","premature stop","nonsense","short","truncation"}
MISLOC = {"mislocalized","wrong location","fails to localize","mislocalization","not targeted"}
NOINIT = {"no initiation","fails to initiate","cannot start","no start","initiation blocked","no transcription","no translation"}
NOELON = {"no elongation","elongation blocked","stalled elongation","cannot elongate"}
NOTERM = {"no termination","termination blocked","fails to terminate","readthrough","no stop"}
FRAME = {"frameshift","shifted frame","reading frame shift","out of frame"}

def norm(s: str) -> str:
    s=(s or "").strip().lower()
    s=re.sub(r"\\s+"," ",s).replace("’","'")
    return s

def matches(user: str, key: str) -> bool:
    u = norm(user)
    if key == "increase":   return any(x in u for x in UP)
    if key == "decrease":   return any(x in u for x in DOWN)
    if key == "no_change":  return any(x in u for x in NOCH)
    if key == "truncated":  return any(x in u for x in TRUNC)
    if key == "mislocalized": return any(x in u for x in MISLOC)
    if key == "no_initiation": return any(x in u for x in NOINIT)
    if key == "no_elongation": return any(x in u for x in NOELON)
    if key == "no_termination": return any(x in u for x in NOTERM)
    if key == "frameshift": return any(x in u for x in FRAME)
    return False

# FITB stems
def extract_focus(stem: str) -> str:
    s = stem.lower()
    m = re.search(r"if ([^,]+?) (is|cannot|can't|does not|doesn't|fails|fail)", s)
    if m: return m.group(1).strip()
    m = re.search(r"without ([^,]+?)[, ]", s)
    if m: return m.group(1).strip()
    return ""

def fallback_hint_fitb(stem: str, key: str, topic: str, rng: random.Random) -> str:
    focus = extract_focus(stem) or "that step"
    t = (topic or "this topic").lower()
    H = {
        "increase": [
            f"What product rises first if {focus} speeds up?",
            f"Which immediate output would go up with faster {focus}?",
            f"What accumulates right after {focus} accelerates?",
        ],
        "decrease": [
            f"What drops first if {focus} is blocked?",
            f"Which product would fall immediately with loss of {focus}?",
            f"What output becomes limiting when {focus} slows?",
        ],
        "no_change": [
            f"Would the immediate output actually change if {focus} is altered? Why or why not?",
            f"Does {focus} directly control the next product here?",
            f"Is this step upstream or downstream of the measured output?",
        ],
        "mislocalized": [
            f"Where would the product go without correct targeting at {focus}?",
            f"What compartment would the protein reach if {focus} is missing?",
            f"Which location is lost when {focus} fails?",
        ],
        "no_initiation": [
            f"What is the very first event that fails when {focus} is absent?",
            f"What never starts if {focus} cannot occur?",
            f"What initial assembly is missing without {focus}?",
        ],
        "no_elongation": [
            f"What step at the machine stalls when {focus} cannot proceed?",
            f"What movement along the template stops if {focus} fails?",
            f"What chain-building step halts when {focus} is blocked?",
        ],
        "no_termination": [
            f"What signal fails to be recognized at the end when {focus} is missing?",
            f"What release event won’t happen if {focus} is absent?",
            f"What keeps going past the endpoint when {focus} fails?",
        ],
        "truncated": [
            f"What would make the product end prematurely if {focus} occurs?",
            f"What feature shortens the product when {focus} happens?",
            f"Why would the product stop early with {focus}?",
        ],
        "frameshift": [
            f"Which reading position changes when {focus} occurs?",
            f"What happens to downstream codons after {focus}?",
            f"How would the grouping of symbols change if {focus} happens?",
        ],
    }
    if "transcription" in t:
        H["no_initiation"].append(f"What fails at the promoter when {focus} is absent?")
        H["decrease"].append(f"Which RNA product falls first if {focus} is blocked?")
    if "translation" in t:
        H["no_elongation"].append(f"What happens at the ribosome when {focus} stalls?")
        H["no_termination"].append(f"What occurs to the polypeptide if {focus} is missing?")
    if "replication" in t:
        H["decrease"].append(f"What happens to fork progression if {focus} is lost?")
        H["no_initiation"].append(f"What starting event at origins fails without {focus}?")
    if "glycolysis" in t:
        H["increase"].append(f"Which metabolite builds up first if {focus} speeds up?")
        H["decrease"].append(f"Which product falls first when {focus} is inhibited?")

    bank = H.get(key, [f"Focus on the immediate effect of {focus}."])
    return rng.choice(bank)

def fitb_stems_from_terms(topic: str, rng: random.Random):
    stems = []
    def add_blockage():
        stems.append((f"If a key step in {topic} is blocked, the immediate output would ______.", "decrease"))
    def add_acceleration():
        stems.append((f"If a rate‑limiting step in {topic} speeds up, the near‑term output would ______.", "increase"))
    def add_removal():
        if "transcription" in topic:
            stems.append(("If capping fails, transcript stability would ______.", "decrease"))
        elif "translation" in topic:
            stems.append(("If the ribosome cannot move forward, elongation would ______.", "no_elongation"))
        elif "replication" in topic:
            stems.append(("If helicase does not load, fork progression would ______.", "decrease"))
        elif "glycolysis" in topic:
            stems.append(("If NAD+ is unavailable, glyceraldehyde‑3‑phosphate would ______.", "increase"))
        elif "membrane transport" in topic:
            stems.append(("If an ATP‑driven pump stops, the gradient would ______.", "decrease"))
        elif "protein sorting" in topic:
            stems.append(("If a signal sequence is missing, the protein would be ______.", "mislocalized"))
        elif "cell cycle" in topic:
            stems.append(("If the spindle checkpoint stays active, anaphase onset would ______.", "decrease"))
        elif "organelle" in topic:
            stems.append(("If mitochondrial function is inhibited, cellular ATP levels would ______.", "decrease"))
    def add_rescue():
        stems.append((f"If an upstream block in {topic} is bypassed, the downstream output would ______.", "increase"))
    add_blockage(); add_acceleration(); add_removal(); add_rescue()
    random.shuffle(stems)
    stems = stems[:4]
    out = []
    for s,k in stems:
        out.append({"stem": s, "key": k})
    return out

# ------------------ DnD generators ------------------
def organelle_pairs(rng: random.Random) -> Tuple[str, list, list, dict, dict]:
    """Return instruction, labels, terms, answer_map, hint_map for organelle matching."""
    bank = [
        ("Nucleus","DNA is housed; transcription occurs"),
        ("Mitochondrion","Most ATP is made via oxidative phosphorylation"),
        ("Rough ER","Ribosome‑bound synthesis of secreted/membrane proteins"),
        ("Smooth ER","Lipid synthesis and detox; Ca²⁺ storage"),
        ("Golgi apparatus","Proteins are modified and sorted (e.g., glycosylation)"),
        ("Lysosome","Macromolecules are degraded by acid hydrolases"),
        ("Peroxisome","Peroxide detox and very‑long‑chain fatty acid oxidation"),
        ("Chloroplast","Light‑driven sugar production (photosynthesis)"),
    ]
    rng.shuffle(bank)
    pairs = bank[:rng.choice([3,4])]
    labels = [org for org,_ in pairs]
    terms  = [fn for _,fn in pairs]
    answer = {fn: org for (org,fn) in pairs}
    # targeted fallback hints per item
    hint_map = {}
    for org, fn in pairs:
        if org == "Mitochondrion":
            hint_map[fn] = "Which organelle has cristae and its own DNA, central to aerobic ATP production?"
        elif org == "Nucleus":
            hint_map[fn] = "Which compartment contains chromatin and nucleolus and is bounded by a double membrane?"
        elif org == "Rough ER":
            hint_map[fn] = "Which organelle is studded with ribosomes and feeds proteins into the secretory pathway?"
        elif org == "Smooth ER":
            hint_map[fn] = "Which organelle lacks ribosomes and is key for lipid synthesis and detoxification?"
        elif org == "Golgi apparatus":
            hint_map[fn] = "Which organelle modifies and sorts proteins in cisternae before shipping them?"
        elif org == "Lysosome":
            hint_map[fn] = "Which acidic compartment contains hydrolases for breakdown of macromolecules?"
        elif org == "Peroxisome":
            hint_map[fn] = "Which organelle handles H₂O₂ detox and very‑long‑chain fatty acids?"
        elif org == "Chloroplast":
            hint_map[fn] = "Which plant organelle with thylakoids captures light energy to make sugars?"
        else:
            hint_map[fn] = "Match the function to the specific organelle known for it."
    instr = "Match each **function** to its **organelle**."
    return instr, labels, terms, answer, hint_map

def order_pairs(topic: str) -> Tuple[str, list, list, dict, dict]:
    """Step ordering with numbered bins."""
    t = topic.lower()
    if "replication" in t:
        steps = [
            "Helicase unwinds DNA at origin",
            "Primase lays RNA primers",
            "DNA polymerase extends new strands",
            "Ligase seals nicks to finish"
        ]
        title = "Put the **DNA replication** steps in order."
    elif "transcription" in t:
        steps = [
            "RNA polymerase binds promoter",
            "RNA synthesis begins (initiation)",
            "RNA chain extends (elongation)",
            "Termination releases RNA"
        ]
        title = "Put the **transcription** steps in order."
    elif "translation" in t:
        steps = [
            "Ribosome assembles at start codon",
            "Initiator tRNA occupies P site",
            "Peptide bonds form (elongation)",
            "Stop codon triggers release"
        ]
        title = "Put the **translation** steps in order."
    elif "glycolysis" in t:
        steps = [
            "Glucose is phosphorylated",
            "PFK‑1 commits pathway (F1,6BP forms)",
            "ATP and NADH are generated",
            "Pyruvate is produced"
        ]
        title = "Put the **glycolysis** steps in order."
    else:
        steps = [
            "Process begins",
            "Key intermediate forms",
            "Main product accumulates",
            "Process completes"
        ]
        title = f"Put the **{topic}** steps in order."
    # Bins Step 1..N
    k = len(steps)
    labels = [f"Step {i}" for i in range(1, k+1)]
    terms = steps[:]  # students place steps into numbered bins
    answer = {steps[i]: f"Step {i+1}" for i in range(k)}
    hint_map = {}
    for i, s in enumerate(steps):
        if i == 0:
            hint_map[s] = "Which step happens first before any downstream components can act?"
        elif i == k-1:
            hint_map[s] = "Which step can only happen after everything else is complete?"
        else:
            hint_map[s] = "Which step requires the product of the previous step but precedes the next one?"
    return title, labels, terms, answer, hint_map

def protein_function_pairs(topic: str) -> Tuple[str, list, list, dict, dict]:
    """Protein → function matching for core topics."""
    t = topic.lower()
    if "replication" in t:
        pairs = [
            ("Helicase","Unwinds parental DNA strands"),
            ("Primase","Synthesizes short RNA primers"),
            ("DNA polymerase","Extends DNA from primers"),
            ("DNA ligase","Seals nicks between fragments"),
        ]
        title = "Match each **protein** to its **function** (DNA replication)."
    elif "transcription" in t:
        pairs = [
            ("RNA polymerase II","Synthesizes mRNA from DNA template"),
            ("General TFs","Help polymerase bind promoter/start"),
            ("Spliceosome","Removes introns from pre‑mRNA"),
            ("Capping enzymes","Add 5′ cap to pre‑mRNA"),
        ]
        title = "Match each **protein/complex** to its **function** (transcription)."
    elif "translation" in t:
        pairs = [
            ("Ribosome P site","Holds peptidyl‑tRNA"),
            ("Ribosome A site","Accepts aminoacyl‑tRNA"),
            ("Peptidyl transferase center","Forms peptide bonds"),
            ("Release factor","Recognizes stop codon; terminates"),
        ]
        title = "Match each **component** to its **function** (translation)."
    elif "membrane transport" in t:
        pairs = [
            ("ATP‑driven pump","Moves solutes against gradient using ATP"),
            ("Channel protein","Allows passive ion flow down gradient"),
            ("Carrier (uniporter)","Facilitates diffusion of one solute"),
            ("Symporter","Cotransports two solutes in same direction"),
        ]
        title = "Match each **transport protein** to its **function**."
    else:
        pairs = [
            ("Enzyme","Catalyzes a specific reaction"),
            ("Receptor","Binds ligand to start a signal"),
            ("Channel","Allows ions to pass"),
            ("Motor protein","Generates movement"),
        ]
        title = "Match each **protein type** to its **function**."
    # select 3–4 pairs for clarity
    rng = random.Random(new_seed())
    rng.shuffle(pairs)
    pairs = pairs[:rng.choice([3,4])]
    labels = [p for p,_ in pairs]
    terms = [f for _,f in pairs]
    answer = {f: p for (p,f) in pairs}
    # hints
    hint_map = {}
    for p,f in pairs:
        if p == "Helicase":
            hint_map[f] = "Which protein breaks hydrogen bonds at the fork?"
        elif p == "Primase":
            hint_map[f] = "Which enzyme lays down short RNA to begin synthesis?"
        elif p == "DNA polymerase":
            hint_map[f] = "Which enzyme extends DNA but needs a primer?"
        elif p == "DNA ligase":
            hint_map[f] = "Which enzyme forms phosphodiester bonds to seal nicks?"
        elif p == "RNA polymerase II":
            hint_map[f] = "Which polymerase makes mRNA in eukaryotes?"
        elif p == "General TFs":
            hint_map[f] = "Which factors position polymerase at the promoter?"
        elif p == "Spliceosome":
            hint_map[f] = "Which complex recognizes intron boundaries?"
        elif p == "Capping enzymes":
            hint_map[f] = "Which enzymes add a protective 5′ modification?"
        elif p == "Ribosome P site":
            hint_map[f] = "Which ribosomal site holds the growing chain?"
        elif p == "Ribosome A site":
            hint_map[f] = "Which site accepts the incoming charged tRNA?"
        elif p == "Peptidyl transferase center":
            hint_map[f] = "Which catalytic center forms the bond between amino acids?"
        elif p == "Release factor":
            hint_map[f] = "Which factor recognizes a stop codon to end synthesis?"
        elif p == "ATP‑driven pump":
            hint_map[f] = "Which transporter consumes ATP to move solutes uphill?"
        elif p == "Channel protein":
            hint_map[f] = "Which pathway allows ions to flow without binding/transport cycles?"
        elif p == "Carrier (uniporter)":
            hint_map[f] = "Which transporter binds a single solute and flips conformation?"
        elif p == "Symporter":
            hint_map[f] = "Which transporter couples two solutes in the same direction?"
        else:
            hint_map[f] = "Focus on the defining feature of this protein's role."
    return title, labels, terms, answer, hint_map

def build_dnd_activity(topic: str) -> Tuple[str, list, list, dict, dict]:
    rng = random.Random(new_seed())
    t = topic.lower()
    if "organelle" in t:
        return organelle_pairs(rng)
    # For core processes, choose an effective activity
    if any(x in t for x in ["replication","transcription","translation","glycolysis","membrane transport"]):
        mode = rng.choice(["order","match"])
        if mode == "order":
            return order_pairs(topic)
        else:
            return protein_function_pairs(topic)
    # Default to protein-function matching
    return protein_function_pairs(topic)

# ------------------ UI ------------------
st.title("Let's Practice Biology!")
prompt = st.text_input(
    "Enter a topic for review and press generate:",
    value="",
    placeholder="e.g., organelle function, transcription, glycolysis, protein sorting…",
    label_visibility="visible",
)

if st.button("Generate"):
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)
    topic = classify_topic(prompt) or "this topic"
    st.session_state.topic = topic

    # Build Drag-and-Drop first (Activity 1)
    instr, labels, terms, answer, hint_map = build_dnd_activity(topic)
    st.session_state.dnd_instr = instr
    st.session_state.drag_labels = labels
    st.session_state.drag_bank   = terms
    st.session_state.drag_answer = answer
    st.session_state.dnd_hints   = hint_map

    # Build FITB (Activity 2)
    rng = random.Random(new_seed())
    st.session_state.fitb = fitb_stems_from_terms(topic, rng)

    st.success("Generated fresh activities.")

# -------- Activity 1: Drag-and-Drop --------
if all(k in st.session_state for k in ["drag_labels","drag_bank","drag_answer","dnd_instr","dnd_hints"]):
    st.markdown("## Activity 1: Drag and Drop")
    topic = st.session_state.get("topic","this topic")
    labels = st.session_state.drag_labels
    terms  = st.session_state.drag_bank
    answer = st.session_state.drag_answer
    hint_map = st.session_state.dnd_hints
    st.markdown(f"{st.session_state.dnd_instr}")

    items_html = "".join([f'<li class="card" draggable="true">{t}</li>' for t in terms])
    cols_count = (len(labels)+1)//2 if len(labels) > 2 else 2
    bins_html = "".join([
        f"""
        <div class="bin">
          <div class="title">{lbl}</div>
          <ul id="bin_{i}" class="droplist"></ul>
        </div>
        """ for i,lbl in enumerate(labels)
    ])

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"></script>
      <style>
        * {{ box-sizing: border-box; }}
        body {{ -webkit-user-select: none; -moz-user-select: none; user-select: none; }}
        .bank, .bin {{
          border: 2px dashed #bbb; border-radius: 10px; padding: 12px; min-height: 120px;
          background: #fafafa; margin-bottom: 14px;
        }}
        .bin {{ background: #f6faff; }}
        .droplist {{ list-style: none; margin: 0; padding: 0; min-height: 90px; }}
        .card {{
          background: white; border: 1px solid #ddd; border-radius: 8px;
          padding: 8px 10px; margin: 6px 0; cursor: grab;
          box-shadow: 0 1px 2px rgba(0,0,0,0.06);
          -webkit-user-drag: element;
        }}
        .card:active {{ cursor: grabbing; }}
        .ghost {{ opacity: 0.5; }}
        .chosen {{ outline: 2px solid #7aa2f7; }}
        .zone {{ display:flex; gap:16px; }}
        .left {{ flex: 1; }}
        .right {{ flex: 2; display:grid; grid-template-columns: repeat({cols_count}, 1fr); gap:16px; }}
        .title {{ font-weight: 600; margin-bottom: 6px; user-select: none; -webkit-user-select: none; }}
        .ok   {{ color:#0a7; font-weight:600; }}
        .bad  {{ color:#b00; font-weight:600; }}
        button {{
          border-radius: 8px; border: 1px solid #ddd; background:#fff; padding:8px 12px; cursor:pointer;
        }}
      </style>
    </head>
    <body>
      <div class="zone">
        <div class="left">
          <div class="title">Bank</div>
          <ul id="bank" class="bank droplist">{items_html}</ul>
        </div>
        <div class="right">{bins_html}</div>
      </div>
      <div style="margin-top:10px;">
        <button id="check">Check bins</button>
        <span id="score" style="margin-left:10px;"></span>
      </div>

      <script>
        const LABELS = {json.dumps(labels)};
        const ANSWERS = {json.dumps(answer)};

        const opts = {{
          group: {{ name: 'bins', pull: true, put: true }},
          animation: 150,
          forceFallback: true,
          fallbackOnBody: true,
          swapThreshold: 0.65,
          emptyInsertThreshold: 8,
          ghostClass: 'ghost',
          chosenClass: 'chosen',
        }};

        const lists = [document.getElementById('bank')];
        LABELS.forEach((_, i) => lists.push(document.getElementById('bin_'+i)));
        lists.forEach(el => new Sortable(el, opts));

        function readBins() {{
          const bins = {{}};
          LABELS.forEach((label, i) => {{
            const ul = document.getElementById('bin_'+i);
            const items = Array.from(ul.querySelectorAll('.card')).map(li => li.textContent.trim());
            bins[label] = items;
          }});
          return bins;
        }}

        document.getElementById('check').addEventListener('click', () => {{
          const bins = readBins();
          let total = 0, correct = 0;
          for (const [term, want] of Object.entries(ANSWERS)) {{
            total += 1;
            let got = "Bank";
            for (const [label, items] of Object.entries(bins)) {{
              if (items.includes(term)) {{ got = label; break; }}
            }}
            if (got === want) correct += 1;
          }}
          const score = document.getElementById('score');
          if (total === 0) {{
            score.innerHTML = "<span class='bad'>Drag items into bins first.</span>";
          }} else if (correct === total) {{
            score.innerHTML = "<span class='ok'>All bins correct! 🎉</span>";
          }} else {{
            score.innerHTML = "<span class='bad'>" + correct + "/" + total + " correct — try adjusting and re-check.</span>";
          }}
        }});
      </script>
    </body>
    </html>
    """
    st.components.v1.html(html, height=650, scrolling=True)

    st.caption("Need a hint for a specific item?")
    chosen_item = st.selectbox("Pick an item for a hint:", ["(choose an item)"] + terms, index=0, key="dnd_hint_select")
    if chosen_item != "(choose an item)":
        target_bin = answer.get(chosen_item, "the correct category")
        fb = hint_map.get(chosen_item, "Focus on the most distinctive clue in this item.")
        st.info(llm_hint_for_dnd(chosen_item, target_bin, labels, topic, fb))

# -------- Activity 2: FITB --------
def fitb_unique_hints(stem: str, key: str, topic: str, rng: random.Random) -> str:
    return fallback_hint_fitb(stem, key, topic, rng)

if "fitb" in st.session_state:
    st.markdown("---")
    topic_name = st.session_state.get("topic","this topic")
    st.markdown("## Activity 2: Fill in the Blank")
    st.markdown(f"Use your knowledge of **{topic_name}** to answer the following.")

    rng = random.Random(new_seed())
    for idx, item in enumerate(st.session_state.fitb):
        u = st.text_input(item["stem"], key=f"fitb_{idx}")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Hint", key=f"hint_{idx}"):
                st.info(llm_hint_for_fitb(item["stem"], item["key"], topic_name, fitb_unique_hints(item["stem"], item["key"], topic_name, rng)))
        with col2:
            if st.button("Check", key=f"check_{idx}"):
                ok = matches(u, item["key"])
                if ok:
                    st.success("That’s right! Great work!")
                else:
                    st.info(llm_hint_for_fitb(item["stem"], item["key"], topic_name, fitb_unique_hints(item["stem"], item["key"], topic_name, rng)))
        with col3:
            if st.button("Reveal", key=f"rev_{idx}"):
                pretty = {
                    "increase":"increase",
                    "decrease":"decrease",
                    "no_change":"no change",
                    "truncated":"truncated",
                    "mislocalized":"mislocalized",
                    "no_initiation":"no initiation",
                    "no_elongation":"no elongation",
                    "no_termination":"no termination",
                    "frameshift":"frameshift",
                }.get(item["key"], item["key"])
                st.info(pretty)
