
import os, re, json, pathlib, random, time, sys, io, unicodedata, requests
import streamlit as st

# ------------------ Utilities: JSON sanitize & prompt expansion ------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_json_block(s: str) -> str:
    s = _strip_code_fences(s)
    # Try array first
    a_start = s.find("[")
    a_end   = s.rfind("]")
    o_start = s.find("{")
    o_end   = s.rfind("}")
    cand = ""
    if a_start != -1 and a_end != -1 and a_end > a_start:
        cand = s[a_start:a_end+1]
        try:
            json.loads(cand); return cand
        except Exception: pass
    if o_start != -1 and o_end != -1 and o_end > o_start:
        cand = s[o_start:o_end+1]
        try:
            json.loads(cand); return cand
        except Exception: pass
    return s  # last resort

BOND_KEYWORDS = {
    "chemical": ["covalent", "ionic", "hydrogen bond", "van der waals", "noncovalent", "polar", "nonpolar", "bond energy", "electronegativity"],
    "junctions": ["tight junction", "adherens", "desmosome", "hemidesmosome", "gap junction", "cell-cell adhesion", "cadherin", "integrin", "extracellular matrix"],
    "cytoskeleton": ["microtubule", "actin", "intermediate filament", "cytoskeleton"]
}

def expand_prompt_keywords(p: str) -> list:
    p = (p or "").lower()
    base = re.findall(r"[a-z0-9']+", p)
    expanded = set(base)
    if "bond" in p or "bonds" in p:
        for group in BOND_KEYWORDS.values():
            expanded.update([w for w in " ".join(group).split() if len(w) > 2])
    if "transport" in p:
        expanded.update(["diffusion","channel","pump","carrier","symporter","antiporter","uniporter","gradient","atp"])
    if "glycolysis" in p:
        expanded.update(["pfk","pyruvate","nad","atp","enzyme","kinase"])
    if "replication" in p:
        expanded.update(["helicase","primase","polymerase","ligase","fork"])
    if "transcription" in p:
        expanded.update(["promoter","rna","pol","splice","cap","terminate"])
    if "translation" in p:
        expanded.update(["ribosome","a site","p site","release","trna","start codon","stop codon"])
    return [w for w in expanded if len(w) > 2]


from typing import List, Tuple, Dict, Any

# ------------------ Page & Globals (unchanged UX) ------------------
st.set_page_config(page_title="Let's Practice Biology!", page_icon="🎓", layout="wide")
st.title("Let's Practice Biology!")
SLIDES_GH_USER   = "mwellis825"   # <-- set once here
SLIDES_GH_REPO   = "cell-bio-gpt-tutor"         # <-- set once here
SLIDES_GH_BRANCH = "main"                    # <-- set once here
SLIDES_DIR_PATH  = "slides"                  # fixed folder in the repo

# Helper for consistent randomness as before
def new_seed() -> int:
    return int(time.time() * 1000) ^ random.randint(0, 1_000_000)

# ------------------ Optional LLM for hints (kept from current behavior) ------------------
def _try_llm(prompt: str) -> str:
    api = os.environ.get("OPENAI_API_KEY")
    if not api:
        raise RuntimeError("No API key")
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=api)
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"),
        messages=[
            {"role":"system","content":"You are a Socratic college-level cell biology tutor. Ask ONE concise guiding question (< 20 words) that nudges the student to the answer without revealing it."},
            {"role":"user","content":prompt},
        ],
        temperature=0.3,
        max_tokens=60,
        seed=42,
    )
    return resp.choices[0].message.content.strip()

def llm_hint_for_fitb(stem: str, target: str, topic: str, fallback: str) -> str:
    try:
        return _try_llm(f"Topic: {topic}\nFITB stem: {stem}\nExpected answer (or label): {target}\nWrite one guiding, specific question.")
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

# ------------------ PDF/Text extraction (compatible with current app) ------------------
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

def read_pdf_bytes(pdf_bytes: bytes) -> str:
    if PDF_BACKEND == "PyPDF2":
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))  # type: ignore
            return "\\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""
    if PDF_BACKEND == "pypdf":
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))  # type: ignore
            return "\\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""
    return ""

def read_text_bytes(raw: bytes) -> str:
    # Try utf-8, fallback latin-1
    for enc in ("utf-8","latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return ""

# ------------------ GitHub fetchers (new, replaces local /slides reading) ------------------
def gh_list_slides(user: str, repo: str, path: str, branch: str) -> List[Dict[str, Any]]:
    """
    Return a list of files in /slides via GitHub Contents API.
    """
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{path}?ref={branch}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json() if isinstance(r.json(), list) else []

def gh_fetch_raw(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

@st.cache_data(show_spinner=False)
def load_corpus_from_github(user: str, repo: str, path: str, branch: str):
    corpus = []
    try:
        items = gh_list_slides(user, repo, path, branch)
    except Exception as e:
        st.error(f"Could not list slides from GitHub: {e}")
        return corpus

    for it in items:
        if it.get("type") != "file":
            continue
        name = (it.get("name") or "").lower()
        raw_url = it.get("download_url")
        if not raw_url:
            # try to build raw URL
            raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}/{it.get('name')}"
        try:
            if name.endswith(".pdf"):
                txt = read_pdf_bytes(gh_fetch_raw(raw_url))
            elif name.endswith((".txt",".md",".mdx",".html",".htm")):
                txt = read_text_bytes(gh_fetch_raw(raw_url))
            else:
                txt = ""
        except Exception:
            txt = ""
        if txt and len(txt.strip()) > 20:
            corpus.append(txt)
    return corpus

# ------------------ Simple retrieval & sentence matching (kept but strengthened) ------------------
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
    return matched[:200]

# ------------------ Topic recognition (unchanged) ------------------
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

# ------------------ Acceptance logic (unchanged) ------------------
UP = {"increase","increases","increased","up","higher","stabilizes","stabilize","stabilized","faster","more","↑","improves","greater","accumulate","accumulates","builds up","build up"}
DOWN = {"decrease","decreases","decreased","down","lower","destabilizes","destabilize","destabilized","slower","less","↓","reduces","reduced","loss"}
NOCH = {"no change","unchanged","same","neutral","nc","~"}
TRUNC = {"truncated","shorter","premature stop","nonsense","short","truncation"}
MISLOC = {"mislocalized","wrong location","fails to localize","mislocalization","not targeted"}
NOINIT = {"no initiation","fails to initiate","cannot start","no start","initiation blocked","no transcription","no translation"}
NOELON = {"no elongation","elongation blocked","stalled elongation","cannot elongate"}
NOTERM = {"no termination","termination blocked","fails to terminate","readthrough","no stop"}
FRAME = {"frameshift","shifted frame","reading frame shift","out of frame"}

def norm_text(s: str) -> str:
    s=(s or "").strip().lower()
    s=re.sub(r"[^a-z0-9\s\-\+]", "", s)
    s=re.sub(r"\s+", " ", s)
    return s

def matches_label(user: str, key: str) -> bool:
    u = norm_text(user)
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

SYNONYMS = {
    "endoplasmic reticulum": {"er","endoplasmic reticulum","rough er","smooth er"},
    "rough endoplasmic reticulum": {"rough er","rer","rough endoplasmic reticulum"},
    "smooth endoplasmic reticulum": {"smooth er","ser","smooth endoplasmic reticulum"},
    "mitochondrial matrix": {"mitochondrial matrix","matrix (mitochondrion)","matrix"},
    "cytosol": {"cytosol","cytoplasm"},
    "nucleus": {"nucleus","nuclear compartment"},
    "golgi apparatus": {"golgi","golgi apparatus"},
    "lysosome": {"lysosome"},
    "peroxisome": {"peroxisome"},
    "chloroplast": {"chloroplast"},
    "rna polymerase ii": {"rna polymerase ii","pol ii","rna pol ii","rna polymerase"},
    "helicase": {"helicase"},
    "primase": {"primase"},
    "dna polymerase": {"dna polymerase"},
    "dna ligase": {"dna ligase","ligase"},
    "ribosome p site": {"p site","ribosome p site"},
    "ribosome a site": {"a site","ribosome a site"},
    "peptidyl transferase center": {"peptidyl transferase center","ptc"},
    "release factor": {"release factor","rf"},
    "pyruvate": {"pyruvate","pyruvic acid"},
    "spontaneous": {"spontaneous","passive"},
    "energy-dependent": {"energy-dependent","active","requires atp","active transport"},
    "glycolysis location": {"cytosol","cytoplasm"},
}

SPECIFIC_HINTS = {
    "mitochondrial matrix": "Innermost mitochondrial compartment (inside inner membrane) where TCA enzymes reside.",
    "endoplasmic reticulum": "Organelle contiguous with the nuclear envelope; entry point for secretory proteins.",
    "rough endoplasmic reticulum": "Membrane with bound ribosomes; co‑translational entry for secreted/membrane proteins.",
    "smooth endoplasmic reticulum": "Ribosome‑free ER region; lipid synthesis and detoxification.",
    "nucleus": "Double membrane with pores; houses DNA and nucleolus.",
    "cytosol": "Aqueous compartment outside organelles; most glycolysis steps happen here.",
    "golgi apparatus": "Stacked cisternae that modify and sort proteins after ER.",
    "lysosome": "Acidic lumen with hydrolases for macromolecule breakdown.",
    "peroxisome": "Oxidative organelle; detoxifies H2O2 and oxidizes very‑long‑chain fatty acids.",
    "chloroplast": "Plant organelle with thylakoids; performs photosynthesis.",
    "rna polymerase ii": "Eukaryotic polymerase for mRNA; works in the nucleus.",
    "helicase": "Unwinds DNA by breaking hydrogen bonds at the fork.",
    "primase": "Synthesizes short RNA primers to start DNA synthesis.",
    "dna polymerase": "Extends DNA from a primer; requires 3′‑OH.",
    "dna ligase": "Forms phosphodiester bonds to seal nicks.",
    "ribosome p site": "Ribosomal site holding peptidyl‑tRNA (growing chain).",
    "ribosome a site": "Ribosomal site accepting incoming aminoacyl‑tRNA.",
    "peptidyl transferase center": "Catalyzes peptide bond formation in the large subunit.",
    "release factor": "Recognizes stop codon and promotes termination.",
    "pyruvate": "Three‑carbon product that feeds the TCA cycle under aerobic conditions.",
    "spontaneous": "Occurs down a gradient without ATP input.",
    "energy-dependent": "Requires ATP hydrolysis or an equivalent energy source.",
}

def normalize_answer(a: str) -> str:
    a = norm_text(a); return a

def matches_specific(user: str, answers: List[str]) -> bool:
    u = normalize_answer(user)
    for ans in answers:
        norm_ans = normalize_answer(ans)
        if u == norm_ans:
            return True
        if ans in SYNONYMS:
            if u in {normalize_answer(x) for x in SYNONYMS[ans]}:
                return True
    return False

def specific_hint_for_answer(ans: str) -> str:
    a = ans.lower()
    if a in SPECIFIC_HINTS:
        return SPECIFIC_HINTS[a]
    if "matrix" in a and "mitochond" in a:
        return "Innermost mitochondrial space enclosed by the inner membrane."
    if "endoplasmic reticulum" in a or a.endswith(" er"):
        return "Membranous network near nucleus; entry to the secretory pathway."
    if "cytosol" in a or "cytoplasm" in a:
        return "Fluid compartment outside organelles."
    if "nucleus" in a:
        return "Double membrane with pores; DNA location."
    return "Use the most specific biological term for the location or component."

# ------------------ Existing FITB generation (fallback preserved) ------------------
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

def specific_fitb_items(topic: str) -> List[Dict]:
    t = topic.lower()
    items = []
    if "glycolysis" in t:
        items.append({"stem":"Glycolysis primarily occurs in the ______ of eukaryotic cells.","answers":["cytosol","glycolysis location"],"hint":"Think compartment: not membrane‑bound, enzyme‑rich fluid."})
        items.append({"stem":"The end product of glycolysis is ______.","answers":["pyruvate"],"hint":"Three‑carbon product feeding the TCA cycle when oxygen is present."})
    if "transcription" in t:
        items.append({"stem":"Which enzyme synthesizes mRNA in eukaryotes? ______.","answers":["rna polymerase ii"],"hint":"Eukaryotic polymerase for mRNA; often abbreviated Pol II."})
        items.append({"stem":"Transcription in eukaryotes occurs in the ______.","answers":["nucleus"],"hint":"Compartment with nuclear pores and chromatin."})
    if "translation" in t:
        items.append({"stem":"The growing polypeptide is held at the ribosomal ______ site.","answers":["ribosome p site"],"hint":"Not the A site; this one holds the peptidyl‑tRNA."})
        items.append({"stem":"Secretory proteins often begin translation on ribosomes bound to the ______.","answers":["rough endoplasmic reticulum","endoplasmic reticulum"],"hint":"Membrane with ribosomes at the entry of the secretory pathway."})
    if "replication" in t:
        items.append({"stem":"The enzyme that unwinds DNA at the replication fork is ______.","answers":["helicase"],"hint":"Breaks hydrogen bonds between base pairs."})
    if "membrane transport" in t:
        items.append({"stem":"Simple diffusion is ______ (spontaneous or energy‑dependent?).","answers":["spontaneous"],"hint":"Occurs down a gradient without ATP input."})
        items.append({"stem":"Pumps that move solutes up their gradient are ______ (spontaneous or energy‑dependent?).","answers":["energy-dependent"],"hint":"Require ATP directly or indirectly."})
    if "protein sorting" in t:
        items.append({"stem":"Proteins entering the secretory pathway are first threaded into the ______.","answers":["endoplasmic reticulum","rough endoplasmic reticulum"],"hint":"SRP targets ribosomes here for co‑translational import."})
    if "organelle" in t:
        items.append({"stem":"Most cellular ATP is produced in the ______ (be specific).","answers":["mitochondrial matrix"],"hint":"Innermost mitochondrial compartment with TCA enzymes."})
    return items

def reasoning_fitb_items(topic: str) -> List[Dict]:
    stems = []
    stems.append((f"If a key step in {topic} is blocked, the immediate output would ______.", "decrease"))
    stems.append((f"If a rate‑limiting step in {topic} speeds up, the near‑term output would ______.", "increase"))
    tp = topic.lower()
    if "transcription" in tp:
        stems.append(("If capping fails, transcript stability would ______.", "decrease"))
    elif "translation" in tp:
        stems.append(("If the ribosome cannot move forward, elongation would ______.", "no_elongation"))
    elif "replication" in tp:
        stems.append(("If helicase does not load, fork progression would ______.", "decrease"))
    elif "glycolysis" in tp:
        stems.append(("If NAD+ is unavailable, glyceraldehyde‑3‑phosphate would ______.", "increase"))
    elif "membrane transport" in tp:
        stems.append(("If an ATP‑driven pump stops, the gradient would ______.", "decrease"))
    elif "protein sorting" in tp:
        stems.append(("If a signal sequence is missing, the protein would be ______.", "mislocalized"))
    elif "cell cycle" in tp:
        stems.append(("If the spindle checkpoint stays active, anaphase onset would ______.", "decrease"))
    out = [{"stem": s, "label": k} for s,k in stems]
    return out

def build_fitb(topic: str, rng: random.Random) -> List[Dict]:
    items = []
    specific = specific_fitb_items(topic)
    rng.shuffle(specific)
    items.extend(specific[:rng.choice([1,2])])
    reason = reasoning_fitb_items(topic)
    rng.shuffle(reason)
    items.extend(reason[:2])
    rng.shuffle(items)
    return items[:4]

# ------------------ DnD activity generators (fallback preserved) ------------------
def organelle_pairs(rng: random.Random) -> Tuple[str, list, list, dict, dict]:
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
    t = topic.lower()
    if "replication" in t:
        steps = ["Helicase unwinds DNA at origin","Primase lays RNA primers","DNA polymerase extends new strands","Ligase seals nicks to finish"]
        title = "Put the **DNA replication** steps in order."
    elif "transcription" in t:
        steps = ["RNA polymerase binds promoter","RNA synthesis begins (initiation)","RNA chain extends (elongation)","Termination releases RNA"]
        title = "Put the **transcription** steps in order."
    elif "translation" in t:
        steps = ["Ribosome assembles at start codon","Initiator tRNA occupies P site","Peptide bonds form (elongation)","Stop codon triggers release"]
        title = "Put the **translation** steps in order."
    elif "glycolysis" in t:
        steps = ["Glucose is phosphorylated","PFK‑1 commits pathway (F1,6BP forms)","ATP and NADH are generated","Pyruvate is produced"]
        title = "Put the **glycolysis** steps in order."
    else:
        steps = ["Process begins","Key intermediate forms","Main product accumulates","Process completes"]
        title = f"Put the **{topic}** steps in order."
    k = len(steps)
    labels = [f"Step {i}" for i in range(1, k+1)]
    terms = steps[:]
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

def glycolysis_pairs(rng: random.Random) -> Tuple[str, list, list, dict, dict]:
    pairs = [
        ("Hexokinase","Phosphorylates glucose in the first step"),
        ("Phosphofructokinase‑1 (PFK‑1)","Commits pathway by converting F6P to F1,6BP"),
        ("Glyceraldehyde‑3‑phosphate dehydrogenase (GAPDH)","Generates NADH from G3P"),
        ("Pyruvate kinase","Produces ATP while forming pyruvate"),
    ]
    rng.shuffle(pairs)
    pairs = pairs[:rng.choice([3,4])]
    labels = [p for p,_ in pairs]
    terms = [f for _,f in pairs]
    answer = {f: p for (p,f) in pairs}
    hint_map = {}
    for p,f in pairs:
        if "Hexokinase" in p:
            hint_map[f] = "Which enzyme acts first, trapping glucose in the cell as a phosphate ester?"
        elif "PFK‑1" in p:
            hint_map[f] = "Which enzyme controls the commitment step and is allosterically regulated by ATP/AMP?"
        elif "GAPDH" in p:
            hint_map[f] = "Which enzyme uses inorganic phosphate and reduces NAD+ during glycolysis?"
        elif "Pyruvate kinase" in p:
            hint_map[f] = "Which enzyme catalyzes a substrate‑level phosphorylation to make ATP at the end?"
        else:
            hint_map[f] = "Match the function to the enzyme’s key hallmark in glycolysis."
    instr = "Match each **glycolytic enzyme** to its **function**."
    return instr, labels, terms, answer, hint_map

def protein_function_pairs(topic: str) -> Tuple[str, list, list, dict, dict]:
    t = topic.lower()
    if "replication" in t:
        pairs = [("Helicase","Unwinds parental DNA strands"),("Primase","Synthesizes short RNA primers"),("DNA polymerase","Extends DNA from primers"),("DNA ligase","Seals nicks between fragments")]
        title = "Match each **protein** to its **function** (DNA replication)."
    elif "transcription" in t:
        pairs = [("RNA polymerase II","Synthesizes mRNA from DNA template"),("General TFs","Help polymerase bind promoter/start"),("Spliceosome","Removes introns from pre‑mRNA"),("Capping enzymes","Add 5′ cap to pre‑mRNA")]
        title = "Match each **protein/complex** to its **function** (transcription)."
    elif "translation" in t:
        pairs = [("Ribosome P site","Holds peptidyl‑tRNA"),("Ribosome A site","Accepts aminoacyl‑tRNA"),("Peptidyl transferase center","Forms peptide bonds"),("Release factor","Recognizes stop codon; terminates")]
        title = "Match each **component** to its **function** (translation)."
    elif "membrane transport" in t:
        pairs = [("ATP‑driven pump","Moves solutes against gradient using ATP"),("Channel protein","Allows passive ion flow down gradient"),("Carrier (uniporter)","Facilitates diffusion of one solute"),("Symporter","Cotransports two solutes in same direction")]
        title = "Match each **transport protein** to its **function**."
    else:
        pairs = [("Enzyme","Catalyzes a specific reaction"),("Receptor","Binds ligand to start a signal"),("Channel","Allows ions to pass"),("Motor protein","Generates movement")]
        title = "Match each **protein type** to its **function**."
    rng = random.Random(new_seed())
    rng.shuffle(pairs)
    pairs = pairs[:rng.choice([3,4])]
    labels = [p for p,_ in pairs]
    terms = [f for _,f in pairs]
    answer = {f: p for (p,f) in pairs}
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
    if "glycolysis" in t:
        mode = rng.choice(["order","match"])
        if mode == "order":
            return order_pairs(topic)
        else:
            return glycolysis_pairs(rng)
    if any(x in t for x in ["replication","transcription","translation","membrane transport"]):
        mode = rng.choice(["order","match"])
        if mode == "order":
            return order_pairs(topic)
        else:
            return protein_function_pairs(topic)
    return protein_function_pairs(topic)

# ------------------ New: LLM activity generation grounded in slides ------------------
LLM_SYS = "You generate concise interactive activities grounded ONLY in the provided slide excerpts. Never reveal answers. Use simple student-friendly language."

def _llm_json(prompt: str, max_tokens=900, temperature=0.1) -> str:
    api = os.environ.get("OPENAI_API_KEY")
    if not api:
        return ""
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=api)
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"),
        messages=[{"role":"system","content":LLM_SYS},{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        seed=42,
    )
    return resp.choices[0].message.content or ""

def build_scope_from_corpus(corpus: List[str], prompt: str, limit_chars: int = 4000) -> str:
    sents = collect_prompt_matched(corpus, prompt, top_docs=6, max_sents=1200)
    scope = "\\n".join(sents)[:limit_chars]
    return scope

def llm_generate_dnd(scope: str, topic: str) -> Tuple[str, list, list, dict, dict]:
    """
    Ask the LLM to produce bins (labels), draggables (terms), a mapping, and short hints — all derived from scope.
    """
    if not scope.strip():
        return ("", [], [], {}, {})
    schema = {
        "title": "string",
        "bins": ["string"],                # labels shown as bins
        "terms": ["string"],               # draggable items
        "mapping": {"term->bin": "dict"},  # keys: terms, values: bin label
        "hints": {"term->hint": "dict"}    # one short hint per term, non-revealing
    }
    prompt = f"""Create ONE drag-and-drop activity based ONLY on the slide excerpts below.

Slide excerpts:
\"\"\"
{scope}
\"\"\"

Topic: {topic}

Rules:
- Return STRICT JSON only (no markdown), with keys: title, bins, terms, mapping, hints.
- bins: 2-4 concise bin labels.
- terms: 2-4 draggable items, each must match one of the bins via mapping.
- mapping: object where each key is a term and each value is a bin label from bins.
- hints: object where each key is a term and the value is ONE helpful hint that does NOT reveal the bin.
- Use only concepts present in the excerpts.
- Keep the language student-friendly.
- Do NOT include answers in hints.
"""
    raw = _llm_json(prompt)
    if not raw.strip():
        return ("", [], [], {}, {})
    raw = _extract_json_block(raw)
    # Basic JSON parse & validation
    try:
        data = json.loads(raw)
        title = data.get("title") or "Match the concepts to the correct category."
        bins = data.get("bins") or []
        terms = data.get("terms") or []
        mapping = data.get("mapping") or {}
        hints = data.get("hints") or {}
        # minimal sanity
        if not (2 <= len(bins) <= 4 and 2 <= len(terms) <= 4):
            return ("", [], [], {}, {})
        # Ensure mapping and hints entries
        if any(t not in mapping for t in terms):
            return ("", [], [], {}, {})
        for t in terms:
            if mapping[t] not in bins:
                return ("", [], [], {}, {})
        for t in terms:
            if not isinstance(hints.get(t,""), str) or len(hints.get(t,"").strip()) < 3:
                hints[t] = "Consider the defining feature that separates this from other categories."
        # Convert to UI structures used by current app
        instr = "Match each **item** to its **category**."
        labels = bins
        draggables = terms
        answer = {t: mapping[t] for t in terms}  # NOTE: current app expects {term: BinLabel}
        hint_map = hints
        # enforce grounding to prompt keywords
        kws = set(expand_prompt_keywords(topic))
        combined = " ".join([title] + labels + draggables + list(mapping.keys()) + list(mapping.values()) + list(hints.values())).lower()
        hit = any(k in combined for k in kws if len(k) > 3)
        if not hit:
            return ("", [], [], {}, {})
        return instr, labels, draggables, answer, hint_map
    except Exception:
        return ("", [], [], {}, {})

def llm_generate_fitb(scope: str, topic: str) -> List[Dict]:
    """
    Ask the LLM to produce up to 4 FITB items grounded only in scope.
    """
    if not scope.strip():
        return []
    prompt = f"""Create up to 4 fill-in-the-blank items based ONLY on the slide excerpts below.

Slide excerpts:
\"\"\"
{scope}
\"\"\"

Topic: {topic}

Rules:
- Return STRICT JSON array only.
- Each item is an object with: "stem" (string, with one blank as ______), "answers" (1-4 short accepted answers), "hint" (ONE short guiding hint).
- Keep stems concise and assess key ideas explicitly present in the excerpts.
- Do NOT reveal the answer in the hint.
- Prefer specific terms actually present in the excerpts.
"""
    raw = _llm_json(prompt, max_tokens=800, temperature=0.1)
    if not raw.strip():
        return []
    raw = _extract_json_block(raw)
    try:
        items = json.loads(raw)
        out = []
        for it in items[:4]:
            stem = it.get("stem","").strip()
            ans  = it.get("answers",[]) or []
            hint = it.get("hint","").strip() or "Focus on the specific term used in the slides."
            if stem and "______" in stem and 1 <= len(ans) <= 4:
                out.append({"stem":stem, "answers":ans, "hint":hint})
        return out
    except Exception:
        return []

# ------------------ UI Input (unchanged) ------------------
prompt = st.text_input(
    "Enter a topic for review and press generate:",
    value="",
    placeholder="e.g., organelle function, glycolysis, transcription…",
    label_visibility="visible",
)

# ------------------ Generate button (mechanics preserved) ------------------
if st.button("Generate"):
    # Load corpus once from GitHub
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus_from_github(SLIDES_GH_USER, SLIDES_GH_REPO, SLIDES_DIR_PATH, SLIDES_GH_BRANCH)

    topic = classify_topic(prompt) or "this topic"
    # Disambiguate "bonds" prompts based on slide scope
    _scope_for_disambig = build_scope_from_corpus(st.session_state.corpus or [], prompt, limit_chars=6000)
    if re.search(r"\bbond(s)?\b", (prompt or "").lower()):
        topic = disambiguate_bonds(_scope_for_disambig)
    st.session_state.topic = topic

    # -------- NEW: Build a scope from slides and try LLM-grounded activities --------
    scope = build_scope_from_corpus(st.session_state.corpus or [], prompt, limit_chars=5000)

    # Try LLM-based DnD first
    instr, labels, terms, answer, hint_map = llm_generate_dnd(scope, topic)
    if not labels or not terms or not answer:
        # Fallback to previous heuristic generator (keeps current behavior)
        instr, labels, terms, answer, hint_map = build_dnd_activity(topic)

    # Save to session_state (same keys as current app)
    st.session_state.dnd_instr = instr
    st.session_state.drag_labels = labels
    st.session_state.drag_bank   = terms
    st.session_state.drag_answer = answer
    st.session_state.dnd_hints   = hint_map

    # FITB via LLM first
    rng = random.Random(new_seed())
    fitb_items = llm_generate_fitb(scope, topic)
    if not fitb_items:
        fitb_items = build_fitb(topic, rng)  # fallback preserved

    
        # Diagnostics: build keyword hits
        kw = expand_prompt_keywords(prompt)
        scope_hits = sum(scope.lower().count(k) for k in kw)

        # Try LLM-based DnD first
        instr, labels, terms, answer, hint_map = llm_generate_dnd(scope, topic)
        used_llm_dnd = bool(labels and terms and answer)
        fallback_reason_dnd = ""
        if not used_llm_dnd:
            fallback_reason_dnd = "LLM JSON invalid/ungrounded or empty; used heuristic DnD."
            instr, labels, terms, answer, hint_map = build_dnd_activity(topic)

        # Save to session_state
        st.session_state.dnd_instr = instr
        st.session_state.drag_labels = labels
        st.session_state.drag_bank   = terms
        st.session_state.drag_answer = answer
        st.session_state.dnd_hints   = hint_map
        st.session_state.used_llm_dnd = used_llm_dnd
        st.session_state.fallback_reason_dnd = fallback_reason_dnd

        # FITB via LLM first
        rng = random.Random(new_seed())
        fitb_items = llm_generate_fitb(scope, topic)
        # Filter boilerplate stems
        def _boiler(s): return "immediate output would" in s.lower() or "near-term output would" in s.lower()
        fitb_items = [it for it in fitb_items if not _boiler(it.get("stem",""))]
        used_llm_fitb = len(fitb_items) > 0
        fallback_reason_fitb = ""
        if not used_llm_fitb:
            fallback_reason_fitb = "LLM returned no usable FITB (or boilerplate); used heuristic FITB."
            fitb_items = build_fitb(topic, rng)

        st.session_state.fitb = fitb_items
        st.session_state.used_llm_fitb = used_llm_fitb
        st.session_state.fallback_reason_fitb = fallback_reason_fitb

        st.success("Generated fresh activities.")
        with st.expander("Diagnostics"):
            st.write(f"Scope chars: {len(scope)} | keyword hits in scope: {scope_hits}")
            st.write("DnD:", "LLM" if used_llm_dnd else "Fallback", fallback_reason_dnd)
            st.write("FITB:", "LLM" if used_llm_fitb else "Fallback", fallback_reason_fitb)


# -------- Activity 1: Drag-and-Drop (unchanged rendering/UI) --------
if all(k in st.session_state for k in ["drag_labels","drag_bank","drag_answer","dnd_instr","dnd_hints"]):
    st.markdown("## Activity 1: Drag and Drop")
    st.caption("Source: " + ("LLM" if st.session_state.get("used_llm_dnd") else "Fallback"))
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
        body {{ -webkit-user-select: none; -moz-user-select: none; user-select: none; margin:0; padding:0; }}
        .bank, .bin {{
          border: 2px dashed #bbb; border-radius: 10px; padding: 10px; min-height: 110px;
          background: #fafafa; margin-bottom: 8px;
        }}
        .bin {{ background: #f6faff; }}
        .droplist {{ list-style: none; margin: 0; padding: 0; min-height: 80px; }}
        .card {{
          background: white; border: 1px solid #ddd; border-radius: 8px;
          padding: 8px 10px; margin: 6px 0; cursor: grab;
          box-shadow: 0 1px 2px rgba(0,0,0,0.06);
          -webkit-user-drag: element;
        }}
        .card:active {{ cursor: grabbing; }}
        .ghost {{ opacity: 0.5; }}
        .chosen {{ outline: 2px solid #7aa2f7; }}
        .zone {{ display:flex; gap:14px; }}
        .left {{ flex: 1; }}
        .right {{ flex: 2; display:grid; grid-template-columns: repeat({cols_count}, 1fr); gap:14px; }}
        .title {{ font-weight: 600; margin-bottom: 6px; user-select: none; -webkit-user-select: none; }}
        .ok   {{ color:#0a7; font-weight:600; }}
        .bad  {{ color:#b00; font-weight:600; }}
        .controls {{ margin-top: 6px; }}
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
          <div class="controls">
            <button id="check">Check bins</button>
            <span id="score" style="margin-left:10px;"></span>
          </div>
        </div>
        <div class="right">{bins_html}</div>
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
    st.components.v1.html(html, height=560, scrolling=True)
    st.markdown("<div style='margin-top:-6px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1,3])
    with c1:
        chosen_item = st.selectbox("Hint for:", ["(choose…)"] + terms, index=0, key="dnd_hint_select")
    with c2:
        if chosen_item != "(choose…)":
            target_bin = answer.get(chosen_item, "the correct category")
            fb = hint_map.get(chosen_item, "Focus on the most distinctive clue in this item.")
            st.info(llm_hint_for_dnd(chosen_item, target_bin, labels, topic, fb))

# -------- Activity 2: FITB (unchanged rendering/UI) --------
if "fitb" in st.session_state:
    st.markdown("---")
    topic_name = st.session_state.get("topic","this topic")
    st.markdown("## Activity 2: Fill in the Blank")
    st.caption("Source: " + ("LLM" if st.session_state.get("used_llm_fitb") else "Fallback"))
    st.markdown(f"Use your knowledge of **{topic_name}** to answer the following.")

    rng = random.Random(new_seed())
    for idx, item in enumerate(st.session_state.fitb):
        u = st.text_input(item["stem"], key=f"fitb_{idx}")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Hint", key=f"hint_{idx}"):
                if "answers" in item:
                    target = item["answers"][0]
                    fb = specific_hint_for_answer(target)
                    st.info(llm_hint_for_fitb(item["stem"], target, topic_name, fb))
                else:
                    st.info(llm_hint_for_fitb(item["stem"], item["label"], topic_name, fallback_hint_fitb(item["stem"], item["label"], topic_name, rng)))
        with col2:
            if st.button("Check", key=f"check_{idx}"):
                ok = False
                if "answers" in item:
                    ok = matches_specific(u, item["answers"])
                else:
                    ok = matches_label(u, item["label"])
                if ok:
                    st.success("That’s right! Great work!")
                else:
                    if "answers" in item:
                        target = item["answers"][0]
                        fb = specific_hint_for_answer(target)
                        st.info(llm_hint_for_fitb(item["stem"], target, topic_name, fb))
                    else:
                        st.info(llm_hint_for_fitb(item["stem"], item["label"], topic_name, fallback_hint_fitb(item["stem"], item["label"], topic_name, rng)))
        with col3:
            if st.button("Reveal", key=f"rev_{idx}"):
                if "answers" in item:
                    st.info(item["answers"][0])
                else:
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
                    }.get(item["label"], item["label"])
                    st.info(pretty)

def disambiguate_bonds(scope_text: str) -> str:
    """Decide whether 'bonds' likely means chemical bonds or cell junctions based on scope occurrences."""
    s = scope_text.lower()
    chem_hits = sum(s.count(k) for k in BOND_KEYWORDS["chemical"])
    junc_hits = sum(s.count(k) for k in BOND_KEYWORDS["junctions"])
    cyto_hits = sum(s.count(k) for k in BOND_KEYWORDS["cytoskeleton"])
    if max(chem_hits, junc_hits, cyto_hits) == 0:
        return "bonds (unspecified)"
    if chem_hits >= junc_hits and chem_hits >= cyto_hits:
        return "chemical bonds"
    if junc_hits >= chem_hits and junc_hits >= cyto_hits:
        return "cell junctions"
    return "cytoskeleton interactions"