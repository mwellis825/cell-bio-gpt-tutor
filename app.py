
import os, re, json, pathlib, random, time, sys
import streamlit as st
from typing import List, Tuple, Dict

# ------------------ Page & Globals ------------------
st.set_page_config(page_title="Let's Practice Biology!", page_icon="🎓", layout="wide")
SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm"}

def new_seed() -> int:
    return int(time.time() * 1000) ^ random.randint(0, 1_000_000)

# ------------------ Optional LLM (auto) ------------------
def llm_coach_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))

def call_llm(prompt: str) -> str:
    """Optional Socratic coaching if OPENAI_API_KEY is set. App works without it."""
    try:
        import openai  # type: ignore
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        sys_prompt = (
            "You are a gentle Socratic biology tutor for college cell biology. "
            "Ask ONE short guiding question that nudges the student toward the immediate effect. "
            "Avoid jargon. Never reveal the answer."
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

# ------------------ PDF backends (optional) ------------------
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

# ------------------ File IO ------------------
def read_text(path: str) -> str:
    for enc in ("utf-8","latin-1"):
        try:
            with open(path,"r",encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""

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

def load_corpus(slides_dir: str) -> List[str]:
    texts = []
    if not os.path.isdir(slides_dir): return texts
    base = pathlib.Path(slides_dir)
    for p in base.rglob("*"):
        if not p.is_file(): continue
        ext = p.suffix.lower()
        if ext in SUPPORTED_TEXT_EXTS:
            t = read_text(str(p))
        elif ext == ".pdf":
            t = read_pdf(str(p))
        else:
            t = ""
        if t and len(t.strip()) > 20:
            texts.append(t)
    return texts

# ------------------ Tokenization & helpers ------------------
STOP = {"the","and","for","that","with","this","from","into","are","was","were","has","have","had","can","will","would","could","should",
        "a","an","of","in","on","to","by","as","at","or","be","is","it","its","their","our","your","if","when","then","than","but",
        "we","you","they","which","these","those","there","here","such","may","might","also","very","much","many","most","more","less"}
META = {"lecture","slide","slides","figure","fig","table","exam","objective","objectives","learning","homework","quiz","next","previous",
        "today","describe","how","identify","define","overview","summary"}

def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())

def tokens_nostop(s: str) -> List[str]:
    return [t for t in tokenize(s) if t not in STOP and len(t) > 2]

def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\\.\\!\\?])\\s+|\\n+", text)
    return [re.sub(r"\\s+"," ",p).strip() for p in parts if p and len(p.strip()) > 30]

def relevance(sent: str, q_tokens: List[str]) -> int:
    bag = {}
    for tk in tokenize(sent):
        bag[tk] = bag.get(tk,0)+1
    score = sum(bag.get(q,0) for q in q_tokens)
    s_low = sent.lower()
    if len(q_tokens) >= 2 and " ".join(q_tokens[:2]) in s_low:
        score += 2
    return score

def collect_prompt_matched(corpus: List[str], prompt: str, top_docs=6, max_sents=600) -> List[str]:
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

# ------------------ Topic profiling ------------------
def classify_topic(prompt: str) -> str:
    p = (prompt or "").lower()
    if "transcription" in p: return "transcription"
    if "translation" in p or "ribosome" in p: return "translation"
    if "glycolysis" in p: return "glycolysis"
    if "membrane" in p: return "membrane transport"
    if "protein sorting" in p or "signal sequence" in p or "er signal" in p: return "protein sorting"
    if "cell cycle" in p or "mitosis" in p or "g1" in p or "s phase" in p: return "cell cycle"
    return (p.split(",")[0].split(";")[0] or "this topic").strip()

def topic_bins(topic: str) -> List[str]:
    t = topic
    if t == "transcription":
        return ["Promoter/TFs", "RNA Pol & Elongation", "RNA Processing Intermediates", "mRNA Output"]
    if t == "translation":
        return ["Initiation Components", "Ribosome/Elongation", "Nascent Chain/Intermediates", "Protein Output"]
    if t == "glycolysis":
        return ["Inputs/Regulators", "Key Enzymatic Steps", "Intermediates", "ATP/NADH Output"]
    if t == "protein sorting":
        return ["Targeting Signals", "Translocation Machinery", "Transit Intermediates", "Final Location/Output"]
    if t == "membrane transport":
        return ["Drivers/Signals", "Transport Mechanism", "Transported Intermediates", "Gradient/Flux Output"]
    if t == "cell cycle":
        return ["Cyclins/CDKs", "Checkpoints/Core Steps", "Intermediate States", "Division Outcome"]
    # default, still explicit not vague
    return ["Regulators", "Core Mechanism", "Observable Intermediates", "Final Product/Outcome"]

EXCLUDE_GENERIC = {"sequence","sequences","protein","proteins","factor","factors","general","level","intermediate","process","step","steps","thing","stuff"}

# ------------------ Term harvesting ------------------
PROCESS_SUFFIXES = ("tion","sion","sis","ing","ment","ance","lation","folding","assembly","binding","transport","import","export","repair","processing",
                    "replication","transcription","translation")
BIO_HINTS = {"atp","adp","nad","nadh","fadh2","gdp","gtp","rna","dna","mrna","trna","rrna","peptide","polypeptide","protein","enzyme","substrate","product","gradient",
             "phosphate","membrane","mitochond","chloroplast","cytosol","nucleus","ribosome","polymerase","helicase","ligase","kinase","phosphatase",
             "carrier","channel","receptor","complex","chromosome","histone","promoter","tata","cap","tail","exon","intron","spliceosome",
             "ribosomal","anticodon","codon","signal","translocon","tom","tim","srp","srp receptor"}

def clean_phrase(p: str) -> str:
    p = re.sub(r"\\s+"," ", p).strip(" -—:;,.")
    if any(m in p.lower() for m in META): return ""
    if len(p) < 3 or len(p) > 60: return ""
    if not any(ch.isalpha() for ch in p): return ""
    return p

def tokens_to_termsets(toks: List[str]) -> Tuple[Dict[str,int], Dict[str,int]]:
    ents, procs = {}, {}
    for t in toks:
        if t in STOP or t in META: continue
        if len(t) < 3 or t.isdigit(): continue
        if not any(ch.isalpha() for ch in t): continue
        if t.endswith(PROCESS_SUFFIXES) or t in BIO_HINTS:
            procs[t] = procs.get(t,0)+1
        else:
            ents[t] = ents.get(t,0)+1
    for n in (2,3):
        for i in range(len(toks)-n+1):
            ng = " ".join(toks[i:i+n])
            if any(m in ng for m in META): continue
            if len(ng) < 5 or len(ng) > 40: continue
            if any(ng.endswith(suf) for suf in PROCESS_SUFFIXES):
                procs[ng] = procs.get(ng,0)+1
            else:
                ents[ng] = ents.get(ng,0)+1
    return ents, procs

def harvest_terms(sentences: List[str], prompt: str) -> Tuple[List[str], List[str]]:
    ents, procs = {}, {}
    for s in sentences:
        s_low = s.lower()
        if any(m in s_low for m in META): continue
        _ents, _procs = tokens_to_termsets(tokens_nostop(s_low))
        for k,v in _ents.items(): ents[k] = ents.get(k,0)+v
        for k,v in _procs.items(): procs[k] = procs.get(k,0)+v
    _e2,_p2 = tokens_to_termsets(tokens_nostop(prompt.lower()))
    for k,v in _e2.items(): ents[k] = ents.get(k,0)+v
    for k,v in _p2.items(): procs[k] = procs.get(k,0)+v

    entities = sorted(ents.keys(), key=lambda k: (ents[k], len(k)), reverse=True)
    processes = sorted(procs.keys(), key=lambda k: (procs[k], len(k)), reverse=True)
    entities = [clean_phrase(e) for e in entities]
    processes = [clean_phrase(p) for p in processes]
    # filter out generic
    entities = [e for e in entities if e and e not in EXCLUDE_GENERIC and len(e.split()) <= 3][:20]
    processes = [p for p in processes if p and p not in EXCLUDE_GENERIC and len(p.split()) <= 3][:20]
    return entities, processes

# ------------------ Matching & Socratic hints ------------------
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
    s=re.sub(r"\\s+"," ",s)
    s=s.replace("’","'")
    return s

def matches(user: str, key: str, noun: str) -> bool:
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

def hint_for(user: str, key: str) -> str:
    if key == "increase": return "What would rise first as an immediate consequence?"
    if key == "decrease": return "Which direct output would drop first?"
    if key == "no_change": return "Is there buffering or a parallel path that keeps it steady?"
    if key == "truncated": return "Would the product be shorter than normal?"
    if key == "mislocalized": return "Where would the protein end up without its targeting info?"
    if key == "no_initiation": return "If initiation can't happen, does anything proceed?"
    if key == "no_elongation": return "If elongation stalls, what happens to building the product?"
    if key == "no_termination": return "If termination fails, does synthesis run past the end?"
    if key == "frameshift": return "What happens to the reading frame after indel events?"
    return "Focus on the immediate effect, not long-term compensation."

# ------------------ FITB synthesis (diverse) ------------------
def synthesize_fitb(entities: List[str], processes: List[str], rng: random.Random) -> List[Dict[str,str]]:
    e = [x for x in entities if x not in EXCLUDE_GENERIC][:10] or ["substrate","cofactor"]
    p = [x for x in processes if x not in EXCLUDE_GENERIC][:10] or ["processing","assembly"]
    def pick(lst): return rng.choice(lst)
    templates = [
        lambda: (f"When {pick(e)} availability rises, the immediate output of {pick(p)} would ______.", "increase",""),
        lambda: (f"A strong inhibitor reduces {pick(p)}. The near-term product formation would ______.", "decrease",""),
        lambda: (f"A mutation prevents recognition of the start signal for {pick(p)}. The process would show ______.", "no_initiation",""),
        lambda: (f"A drug stalls movement required for {pick(p)} to continue. The process would show ______.", "no_elongation",""),
        lambda: (f"A factor that ends {pick(p)} cannot act. The process would show ______.", "no_termination",""),
        lambda: (f"A single-base substitution creates a premature stop during {pick(p)}. The final product would be ______.", "truncated",""),
        lambda: (f"A one-base insertion in the coding region during {pick(p)} most likely causes a ______ mutation.", "frameshift",""),
        lambda: (f"The targeting signal for {pick(e)} is deleted. The protein would be ______.", "mislocalized",""),
        lambda: (f"When a downstream step of {pick(p)} is blocked, unfinished forms would ______.", "increase",""),
        lambda: (f"When a parallel pathway compensates for {pick(p)}, the immediate output would show ______.", "no_change",""),
    ]
    rng.shuffle(templates)
    stems, items = set(), []
    for make in templates:
        stem, key, noun = make()
        if any(w in META for w in tokenize(stem)): continue
        if stem.lower() not in stems:
            stems.add(stem.lower())
            items.append({"stem": stem, "key": key, "noun": noun})
        if len(items) == 4: break
    while len(items) < 4:
        stem, key, noun = (f"When {pick(e)} engages earlier, the immediate output of {pick(p)} would ______.","increase","")
        if stem.lower() not in stems:
            stems.add(stem.lower()); items.append({"stem": stem, "key": key, "noun": noun})
    return items

# ------------------ Drag bins & tailored terms ------------------
def make_bins(prompt: str) -> Tuple[str, List[str]]:
    topic = classify_topic(prompt)
    labels = topic_bins(topic)
    return topic, labels

def choose_specific_terms(entities: List[str], processes: List[str], topic: str, rng: random.Random) -> List[str]:
    ban = set(EXCLUDE_GENERIC) | {"mrna cap","rna","dna sequence","sequence motif","complex","component","site","sites","region","regions"}
    cands = [t for t in (entities + processes) if t and t not in ban and any(ch.isalpha() for ch in t)]
    # topic-preferred keywords to bias specificity
    prefs = {
        "transcription": ["tata box","promoter","rna polymerase ii","transcription factor","spliceosome","exon","intron","5' cap","poly-a tail"],
        "translation": ["ribosome","initiation factors","trna","anticodon","start codon","elongation factor","peptidyl transferase","stop codon"],
        "glycolysis": ["hexokinase","pfk-1","aldolase","pyruvate kinase","nad+","nadh","atp","fructose 1,6-bisphosphate"],
        "protein sorting": ["signal sequence","srp","translocon","tom","tim","signal peptidase","nuclear localization signal","lysosome"],
        "membrane transport": ["uniporter","symporter","antiporter","ion channel","carrier","electrochemical gradient","aquaporin","na+/k+ atpase"],
        "cell cycle": ["cyclin","cdk","checkpoint","anaphase","metaphase","g1","s phase","g2","mitosis"],
    }.get(topic, [])
    # promote prefs
    promoted = [p for p in prefs if p in cands]
    rest = [c for c in cands if c not in promoted]
    rng.shuffle(rest)
    terms = (promoted + rest)[:4]
    # fallback to safe specifics if too few
    defaults = {
        "transcription": ["promoter","rna polymerase ii","tata box","exon"],
        "translation": ["ribosome","trna","start codon","elongation factor"],
        "glycolysis": ["pfk-1","nadh","pyruvate kinase","atp"],
        "protein sorting": ["signal sequence","srp","translocon","lysosome"],
        "membrane transport": ["ion channel","carrier","symporter","gradient"],
        "cell cycle": ["cyclin","cdk","checkpoint","anaphase"],
    }.get(topic, ["enzyme","substrate","intermediate","product"])
    if len(terms) < 4:
        for d in defaults:
            if d not in terms:
                terms.append(d)
            if len(terms) == 4: break
    return terms[:4]

def build_drag_items(entities: List[str], processes: List[str], prompt: str, rng: random.Random) -> Tuple[str, List[str], List[str], Dict[str,str]]:
    topic, labels = make_bins(prompt)
    terms = choose_specific_terms(entities, processes, topic, rng)
    # simple heuristic mapping to explicit bins
    def map_term(term: str) -> int:
        t = term.lower()
        if topic == "transcription":
            if "promoter" in t or "tata" in t or "factor" in t: return 0
            if "polymerase" in t or "elongation" in t: return 1
            if "intron" in t or "exon" in t or "cap" in t or "tail" in t or "spliceosome" in t: return 2
            return 3
        if topic == "translation":
            if "initiation" in t or "start codon" in t: return 0
            if "elongation" in t or "ribosome" in t or "trna" in t: return 1
            if "peptidyl" in t or "nascent" in t: return 2
            return 3
        if topic == "glycolysis":
            if t in {"glucose","atp","adp","nad+","nad"}: return 0
            if any(x in t for x in ["hexokinase","pfk","aldolase","pyruvate kinase"]): return 1
            if any(x in t for x in ["fructose","bisphosphate","g3p","pep","intermediate"]): return 2
            if any(x in t for x in ["atp","nadh","pyruvate"]): return 3
            return 1
        if topic == "protein sorting":
            if "signal" in t or "nuclear localization" in t: return 0
            if "srp" in t or "translocon" in t or "tom" in t or "tim" in t: return 1
            if "peptidase" in t or "transit" in t: return 2
            return 3
        if topic == "membrane transport":
            if any(x in t for x in ["signal","stimulus","gradient"]): return 0
            if any(x in t for x in ["channel","transporter","carrier","symporter","antiporter","uniporter","na+/k+ atpase"]): return 1
            if "intermediate" in t: return 2
            return 3
        if topic == "cell cycle":
            if any(x in t for x in ["cyclin","cdk"]): return 0
            if any(x in t for x in ["checkpoint","metaphase","anaphase","g1","s phase","g2","mitosis"]): return 1
            if any(x in t for x in ["intermediate","state"]): return 2
            return 3
        # default
        if any(x in t for x in ["signal","regulator"]): return 0
        if any(x in t for x in ["ase","enzyme","mechanism","complex"]): return 1
        if "intermediate" in t: return 2
        return 3
    answers = {term: labels[map_term(term)] for term in terms}
    return topic, labels, terms, answers

# ------------------ UI (prompt + generate + both activities) ------------------
st.title("Let's Practice Biology!")
prompt = st.text_input(
    "Enter a topic for review and press generate:",
    value="",
    placeholder="e.g., transcription, translation, glycolysis, membrane transport…",
    label_visibility="visible",
)

if st.button("Generate"):
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)
    matched = collect_prompt_matched(st.session_state.corpus, prompt)
    rng = random.Random(new_seed())
    entities, processes = harvest_terms(matched, prompt)
    st.session_state.topic_name = classify_topic(prompt) or prompt.strip() or "this topic"
    st.session_state.fitb = synthesize_fitb(entities, processes, rng)
    st.session_state.drag_topic, st.session_state.drag_labels, st.session_state.drag_bank, st.session_state.drag_answer = build_drag_items(entities, processes, prompt, rng)
    st.success("Generated fresh activities from your slides for this prompt.")

# -------- Activity 1: FITB with Socratic feedback --------
if "fitb" in st.session_state:
    topic_name = st.session_state.get("topic_name","this topic")
    st.markdown(f"## Use your knowledge of **{topic_name}** to answer the following")
    for i, item in enumerate(st.session_state.fitb, start=1):
        u = st.text_input(f"{i}. {item['stem']}", key=f"fitb_{i}")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Hint", key=f"fitb_hint_{i}"):
                if llm_coach_available():
                    q = f"Question: {item['stem']}\nStudent answer: {u or '(blank)'}\nTarget concept: {item['key']}\nGive one short guiding question."
                    out = call_llm(q)
                    st.info(out or hint_for(u, item['key']))
                else:
                    st.info(hint_for(u, item["key"]))
        with col2:
            if st.button("Check", key=f"fitb_check_{i}"):
                ok = matches(u, item["key"], item.get("noun",""))
                if ok:
                    st.success("Good — that aligns with the immediate consequence.")
                else:
                    guide = hint_for(u, item["key"])
                    if llm_coach_available():
                        q = f"Question: {item['stem']}\nStudent answer: {u or '(blank)'}\nTarget concept: {item['key']}\nGive one short guiding question."
                        out = call_llm(q)
                        guide = out or guide
                    st.info(guide)
        with col3:
            if st.button("Reveal", key=f"fitb_rev_{i}"):
                gloss = {
                    "increase":"Immediate output goes up (accumulation counts).",
                    "decrease":"Immediate output drops.",
                    "no_change":"Parallel route/buffer keeps output steady.",
                    "truncated":"Product is shorter due to early stop.",
                    "mislocalized":"Protein ends up in the wrong place.",
                    "no_initiation":"Process never starts (accepts 'no transcription'/'no translation').",
                    "no_elongation":"Process starts but cannot proceed.",
                    "no_termination":"Process cannot end properly; may read through.",
                    "frameshift":"Reading frame changes after an insertion/deletion."
                }.get(item["key"], "Focus on the immediate, direct effect.")
                st.info(gloss)

# -------- Activity 2: Drag-into-Bins (topic-tailored) --------
if all(k in st.session_state for k in ["drag_topic","drag_labels","drag_bank","drag_answer"]):
    st.markdown("---")
    topic = st.session_state.drag_topic
    st.markdown(f"## Place the following items in the corresponding categories related to **{topic}**")
    labels = st.session_state.drag_labels
    terms  = st.session_state.drag_bank
    answer = st.session_state.drag_answer

    items_html = "".join([f'<li class="card">{t}</li>' for t in terms])
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
        }}
        .zone {{ display:flex; gap:16px; }}
        .left {{ flex: 1; }}
        .right {{ flex: 2; display:grid; grid-template-columns: repeat(2, 1fr); gap:16px; }}
        .title {{ font-weight: 600; margin-bottom: 6px; }}
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
            score.innerHTML = "<span class='bad'>Drag terms into bins first.</span>";
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
    st.components.v1.html(html, height=640, scrolling=True)
