
import os, re, json, pathlib, random, time
import streamlit as st
from typing import List, Tuple, Dict

# ------------------ Page & Globals ------------------
st.set_page_config(page_title="Let's Practice Biology!", page_icon="🧬", layout="wide")
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
        # gpt-4o-mini is efficient; user may swap to their preferred model
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
        "today","describe","how","identify","define"}

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

# ------------------ Term harvesting ------------------
PROCESS_SUFFIXES = ("tion","sion","sis","ing","ment","ance","lation","folding","assembly","binding","transport","import","export","repair","processing",
                    "replication","transcription","translation")
BIO_HINTS = {"atp","adp","nad","nadh","fadh2","gdp","gtp","rna","dna","mrna","trna","rrna","peptide","protein","enzyme","substrate","product","gradient",
             "phosphate","membrane","mitochond","chloroplast","cytosol","nucleus","ribosome","polymerase","helicase","ligase","kinase","phosphatase",
             "carrier","channel","receptor","complex","chromosome","histone","promoter","exon","intron","ribosomal","anticodon","codon"}

def clean_phrase(p: str) -> str:
    p = re.sub(r"\\s+"," ", p).strip(" -—:;,.")
    if any(m in p.lower() for m in META): return ""
    if len(p) < 3 or len(p) > 60: return ""
    if not any(ch.isalpha() for ch in p): return ""
    return p

def harvest_terms(sentences: List[str], prompt: str) -> Tuple[List[str], List[str]]:
    ents, procs = {}, {}
    def add_terms(toks: List[str]):
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
    for s in sentences:
        s_low = s.lower()
        if any(m in s_low for m in META): continue
        toks = tokens_nostop(s_low)
        add_terms(toks)
    add_terms(tokens_nostop(prompt.lower()))
    entities = sorted(ents.keys(), key=lambda k: (ents[k], len(k)), reverse=True)
    processes = sorted(procs.keys(), key=lambda k: (procs[k], len(k)), reverse=True)
    entities = [clean_phrase(e) for e in entities]
    processes = [clean_phrase(p) for p in processes]
    entities = [e for e in entities if e and len(e.split()) <= 2][:12]
    processes = [p for p in processes if p and len(p.split()) <= 3][:12]
    return entities, processes

# ------------------ Matching & Socratic hints ------------------
UP = {"increase","increases","increased","up","higher","stabilizes","stabilize","stabilized","faster","more","↑","improves","greater","accumulate","accumulates","builds up"}
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
    if key == "increase": return "Does this change make the immediate output go up, even before regulation catches up?"
    if key == "decrease": return "Which output would drop first if this step slows down?"
    if key == "no_change": return "Is there a parallel path or buffer that keeps output steady for a while?"
    if key == "truncated": return "If a premature stop occurs, would the product be shorter than normal?"
    if key == "mislocalized": return "Without a targeting signal, where does the protein likely end up?"
    if key == "no_initiation": return "If the start step can’t occur, does the process proceed at all?"
    if key == "no_elongation": return "If movement along the template stalls, what happens to building the product?"
    if key == "no_termination": return "If the stop step fails, does the machine run past the end?"
    if key == "frameshift": return "A one‑base insertion/deletion in coding sequence often causes what reading effect?"
    return "Focus on the immediate, direct effect — not compensatory changes."

# ------------------ FITB synthesis (diverse) ------------------
def synthesize_fitb(entities: List[str], processes: List[str], rng: random.Random) -> List[Dict[str,str]]:
    e = entities[:8] or ["substrate","cofactor"]
    p = processes[:8] or ["processing","assembly"]
    def pick(lst): return rng.choice(lst)
    templates = [
        lambda: (f"When {pick(e)} availability rises, the immediate output of {pick(p)} would ______.", "increase",""),
        lambda: (f"A strong inhibitor reduces {pick(p)}. The near‑term product formation would ______.", "decrease",""),
        lambda: (f"A mutation prevents recognition of the start signal for {pick(p)}. The process would show ______.", "no_initiation",""),
        lambda: (f"A drug stalls movement required for {pick(p)} to continue. The process would show ______.", "no_elongation",""),
        lambda: (f"A factor that ends {pick(p)} cannot act. The process would show ______.", "no_termination",""),
        lambda: (f"A single‑base substitution creates a premature stop during {pick(p)}. The final product would be ______.", "truncated",""),
        lambda: (f"A one‑base insertion in the coding region during {pick(p)} most likely causes a ______ mutation.", "frameshift",""),
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

# ------------------ Drag bins & simple terms ------------------
BIN_TITLES = ["Upstream factors", "Core step", "Intermediates/Evidence", "Immediate output"]

def make_bins(prompt: str) -> List[str]:
    # IMPORTANT: no pre-colon words; return plain, simple bin titles only.
    return BIN_TITLES[:]  # exactly 4

def choose_simple_terms(entities: List[str], processes: List[str], rng: random.Random) -> List[str]:
    candidates = [t for t in (entities + processes) if t and len(t.split()) <= 2 and t not in META]
    bad = {"factor","factors","general","level","intermediate","process","step","steps"}
    candidates = [c for c in candidates if c not in bad and not c.isdigit() and any(ch.isalpha() for ch in c)]
    rng.shuffle(candidates)
    # Guarantee max 4 simple draggables
    defaults = ["enzyme","substrate","product","protein"]
    out = (candidates + [x for x in defaults if x not in candidates])[:4]
    return out

def map_term(term: str) -> int:
    t = term.lower()
    upstream_kw = {"polymerase","helicase","ligase","kinase","phosphatase","tf","cap","promoter","aminoacyl","initiator","chaperone","carrier","channel","receptor"}
    core_kw = {"replication","transcription","translation","synthesis","glycolysis","transport","splicing","folding","elongation","translocation","processing"}
    evid_kw = {"mrna","nascent","okazaki","intermediate","precursor","fragment","polyribosome","charged","nadh","atp","signal","gradient"}
    out_kw = {"protein","polypeptide","copy","replicated","atp","product"}
    if any(k in t for k in upstream_kw): return 0
    if any(k in t for k in core_kw): return 1
    if any(k in t for k in evid_kw): return 2
    if any(k in t for k in out_kw): return 3
    return 1

def build_drag_items(entities: List[str], processes: List[str], prompt: str, rng: random.Random) -> Tuple[List[str], List[str], Dict[str,str]]:
    labels = make_bins(prompt)
    terms = choose_simple_terms(entities, processes, rng)
    answers = {term: labels[map_term(term)] for term in terms}
    return labels, terms, answers

# ------------------ UI (veneer only changed) ------------------
st.title("Let's Practice Biology!")
prompt = st.text_input(
    "Enter a topic for review and press generate:",
    value="",
    placeholder="e.g., energetics, protein structure, membrane transport…",
    label_visibility="visible",
)

if st.button("Generate"):
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)
    matched = collect_prompt_matched(st.session_state.corpus, prompt)
    rng = random.Random(new_seed())
    entities, processes = harvest_terms(matched, prompt)
    st.session_state.fitb = synthesize_fitb(entities, processes, rng)
    st.session_state.drag_labels, st.session_state.drag_bank, st.session_state.drag_answer = build_drag_items(entities, processes, prompt, rng)
    st.success("Generated fresh activities from your slides for this prompt.")

# -------- Activity 1: FITB with Socratic feedback (auto LLM if available) --------
if "fitb" in st.session_state:
    st.markdown("## Activity 1 — Apply the concept")
    # Keep the rubric concise; accept increase/decrease/accumulate/no change + initiation/elongation/termination issues
    st.caption("Short phrases accepted: increase/decrease, accumulate (counts as increase), no change; truncated, mislocalized; no initiation (also 'no transcription'/'no translation'), no elongation, no termination; frameshift.")
    for i, item in enumerate(st.session_state.fitb, start=1):
        u = st.text_input(f"{i}. {item['stem']}", key=f"fitb_{i}")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button(f"Hint {i}", key=f"fitb_hint_{i}"):
                if llm_coach_available():
                    q = f"Question: {item['stem']}\nStudent answer: {u or '(blank)'}\nTarget concept: {item['key']}\nGive one short guiding question."
                    out = call_llm(q)
                    st.info(out or hint_for(u, item["key"]))
                else:
                    st.info(hint_for(u, item["key"]))
        with col2:
            if st.button(f"Check {i}", key=f"fitb_check_{i}"):
                ok = matches(u, item["key"], item.get("noun",""))
                if ok:
                    st.success("Good — that aligns with the immediate consequence.")
                else:
                    # Socratic feedback: don't mark 'incorrect'; guide instead
                    guide = hint_for(u, item["key"])
                    if llm_coach_available():
                        q = f"Question: {item['stem']}\nStudent answer: {u or '(blank)'}\nTarget concept: {item['key']}\nGive one short guiding question."
                        out = call_llm(q)
                        guide = out or guide
                    st.info(guide)
        with col3:
            if st.button(f"Reveal {i}", key=f"fitb_rev_{i}"):
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

# -------- Activity 2: Drag‑into‑Bins (exactly 4 draggables, simple terms; no pre-colon words) --------
if all(k in st.session_state for k in ["drag_labels","drag_bank","drag_answer"]):
    st.markdown("---")
    st.markdown("## Activity 2 — Drag the terms into the correct bin")
    labels = st.session_state.drag_labels            # e.g., ["Upstream factors", "Core step", "Intermediates/Evidence", "Immediate output"]
    terms  = st.session_state.drag_bank             # <= 4 simple terms
    answer = st.session_state.drag_answer           # term -> label

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
          const bins = {{
            { " + ".join([]) + " }
          }};
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
            score.innerHTML = "<span class='bad'>" + correct + "/" + total + " correct — try adjusting a couple and re-check.</span>";
          }}
        }});
      </script>
    </body>
    </html>
    """
    st.components.v1.html(html, height=640, scrolling=True)
