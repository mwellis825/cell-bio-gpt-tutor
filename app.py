
import os, re, json, pathlib, random, time, sys
import streamlit as st
from typing import List, Tuple, Dict

# ------------------ Page & Globals ------------------
st.set_page_config(page_title="Let's Practice Biology!", page_icon="🎓", layout="wide")
SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm", ".pdf"}

def new_seed() -> int:
    return int(time.time() * 1000) ^ random.randint(0, 1_000_000)

# ------------------ LLM + fallback hinting ------------------
def llm_hint(stem: str, target: str, topic: str) -> str:
    api = os.environ.get("OPENAI_API_KEY")
    if api:
        try:
            import openai  # type: ignore
            openai.api_key = api
            sys_prompt = (
                "You are a Socratic college-level cell biology tutor. "
                "Given a single question (fill‑in‑the‑blank or drag-and-drop item) and the intended target concept, "
                "ask ONE concise, specific guiding question that nudges the student toward the answer. "
                "Avoid jargon, do not reveal the answer."
            )
            user_msg = f"Topic: {topic}\nQuestion: {stem}\nTarget concept: {target}\nWrite one guiding question (≤20 words)."
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_msg}],
                temperature=0.3,
                max_tokens=60,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
    # Concise topic-aware fallback
    t = (topic or "").lower()
    if "transcription" in t:
        return "What changes first at the promoter if that step can’t occur?"
    if "translation" in t:
        return "What happens at the ribosome if that step is blocked?"
    if "replication" in t:
        return "What would you observe at the fork if that factor fails?"
    if "glycolysis" in t:
        return "Which metabolite rises or falls immediately when that step is inhibited?"
    if "membrane" in t:
        return "How would the gradient or flux change right away?"
    if "sorting" in t:
        return "Where would the protein go without that targeting signal?"
    if "cell cycle" in t:
        return "Which checkpoint or phase is directly affected first?"
    return "Focus on the immediate, direct effect before compensation."

# ------------------ PDF/text loading ------------------
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
                return "\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""
    if PDF_BACKEND == "pypdf":
        try:
            with open(path,"rb") as f:
                reader = pypdf.PdfReader(f)  # type: ignore
                return "\n".join([(p.extract_text() or "") for p in reader.pages])
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

def load_corpus(slides_dir: str) -> List[str]:
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
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    return [re.sub(r"\s+"," ",p).strip() for p in parts if p and len(p.strip()) > 30]

def relevance(sent: str, q_tokens: List[str]) -> int:
    bag = {}
    for tk in tokenize(sent):
        bag[tk] = bag.get(tk,0)+1
    score = sum(bag.get(q,0) for q in q_tokens)
    s_low = sent.lower()
    if len(q_tokens) >= 2 and " ".join(q_tokens[:2]) in s_low:
        score += 2
    return score

def collect_prompt_matched(corpus: List[str], prompt: str, top_docs=6, max_sents=800) -> List[str]:
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

# ------------------ Topic recognition ------------------
def classify_topic(prompt: str) -> str:
    p = (prompt or "").lower()
    if "replicat" in p: return "dna replication"
    if "transcription" in p: return "transcription"
    if "translation" in p: return "translation"
    if "glycolysis" in p: return "glycolysis"
    if "membrane" in p or "transport" in p: return "membrane transport"
    if "protein sorting" in p or "signal sequence" in p or "er signal" in p: return "protein sorting"
    if "cell cycle" in p or "mitosis" in p: return "cell cycle"
    return (p.split(",")[0].split(";")[0] or "this topic").strip()

# ------------------ Harvest terms ------------------
PROCESS_SUFFIXES = ("tion","sion","sis","ing","ment","ance","lation","folding","assembly","binding","transport","import","export","repair","processing",
                    "replication","transcription","translation")
BIO_HINTS = {"atp","adp","nad","nadh","fadh2","gdp","gtp","rna","dna","mrna","trna","rrna","peptide","polypeptide","protein","enzyme","substrate","product","gradient",
             "phosphate","membrane","mitochond","chloroplast","cytosol","nucleus","ribosome","polymerase","helicase","ligase","kinase","phosphatase",
             "carrier","channel","receptor","complex","chromosome","histone","promoter","tata","cap","tail","exon","intron","spliceosome",
             "ribosomal","anticodon","codon","signal","translocon","tom","tim","srp","srp receptor","pyruvate","glucose","lactate","g3p","pep","f1,6bp"}

EXCLUDE_GENERIC = {"sequence","sequences","protein","proteins","factor","factors","general","level","intermediate","process","step","steps","thing","stuff"}

def clean_phrase(p: str) -> str:
    p = re.sub(r"\s+"," ", p).strip(" -—:;,.")
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
    _e2,_p2 = tokens_to_termsets(tokens_nostop((prompt or "").lower()))
    for k,v in _e2.items(): ents[k] = ents.get(k,0)+v
    for k,v in _p2.items(): procs[k] = procs.get(k,0)+v

    entities = sorted(ents.keys(), key=lambda k: (ents[k], len(k)), reverse=True)
    processes = sorted(procs.keys(), key=lambda k: (procs[k], len(k)), reverse=True)
    entities = [clean_phrase(e) for e in entities]
    processes = [clean_phrase(p) for p in processes]
    entities = [e for e in entities if e and e not in EXCLUDE_GENERIC and len(e.split()) <= 3][:20]
    processes = [p for p in processes if p and p not in EXCLUDE_GENERIC and len(p.split()) <= 4][:20]
    return entities, processes

# ------------------ Acceptance sets (lenient) ------------------
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
    s=re.sub(r"\s+"," ",s)
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

# ------------------ FITB synthesis (diverse reasoning) ------------------
def fitb_stems_from_terms(entities: List[str], processes: List[str], topic: str, rng: random.Random) -> List[Dict[str,str]]:
    # Four reasoning modes: blockage, acceleration, removal, rescue/comparison
    E = entities[:8]
    P = processes[:8]
    stems = []

    def add_blockage():
        if P:
            p = rng.choice(P); stems.append((f"If {p} is blocked, the immediate output would ______.", "decrease",""))
        if E:
            e = rng.choice(E); stems.append((f"Without {e}, the process would show ______.", "no_initiation",""))

    def add_acceleration():
        if P:
            p = rng.choice(P); stems.append((f"If {p} speeds up, the near-term product would ______.", "increase",""))
        if E:
            e = rng.choice(E); stems.append((f"If {e} accumulates, upstream pressure would ______.", "increase",""))

    def add_removal():
        if "transcription" in topic:
            stems.append(("If capping fails, transcript stability would ______.", "decrease",""))
        elif "translation" in topic:
            stems.append(("If the ribosome cannot move forward, elongation would ______.", "no_elongation",""))
        elif "replication" in topic:
            stems.append(("If helicase does not load, fork progression would ______.", "decrease",""))
        elif "glycolysis" in topic:
            stems.append(("If NAD+ is unavailable, glyceraldehyde-3-phosphate would ______.", "increase",""))
        elif "membrane transport" in topic:
            stems.append(("If a pump stops, the gradient would ______.", "decrease",""))
        elif "protein sorting" in topic:
            stems.append(("If a signal sequence is missing, the protein would be ______.", "mislocalized",""))
        elif "cell cycle" in topic:
            stems.append(("If the spindle checkpoint is active, anaphase onset would ______.", "decrease",""))

    def add_rescue():
        if "glycolysis" in topic:
            stems.append(("If PFK-1 inhibition is relieved, ATP production would ______.", "increase",""))
        elif "translation" in topic:
            stems.append(("Restoring a stop codon would make termination ______.", "increase",""))
        elif "transcription" in topic:
            stems.append(("If promoter recognition is restored, initiation would ______.", "increase",""))
        elif "replication" in topic:
            stems.append(("Supplying primers externally would make synthesis ______.", "increase",""))
        else:
            stems.append(("If the upstream block is bypassed, downstream output would ______.", "increase",""))

    # Ensure variety
    add_blockage(); add_acceleration(); add_removal(); add_rescue()

    # Dedup, shuffle, pick 4
    seen = set(); uniq = []
    for s in stems:
        if s[0].lower() not in seen:
            uniq.append(s); seen.add(s[0].lower())
    rng.shuffle(uniq)
    return [{"stem": s, "key": k, "noun": n} for (s,k,n) in uniq[:4]]

# ------------------ Drag-and-Drop: (bin, action) pairs first ------------------
def fundamental_pairs(topic: str) -> List[Tuple[str,str]]:
    t = topic.lower()
    if "glycolysis" in t:
        return [
            ("Initial substrate/step","Glucose enters glycolysis"),
            ("Regulatory enzyme step","PFK-1 commits glycolysis"),
            ("Intermediate metabolite","Fructose-1,6-bisphosphate forms"),
            ("End product of glycolysis","Pyruvate is produced"),
        ]
    if "transcription" in t:
        return [
            ("Start of transcription","Promoter recruits polymerase"),
            ("Protein involved in transcription","RNA polymerase synthesizes RNA"),
            ("RNA processing event","mRNA is capped"),
            ("End product of transcription","mRNA is produced"),
        ]
    if "translation" in t:
        return [
            ("Start of translation","Ribosome assembles at start codon"),
            ("Ribosome site/function","tRNA pairs in the P site"),
            ("Elongation step","Polypeptide chain elongates"),
            ("End product of translation","Polypeptide is released"),
        ]
    if "dna replication" in t:
        return [
            ("First step of replication","Helicase unwinds DNA"),
            ("Protein involved in replication","DNA polymerase extends DNA"),
            ("Intermediate state","Okazaki fragments accumulate"),
            ("End product of replication","Two DNA copies form"),
        ]
    if "membrane transport" in t:
        return [
            ("Type of transporter","Channel opens for ions"),
            ("Driving force/gradient","Gradient drives diffusion"),
            ("Molecule moved","Transporter moves solute"),
            ("Outcome of transport","Solute concentration changes"),
        ]
    if "protein sorting" in t:
        return [
            ("Targeting signal","Signal sequence directs protein"),
            ("Targeting machinery","SRP directs ribosome"),
            ("Transit/intermediate","Translocon imports protein"),
            ("Final location/output","Protein reaches destination"),
        ]
    if "cell cycle" in t:
        return [
            ("Checkpoint/start event","Checkpoint allows progression"),
            ("Core division step","Chromosomes align at metaphase"),
            ("Intermediate state","Sister chromatids separate"),
            ("Outcome of division","Two daughter cells form"),
        ]
    return [
        ("Start event","Process begins"),
        ("Key protein","Enzyme acts on substrate"),
        ("Intermediate","Intermediate appears"),
        ("Final product","Product is formed"),
    ]

def build_pairs_from_slides(entities: List[str], processes: List[str], topic: str, rng: random.Random) -> List[Tuple[str,str]]:
    pairs = []
    def add(bin_label, action):
        if (bin_label, action) not in pairs:
            pairs.append((bin_label, action))

    t = topic.lower()
    all_text = " ".join(entities + processes).lower()

    if "glycolysis" in t:
        if "glucose" in all_text:
            add("Initial substrate/step","Glucose enters glycolysis")
        if "pfk" in all_text:
            add("Regulatory enzyme step","PFK-1 commits glycolysis")
        if any(x in all_text for x in ["f1,6bp","fructose","g3p","pep"]):
            add("Intermediate metabolite","Fructose-1,6-bisphosphate forms")
        add("End product of glycolysis","Pyruvate is produced")
    elif "transcription" in t:
        add("Start of transcription","Promoter recruits polymerase")
        add("Protein involved in transcription","RNA polymerase synthesizes RNA")
        if any(x in all_text for x in ["cap","tail","splice"]):
            add("RNA processing event","mRNA is capped")
        add("End product of transcription","mRNA is produced")
    elif "translation" in t:
        add("Start of translation","Ribosome assembles at start codon")
        add("Ribosome site/function","tRNA pairs in the P site")
        add("Elongation step","Polypeptide chain elongates")
        add("End product of translation","Polypeptide is released")
    elif "dna replication" in t:
        add("First step of replication","Helicase unwinds DNA")
        add("Protein involved in replication","DNA polymerase extends DNA")
        add("Intermediate state","Okazaki fragments accumulate")
        add("End product of replication","Two DNA copies form")
    elif "membrane transport" in t:
        add("Type of transporter","Channel opens for ions")
        add("Driving force/gradient","Gradient drives diffusion")
        add("Molecule moved","Transporter moves solute")
        add("Outcome of transport","Solute concentration changes")
    elif "protein sorting" in t:
        add("Targeting signal","Signal sequence directs protein")
        add("Targeting machinery","SRP directs ribosome")
        add("Transit/intermediate","Translocon imports protein")
        add("Final location/output","Protein reaches destination")
    elif "cell cycle" in t:
        add("Checkpoint/start event","Checkpoint allows progression")
        add("Core division step","Chromosomes align at metaphase")
        add("Intermediate state","Sister chromatids separate")
        add("Outcome of division","Two daughter cells form")
    else:
        pairs = fundamental_pairs(topic)

    if not pairs:
        pairs = fundamental_pairs(topic)

    rng.shuffle(pairs)
    k = rng.choice([2,3,4])  # choose 2–4 bins for clarity
    pairs = pairs[:k]
    return pairs

# ------------------ UI ------------------
st.title("Let's Practice Biology!")
prompt = st.text_input(
    "Enter a topic for review and press generate:",
    value="",
    placeholder="e.g., transcription, translation, glycolysis, protein sorting…",
    label_visibility="visible",
)

if st.button("Generate"):
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)
    matched = collect_prompt_matched(st.session_state.corpus, prompt)
    rng = random.Random(new_seed())
    topic = classify_topic(prompt) or "this topic"
    entities, processes = harvest_terms(matched, prompt)
    st.session_state.topic = topic
    st.session_state.fitb = fitb_stems_from_terms(entities, processes, topic, rng)

    pairs = build_pairs_from_slides(entities, processes, topic, rng)  # (bin, action)
    st.session_state.drag_labels = [b for (b, _) in pairs]
    st.session_state.drag_bank   = [a for (_, a) in pairs]
    st.session_state.drag_answer = {a: b for (b, a) in pairs}
    st.success("Generated fresh activities.")

# -------- Activity 1: FITB --------
if "fitb" in st.session_state:
    topic_name = st.session_state.get("topic","this topic")
    st.markdown(f"## Use your knowledge of **{topic_name}** to answer the following")
    for idx, item in enumerate(st.session_state.fitb):
        u = st.text_input(item["stem"], key=f"fitb_{idx}")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Hint", key=f"hint_{idx}"):
                st.info(llm_hint(item["stem"], item["key"], topic_name))
        with col2:
            if st.button("Check", key=f"check_{idx}"):
                ok = matches(u, item["key"], item.get("noun",""))
                if ok:
                    st.success("That’s right! Great work!")
                else:
                    st.info(llm_hint(item["stem"], item["key"], topic_name))
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

# -------- Activity 2: Drag-and-Drop --------
if all(k in st.session_state for k in ["drag_labels","drag_bank","drag_answer"]):
    st.markdown("---")
    topic = st.session_state.get("topic","this topic")
    labels = st.session_state.drag_labels
    terms  = st.session_state.drag_bank
    answer = st.session_state.drag_answer

    st.markdown(f"## Place the following items in the corresponding categories related to **{topic}**")

    items_html = "".join([f'<li class="card" draggable="true">{t}</li>' for t in terms])
    # grid columns count: 2 if 2 bins, otherwise ceil(n/2)
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
          const wrong = [];
          for (const [term, want] of Object.entries(ANSWERS)) {{
            total += 1;
            let got = "Bank";
            for (const [label, items] of Object.entries(bins)) {{
              if (items.includes(term)) {{ got = label; break; }}
            }}
            if (got === want) correct += 1;
            else wrong.push([term, want, got]);
          }}
          const score = document.getElementById('score');
          if (total === 0) {{
            score.innerHTML = "<span class='bad'>Drag items into bins first.</span>";
          }} else if (correct === total) {{
            score.innerHTML = "<span class='ok'>All bins correct! 🎉</span>";
          }} else {{
            score.innerHTML = "<span class='bad'>" + correct + "/" + total + " correct — try adjusting and re-check.</span>";
          }}
          // Expose the wrong list so Streamlit can request hints per item via dropdown
          const carrier = document.getElementById('wrong-carrier') || document.createElement('div');
          carrier.id = 'wrong-carrier';
          carrier.setAttribute('data-wrong', JSON.stringify(wrong));
          document.body.appendChild(carrier);
        }});
      </script>
    </body>
    </html>
    """
    st.components.v1.html(html, height=650, scrolling=True)

    # DnD hint helper: let the student pick any item for a hint
    st.caption("Need a nudge on a specific item?")
    chosen_item = st.selectbox("Pick an item for a hint:", ["(choose an item)"] + terms, index=0)
    if chosen_item != "(choose an item)":
        target_bin = answer.get(chosen_item, "the correct category")
        st.info(llm_hint(f"Place '{chosen_item}' into the correct category", target_bin, topic))
