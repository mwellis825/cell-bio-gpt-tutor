
import os, re, json, pathlib, random, time, sys
import streamlit as st
from typing import List, Tuple, Dict

# ------------------ Page & Globals ------------------
st.set_page_config(page_title="Let's Practice Biology!", page_icon="🎓", layout="wide")
SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm", ".pdf"}

def new_seed() -> int:
    return int(time.time() * 1000) ^ random.randint(0, 1_000_000)

# ------------------ Optional LLM (auto) ------------------
def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))

def llm_hint(stem: str, target: str) -> str:
    try:
        import openai  # type: ignore
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        sys_prompt = (
            "You are a Socratic college-level cell biology tutor. "
            "Given a single fill‑in‑the‑blank stem and the intended target concept "
            "(e.g., 'increase', 'no initiation', 'mislocalized'), ask ONE concise, specific guiding question. "
            "Avoid jargon and do not reveal the answer."
        )
        user_msg = f"Stem: {stem}\nTarget concept: {target}\nWrite one guiding question (1 sentence)."
        # Use Chat Completions if available; keep code simple
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_msg}],
            temperature=0.3,
            max_tokens=60,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Hint unavailable."

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

# ------------------ Load slide corpus ------------------
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
        if ext in {".pdf"}:
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

# ------------------ Harvest terms from slides ------------------
PROCESS_SUFFIXES = ("tion","sion","sis","ing","ment","ance","lation","folding","assembly","binding","transport","import","export","repair","processing",
                    "replication","transcription","translation")
BIO_HINTS = {"atp","adp","nad","nadh","fadh2","gdp","gtp","rna","dna","mrna","trna","rrna","peptide","polypeptide","protein","enzyme","substrate","product","gradient",
             "phosphate","membrane","mitochond","chloroplast","cytosol","nucleus","ribosome","polymerase","helicase","ligase","kinase","phosphatase",
             "carrier","channel","receptor","complex","chromosome","histone","promoter","tata","cap","tail","exon","intron","spliceosome",
             "ribosomal","anticodon","codon","signal","translocon","tom","tim","srp","srp receptor","pyruvate","glucose","lactate"}

EXCLUDE_GENERIC = {"sequence","sequences","protein","proteins","factor","factors","general","level","intermediate","process","step","steps","thing","stuff"}

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

# ------------------ FITB synthesis (critical-thinking, dynamic) ------------------
def fitb_stems_from_terms(entities: List[str], processes: List[str], topic: str, rng: random.Random) -> List[Dict[str,str]]:
    # Build reasoning stems from slide-derived terms; if thin, use broad fundamentals for the topic
    E = entities[:8] or []
    P = processes[:8] or []

    fundamentals = {
        "dna replication": {
            "stems": [
                ("If helicase cannot unwind DNA, replication at the fork would ______.", "decrease",""),
                ("A loss of ligase activity would cause Okazaki fragments to remain ______.", "increase",""),
                ("If DNA polymerase cannot start, DNA synthesis would show ______.", "no_initiation",""),
                ("Blocking primer removal would cause DNA fragments to ______.", "increase",""),
            ]
        },
        "transcription": {
            "stems": [
                ("If the promoter cannot be recognized, mRNA levels would ______.", "decrease",""),
                ("Blocking RNA polymerase binding would cause initiation to ______.", "no_initiation",""),
                ("If capping is blocked, the transcript’s stability would ______.", "decrease",""),
                ("If splicing fails, intron-containing RNA would ______.", "increase",""),
            ]
        },
        "translation": {
            "stems": [
                ("If the start codon is obscured, initiation of translation would ______.", "no_initiation",""),
                ("If the ribosome cannot translocate, elongation would ______.", "no_elongation",""),
                ("Without a stop codon, translation would ______.", "no_termination",""),
                ("If tRNA charging is impaired, polypeptide synthesis rate would ______.", "decrease",""),
            ]
        },
        "glycolysis": {
            "stems": [
                ("If PFK-1 is inhibited, the level of F1,6BP would ______.", "decrease",""),
                ("If NAD+ is unavailable, glyceraldehyde-3-phosphate would ______.", "increase",""),
                ("If pyruvate kinase is blocked, ATP output from glycolysis would ______.", "decrease",""),
                ("Under anaerobic conditions, pyruvate would ______.", "increase",""),
            ]
        },
        "membrane transport": {
            "stems": [
                ("If an ion channel remains closed, ion flux across the membrane would ______.", "decrease",""),
                ("If the gradient increases, passive transport rate would ______.", "increase",""),
                ("If a pump stops working, the gradient would ______.", "decrease",""),
                ("Blocking aquaporins would cause water movement to ______.", "decrease",""),
            ]
        },
        "protein sorting": {
            "stems": [
                ("If a protein lacks a signal sequence, its cellular location would be ______.", "mislocalized",""),
                ("If SRP recognition fails, cotranslational targeting would ______.", "decrease",""),
                ("If the translocon is blocked, entry into the ER would ______.", "decrease",""),
                ("If signal peptidase cannot act, signal peptides on precursors would ______.", "increase",""),
            ]
        },
        "cell cycle": {
            "stems": [
                ("If the G1 checkpoint fails, entry into S phase would ______.", "increase",""),
                ("If spindle attachment is incomplete, anaphase onset would ______.", "decrease",""),
                ("Loss of p53 function would cause DNA-damaged cells to ______.", "increase",""),
                ("If cyclin levels drop prematurely, progression through the cycle would ______.", "decrease",""),
            ]
        },
    }

    topic_key = topic if topic in fundamentals else None

    stems: List[Tuple[str,str,str]] = []
    # Build from slide terms first
    for e in E:
        stems.append((f"If {e} is missing, the immediate output of the process would ______.", "decrease",""))
        stems.append((f"If {e} accumulates, upstream steps are likely to ______.", "increase",""))
    for p in P:
        stems.append((f"If {p} is blocked, the product of that pathway would ______.", "decrease",""))
        stems.append((f"If {p} speeds up, you would expect the immediate output to ______.", "increase",""))
    # If still thin, extend with topic fundamentals (broad, reasoning-based)
    if len(stems) < 6 and topic_key:
        stems.extend(fundamentals[topic_key]["stems"])

    # Dedup and sample 4 diverse stems
    uniq = []
    seen = set()
    rng.shuffle(stems)
    for s in stems:
        k = s[0].lower()
        if k not in seen:
            uniq.append(s); seen.add(k)
        if len(uniq) == 8: break
    rng.shuffle(uniq)
    pick = uniq[:4] if len(uniq) >= 4 else uniq
    return [{"stem": s, "key": k, "noun": n} for (s,k,n) in pick]

# ------------------ Drag-and-Drop (dynamic, explicit bins & action terms) ------------------
def bins_for_topic(topic: str) -> List[str]:
    if topic == "dna replication":
        return ["First step of replication", "Protein involved in replication", "Intermediate state", "End product of replication"]
    if topic == "transcription":
        return ["Start of transcription", "Protein involved in transcription", "RNA processing event", "End product of transcription"]
    if topic == "translation":
        return ["Start of translation", "Ribosome site/function", "Elongation step", "End product of translation"]
    if topic == "glycolysis":
        return ["Initial substrate/step", "Regulatory enzyme step", "Intermediate metabolite", "End product of glycolysis"]
    if topic == "membrane transport":
        return ["Type of transporter", "Driving force/gradient", "Molecule moved", "Outcome of transport"]
    if topic == "protein sorting":
        return ["Targeting signal", "Targeting machinery", "Transit/intermediate", "Final location/output"]
    if topic == "cell cycle":
        return ["Checkpoint/start event", "Core division step", "Intermediate state", "Outcome of division"]
    # default
    return ["Start event", "Key protein", "Intermediate", "Final product"]

def action_terms_from_slides(entities: List[str], processes: List[str], topic: str, rng: random.Random) -> List[str]:
    # Convert nouns to short action statements, prefer fundamentals if slide terms are vague
    def actify(x: str) -> str:
        t = x.lower()
        if "helicase" in t: return "Helicase unwinds DNA"
        if "polymerase" in t and "rna" in t: return "RNA polymerase synthesizes RNA"
        if "polymerase" in t and "dna" in t: return "DNA polymerase extends DNA"
        if "ligase" in t: return "Ligase seals nicks"
        if "ribosome" in t: return "Ribosome assembles at start codon"
        if "channel" in t or "transporter" in t: return "Transporter moves solute"
        if "pump" in t or "atpase" in t: return "Pump builds gradient"
        if "pfk" in t: return "PFK-1 commits glycolysis"
        if "pyruvate kinase" in t: return "Pyruvate kinase produces ATP"
        if "srp" in t: return "SRP directs ribosome"
        if "translocon" in t or "tom" in t or "tim" in t: return "Translocon imports protein"
        if "signal sequence" in t: return "Signal sequence directs protein"
        if "promoter" in t or "tata" in t: return "Promoter recruits polymerase"
        if "cap" in t: return "mRNA is capped"
        if "tail" in t: return "mRNA gets a poly-A tail"
        if "start codon" in t: return "Start codon begins translation"
        if "stop codon" in t: return "Stop codon ends translation"
        if "atp" in t: return "ATP is synthesized"
        if "nadh" in t: return "NADH is generated"
        if "pyruvate" in t: return "Pyruvate is produced"
        if "glucose" in t: return "Glucose enters glycolysis"
        if "lactate" in t: return "Lactate is formed"
        # fallback: lightly actionify general terms
        if "mrna" in t: return "mRNA is produced"
        if "trna" in t: return "tRNA pairs with codon"
        if "gradient" in t: return "Gradient drives diffusion"
        if "channel" in t: return "Channel opens for ions"
        if "membrane" in t: return "Membrane allows transport"
        return ""

    cands = []
    for x in (entities + processes):
        if not x or x in EXCLUDE_GENERIC: continue
        phrase = actify(x)
        if phrase and phrase not in cands:
            cands.append(phrase)

    # Ensure basic fundamentals if slides are thin
    fundamentals = {
        "dna replication": ["Helicase unwinds DNA","DNA polymerase extends DNA","Ligase seals nicks","Primase lays RNA primer"],
        "transcription": ["Promoter recruits polymerase","RNA polymerase synthesizes RNA","mRNA is capped","mRNA gets a poly-A tail"],
        "translation": ["Ribosome assembles at start codon","tRNA pairs with codon","Polypeptide chain elongates","Stop codon ends translation"],
        "glycolysis": ["Glucose enters glycolysis","PFK-1 commits glycolysis","Pyruvate is produced","ATP is synthesized"],
        "membrane transport": ["Transporter moves solute","Pump builds gradient","Channel opens for ions","Gradient drives diffusion"],
        "protein sorting": ["Signal sequence directs protein","SRP directs ribosome","Translocon imports protein","Signal peptide is removed"],
        "cell cycle": ["Checkpoint allows progression","Chromosomes align at metaphase","Sister chromatids separate","Two daughter cells form"],
    }.get(topic, ["Start event occurs","Key protein acts","Intermediate appears","Product is formed"])

    if len(cands) < 4:
        for f in fundamentals:
            if f not in cands:
                cands.append(f)
            if len(cands) == 4: break

    rng.shuffle(cands)
    return cands[:4]

def map_action_to_bin(action: str, labels: List[str], topic: str) -> str:
    a = action.lower()
    # keyword routing to explicit bins
    def find(label_sub: str) -> str:
        for lab in labels:
            if label_sub.lower() in lab.lower():
                return lab
        return labels[-1]

    if topic == "dna replication":
        if "helicase" in a or "primer" in a: return find("First step")
        if "polymerase" in a or "ligase" in a: return find("Protein involved")
        if "primer" in a or "intermediate" in a or "nick" in a: return find("Intermediate")
        if "daughter" in a or "dna" in a and "produced" in a: return find("End product")
    if topic == "transcription":
        if "promoter" in a or "start codon" in a: return find("Start")
        if "polymerase" in a: return find("Protein involved")
        if "cap" in a or "tail" in a or "splic" in a: return find("processing")
        if "mrna" in a or "rna is produced" in a: return find("End product")
    if topic == "translation":
        if "start codon" in a or "ribosome assembles" in a: return find("Start")
        if "p site" in a or "elongate" in a or "tRNA" in a: return find("Elongation")
        if "stop codon" in a: return find("End product")
        if "polypeptide" in a: return find("End product")
        return find("Ribosome")
    if topic == "glycolysis":
        if "glucose enters" in a or "initial" in a: return find("Initial")
        if "pfk" in a or "commits" in a: return find("Regulatory")
        if "intermediate" in a or "g3p" in a or "fructose" in a or "pyruvate" in a and "produced" not in a: return find("Intermediate")
        if "pyruvate is produced" in a or "atp is synthesized" in a or "nadh" in a: return find("End product")
    if topic == "membrane transport":
        if "transporter" in a or "channel" in a or "pump" in a: return find("Type of transporter")
        if "gradient" in a: return find("Driving force")
        if "solute" in a or "ions" in a: return find("Molecule moved")
        if "diffusion" in a or "outcome" in a: return find("Outcome")
    if topic == "protein sorting":
        if "signal sequence" in a: return find("Targeting signal")
        if "srp" in a or "translocon" in a: return find("machinery")
        if "imports" in a or "transit" in a or "signal peptide is removed" in a: return find("Transit")
        if "final" in a or "location" in a or "protein is in" in a: return find("Final")
    if topic == "cell cycle":
        if "checkpoint" in a or "allows progression" in a: return find("Checkpoint")
        if "align" in a or "separate" in a or "division" in a: return find("Core")
        if "intermediate" in a: return find("Intermediate")
        if "daughter" in a or "two" in a: return find("Outcome")

    # default route
    if "start" in a or "assembles" in a: return labels[0]
    if "protein" in a or "enzyme" in a or "acts" in a: return labels[1]
    if "intermediate" in a or "appears" in a: return labels[2]
    return labels[3]

def build_drag_items(entities: List[str], processes: List[str], topic: str, rng: random.Random) -> Tuple[List[str], List[str], Dict[str,str]]:
    labels = bins_for_topic(topic)
    actions = action_terms_from_slides(entities, processes, topic, rng)
    answers = {a: map_action_to_bin(a, labels, topic) for a in actions}
    return labels, actions, answers

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
    st.session_state.drag_labels, st.session_state.drag_bank, st.session_state.drag_answer = build_drag_items(entities, processes, topic, rng)
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
                if llm_available():
                    st.info(llm_hint(item["stem"], item["key"]))
                else:
                    st.info("Hint unavailable.")
        with col2:
            if st.button("Check", key=f"check_{idx}"):
                ok = matches(u, item["key"], item.get("noun",""))
                if ok:
                    st.success("That’s right! Great work!")
                else:
                    if llm_available():
                        st.info(llm_hint(item["stem"], item["key"]))
                    else:
                        st.info("Consider the immediate, direct effect.")
        with col3:
            if st.button("Reveal", key=f"rev_{idx}"):
                # Show only the plain expected answer
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
    st.markdown(f"## Place the following items in the corresponding categories related to **{topic}**")

    labels = st.session_state.drag_labels
    terms  = st.session_state.drag_bank
    answer = st.session_state.drag_answer

    items_html = "".join([f'<li class="card" draggable="true">{t}</li>' for t in terms])
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
        .drag-over {{ background: #eef5ff; }}
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
          ghostClass: 'ghost',
          chosenClass: 'chosen',
        }};

        const lists = [document.getElementById('bank')];
        LABELS.forEach((_, i) => lists.push(document.getElementById('bin_'+i)));
        lists.forEach(el => new Sortable(el, opts));

        // Prevent text selection highlighting on titles while dragging
        document.querySelectorAll('.title').forEach(el => {{
          el.style.userSelect = 'none';
          el.style.webkitUserSelect = 'none';
          el.style.MozUserSelect = 'none';
        }});

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
