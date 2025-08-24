# app.py
import os, re, json, pathlib, random, time
import streamlit as st
from typing import List, Tuple, Dict

st.set_page_config(page_title="Study Mode — Critical Thinking", page_icon="📘", layout="wide")
SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm"}

def new_seed() -> int:
    return int(time.time() * 1000) ^ random.randint(0, 1_000_000)

# -------- PDF backends (optional) --------
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

# -------- File IO --------
def read_text(path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""

def read_pdf(path: str) -> str:
    if PDF_BACKEND == "PyPDF2":
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)  # type: ignore
                return "\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""
    if PDF_BACKEND == "pypdf":
        try:
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)  # type: ignore
                return "\n".join([(p.extract_text() or "") for p in reader.pages])
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

# -------- Tokenization & helpers --------
STOP = {
    "the","and","for","that","with","this","from","into","are","was","were","has","have","had","can","will","would","could","should",
    "a","an","of","in","on","to","by","as","at","or","be","is","it","its","their","our","your","if","when","then","than","but",
    "we","you","they","which","these","those","there","here","such","may","might","also"
}
META = {"lecture","slide","slides","figure","fig","table","exam","objective","objectives","learning","homework","quiz","next","previous","today"}

def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())

def tokens_nostop(s: str) -> List[str]:
    return [t for t in tokenize(s) if t not in STOP and len(t) > 2]

def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p and len(p.strip()) > 30]

def relevance(sent: str, q_tokens: List[str]) -> int:
    bag = {}
    for tk in tokenize(sent):
        bag[tk] = bag.get(tk, 0) + 1
    score = sum(bag.get(q, 0) for q in q_tokens)
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

# -------- Domain-ish phrase harvesting --------
PROCESS_SUFFIXES = ("tion","sion","sis","ing","ment","ance","ation","lation","lation","folding","assembly","binding","transport","import","export")
BIO_HINTS = {"atp","adp","nad","nadh","fadh2","gdp","gtp","rna","dna","mrna","trna","rrna","peptide","protein","enzyme","substrate","product","grad","gradient","phosphate","membrane","mitochond","chloroplast","cytosol","nucleus","ribosome","polymerase","helicase","ligase","kinase","phosphatase","carrier","channel","receptor","complex"}

def clean_phrase(p: str) -> str:
    p = re.sub(r"\s+", " ", p).strip(" -—:;,.")
    # Remove obvious meta phrases
    if any(m in p.lower() for m in META): return ""
    # length bounds
    if len(p) < 3 or len(p) > 60: return ""
    # must include alpha
    if not any(ch.isalpha() for ch in p): return ""
    return p

def harvest_terms(sentences: List[str], prompt: str) -> Tuple[List[str], List[str]]:
    """Return (entities, processes) inferred from matched sentences and the prompt tokens."""
    ents = {}
    procs = {}

    def add_terms(toks: List[str]):
        # 1-grams
        for t in toks:
            if t in STOP or t in META: continue
            if any(m in t for m in META): continue
            if len(t) < 3: continue
            if t.isdigit(): continue
            if any(ch.isalpha() for ch in t) is False: continue
            if t.endswith(PROCESS_SUFFIXES) or t in BIO_HINTS:
                procs[t] = procs.get(t, 0) + 1
            else:
                ents[t] = ents.get(t, 0) + 1
        # 2-grams / 3-grams
        for n in (2,3):
            for i in range(len(toks)-n+1):
                ng = " ".join(toks[i:i+n])
                if any(m in ng for m in META): continue
                if len(ng) < 5 or len(ng) > 40: continue
                # heuristic: if ends with process-y suffix, call it a process
                if any(ng.endswith(suf) for suf in PROCESS_SUFFIXES):
                    procs[ng] = procs.get(ng, 0) + 1
                else:
                    ents[ng] = ents.get(ng, 0) + 1

    for s in sentences:
        s_low = s.lower()
        if any(m in s_low for m in META):  # skip meta lines
            continue
        toks = tokens_nostop(s_low)
        add_terms(toks)

    # Seed with prompt tokens (to keep focus) — but never meta
    add_terms(tokens_nostop(prompt.lower()))

    # Rank by frequency then length (prefer concise but specific)
    entities = sorted(ents.keys(), key=lambda k: (ents[k], len(k)), reverse=True)
    processes = sorted(procs.keys(), key=lambda k: (procs[k], len(k)), reverse=True)

    # Clean and trim
    entities = [clean_phrase(e) for e in entities]
    processes = [clean_phrase(p) for p in processes]
    entities = [e for e in entities if e][:20]
    processes = [p for p in processes if p][:20]
    return entities, processes

# -------- FITB synthesis (application-only, varied) --------
UP = {"increase","increases","increased","up","higher","stabilizes","stabilize","stabilized","faster","more","↑","improves","greater"}
DOWN = {"decrease","decreases","decreased","down","lower","destabilizes","destabilize","destabilized","slower","less","↓","reduces","reduced"}
NOCH = {"no change","unchanged","same","neutral","nc","~"}
ACCU = {"accumulate","accumulates","accumulated","builds up","build up","piles up","pile up","amass","gathers"}

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+"," ", s)
    s = s.replace("’","'")
    return s

def matches(user: str, key: str, noun: str) -> bool:
    u = norm(user)
    if key == "increase":   return any(x in u for x in UP)
    if key == "decrease":   return any(x in u for x in DOWN)
    if key == "no_change":  return any(x in u for x in NOCH)
    if key == "accumulate": return (any(x in u for x in ACCU) or (noun and noun in u))
    return key in u

def synthesize_fitb(entities: List[str], processes: List[str], rng: random.Random) -> List[Dict[str,str]]:
    # Guard rails: ensure we have some slots
    e = entities[:8] or ["substrate", "cofactor"]
    p = processes[:8] or ["processing", "assembly"]

    def pick(lst): return rng.choice(lst)

    templates = [
        lambda: (f"When {pick(e)} availability rises, immediate output of {pick(p)} would ______.", "increase",""),
        lambda: (f"When a step in {pick(p)} is slowed, the near‑term product formation would ______.", "decrease",""),
        lambda: (f"When {pick(e)} is missing but the rest of the pathway is intact, the net change in final output would be ______.", "decrease",""),
        lambda: (f"When late‑stage processing in {pick(p)} is blocked, intermediate forms would ______.", "accumulate","intermediate"),
        lambda: (f"When quality control during {pick(p)} becomes more efficient, misprocessed products would ______.", "decrease",""),
        lambda: (f"When targeting of {pick(e)} to its site is improved, off‑target effects would ______.", "decrease",""),
        lambda: (f"When stabilization of a key complex in {pick(p)} improves, throughput would ______.", "increase",""),
        lambda: (f"When parallel backup routes compensate for {pick(p)}, the immediate output would show ______.", "no_change",""),
    ]

    # Build 4 varied, deduped items
    stems = set(); items = []
    rng.shuffle(templates)
    for make in templates:
        stem, key, noun = make()
        if stem.lower() not in stems:
            stems.add(stem.lower())
            items.append({"stem": stem, "key": key, "noun": noun})
        if len(items) == 4: break
    # Final safety: if <4, fill with different patterns
    while len(items) < 4:
        stem, key, noun = (f"When {pick(e)} engages earlier, immediate output of {pick(p)} would ______.","increase","")
        if stem.lower() not in stems:
            stems.add(stem.lower()); items.append({"stem": stem, "key": key, "noun": noun})
    return items

# -------- Drag bins: semantic & adaptive --------
def make_bins(prompt: str, entities: List[str], processes: List[str]) -> List[str]:
    topic = (tokens_nostop(prompt)[:1] or ["Topic"])[0].capitalize()
    labels = [
        f"{topic}: Upstream factors",
        f"{topic}: Core step",
        f"{topic}: Intermediates/Evidence",
        f"{topic}: Immediate output",
    ]
    return labels

def build_drag_items(entities: List[str], processes: List[str], rng: random.Random) -> Tuple[List[str], List[str], Dict[str,str]]:
    labels = ["Upstream factors", "Core step", "Intermediates/Evidence", "Immediate output"]
    # Curate candidates (filter meta/noise)
    ents = [x for x in entities if x and x not in META][:12]
    procs = [x for x in processes if x and x not in META][:12]

    # Compose readable cards
    bank = []
    for s in ents[:6]:
        bank.append(f"Upstream: {s}")
    for pr in procs[:6]:
        # prefer gerund/nominal forms
        txt = pr
        bank.append(f"Process: {txt}")
    # Pad if thin
    while len(bank) < 6:
        bank.append(f"Process: {rng.choice(['assembly','processing','transport','binding'])}")
    bank = bank[:8]

    # Map to bins
    answers = {}
    for card in bank:
        if card.startswith("Upstream:"):
            answers[card] = labels[0]
        elif card.startswith("Process:"):
            answers[card] = labels[1]
        else:
            # fallback based on keywords
            if any(k in card.lower() for k in ("intermediate","nascent","precursor","fragment")):
                answers[card] = labels[2]
            else:
                answers[card] = labels[3]

    # Prepend topic to labels for display
    display_labels = make_bins(" ".join(entities[:1] + processes[:1]) or "Topic", entities, processes)
    return display_labels, bank, answers

# -------------------------------- UI --------------------------------
st.title("📘 Prompt → Critical‑Thinking Activities")
st.caption("Type any topic. The app mines your slides for that topic’s vocabulary, then synthesizes new, application‑style activities at that breadth/depth (no trivia, no meta).")

prompt = st.text_input("Enter a topic (anything in your slides)", value="", placeholder="e.g., energetics, protein structure, membrane transport…")

if st.button("Generate"):
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)

    if not st.session_state.corpus:
        if PDF_BACKEND is None:
            st.error("I couldn't read slide text. Install **PyPDF2** or **pypdf** on the server to extract PDF text.")
        else:
            st.warning("No extractable text found in /slides. Confirm your PDFs contain selectable text (not scanned images).")

    matched = collect_prompt_matched(st.session_state.corpus, prompt)
    rng = random.Random(new_seed())
    entities, processes = harvest_terms(matched, prompt)

    st.session_state.fitb = synthesize_fitb(entities, processes, rng)
    st.session_state.drag_labels, st.session_state.drag_bank, st.session_state.drag_answer = build_drag_items(entities, processes, rng)
    st.success("Generated fresh activities based on your slides’ vocabulary and your prompt.")

# -------- Activity 1: FITB --------
if "fitb" in st.session_state:
    st.markdown("## Activity 1 — Predict the immediate effect")
    st.caption("Answer with **increase**, **decrease**, **no change**, or **accumulates**. Lenient matching is OK.")
    for i, item in enumerate(st.session_state.fitb, start=1):
        u = st.text_input(f"{i}. {item['stem']}", key=f"fitb_{i}")
        if st.button(f"Check {i}", key=f"fitb_check_{i}"):
            ok = matches(u, item["key"], item.get("noun",""))
            if ok:
                st.success("Nice — that matches the immediate outcome.")
            else:
                st.error("Not quite — think about the direct, near‑term effect of that change.")

# -------- Activity 2: Drag‑into‑Bins via SortableJS (CDN) --------
if all(k in st.session_state for k in ["drag_labels","drag_bank","drag_answer"]):
    st.markdown("---")
    st.markdown("## Activity 2 — Drag the statements into the correct bin")
    labels = st.session_state.drag_labels
    bank   = st.session_state.drag_bank
    answer = st.session_state.drag_answer

    bins_html = "".join([
        f"""
        <div class="bin">
          <div class="title">{lbl}</div>
          <ul id="bin_{i}" class="droplist"></ul>
        </div>
        """ for i,lbl in enumerate(labels)
    ])
    items_html = "".join([f'<li class="card">{txt}</li>' for txt in bank])

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"></script>
      <style>
        .bank, .bin {{
          border: 2px dashed #bbb; border-radius: 10px; padding: 12px; min-height: 140px;
          background: #fafafa; margin-bottom: 14px;
        }}
        .bin {{ background: #f6faff; }}
        .droplist {{ list-style: none; margin: 0; padding: 0; min-height: 100px; }}
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
          <ul id="bank" class="bank droplist">
            {items_html}
          </ul>
        </div>
        <div class="right">
          {bins_html}
        </div>
      </div>
      <div style="margin-top:10px;">
        <button id="check">Check bins</button>
        <span id="score" style="margin-left:10px;"></span>
      </div>

      <script>
        const LABELS_SHOWN = {json.dumps(labels)};
        const LABELS_MAP = ["Upstream factors","Core step","Intermediates/Evidence","Immediate output"];
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
        LABELS_SHOWN.forEach((_, i) => lists.push(document.getElementById('bin_'+i)));
        lists.forEach(el => new Sortable(el, opts));

        function readBins() {{
          const bins = {{}};
          LABELS_SHOWN.forEach((label, i) => {{
            const ul = document.getElementById('bin_'+i);
            const items = Array.from(ul.querySelectorAll('.card')).map(li => li.textContent.trim());
            bins[label] = items;
          }});
          return bins;
        }}

        document.getElementById('check').addEventListener('click', () => {{
          const bins = readBins();
          let total = 0, correct = 0;
          for (const [stmt, want] of Object.entries(ANSWERS)) {{
            total += 1;
            let got = "Bank";
            for (const [label, items] of Object.entries(bins)) {{
              if (items.includes(stmt)) {{ got = label; break; }}
            }}
            // Reduce label to canonical bin for comparison
            const canonGot = LABELS_MAP.find(m => label.startsWith(m.split(':')[0])) || label;
            const canonWant = want;
            if (got && want && got === want) correct += 1;
          }}
          const score = document.getElementById('score');
          if (total === 0) {{
            score.innerHTML = "<span class='bad'>Drag items into bins first.</span>";
          }} else if (correct === total) {{
            score.innerHTML = "<span class='ok'>All bins correct! 🎉</span>";
          }} else {{
            score.innerHTML = "<span class='bad'>" + correct + "/" + total + " correct — adjust and try again.</span>";
          }}
        }});
      </script>
    </body>
    </html>
    """
    st.components.v1.html(html, height=700, scrolling=True)
