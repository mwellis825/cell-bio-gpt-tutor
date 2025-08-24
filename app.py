# app.py
# Universal prompt → activities from your slides
# (1) 4× Fill‑in‑the‑Blank (application, lenient, randomized from slide‑matched lines)
# (2) Drag into dynamic bins inferred from nearby slide phrases (SortableJS; no installs)
# ----------------------------------------------------------------------------------------------------

import os, re, json, pathlib, random, time
import streamlit as st
from typing import List, Tuple, Dict

st.set_page_config(page_title="Study Mode (Universal)", page_icon="📘", layout="wide")
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
STOP = {"the","and","for","that","with","this","from","into","are","was","were","has","have","had","can","will","would","could","should",
        "a","an","of","in","on","to","by","as","at","or","be","is","it","its","their","our","your","if","when","then","than","but"}

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

# -------- Mine prompt-matched sentences & infer bins --------
def collect_candidates(corpus: List[str], prompt: str, top_docs=6, max_sents_per_doc=500) -> List[str]:
    q = tokens_nostop(prompt)
    if not q: return []
    doc_scores = []
    for doc in corpus:
        sents = split_sentences(doc)
        sc = sum(relevance(s, q) for s in sents[:max_sents_per_doc])
        if sc > 0:
            doc_scores.append((sc, sents))
    doc_scores.sort(reverse=True, key=lambda x: x[0])
    matched = []
    for _, sents in doc_scores[:top_docs]:
        for s in sents:
            if relevance(s, q) > 0:
                matched.append(s)
    return matched

def infer_bins(matched_sents: List[str], prompt: str) -> List[str]:
    # collect frequent 1–3grams from matched sentences + prompt tokens
    vocab = {}
    for s in matched_sents:
        toks = tokens_nostop(s)
        for t in toks:
            vocab[t] = vocab.get(t, 0) + 1
        for i in range(len(toks)-1):
            big = f"{toks[i]} {toks[i+1]}"
            vocab[big] = vocab.get(big, 0) + 1
        for i in range(len(toks)-2):
            tri = f"{toks[i]} {toks[i+1]} {toks[i+2]}"
            vocab[tri] = vocab.get(tri, 0) + 1

    # seed with prompt tokens to avoid "generic" labels
    for t in tokens_nostop(prompt):
        vocab[t] = vocab.get(t, 0) + 3

    pref = sorted(vocab.items(), key=lambda kv: (len(kv[0].split())>=2, kv[1]), reverse=True)
    labels = []
    for phrase, _ in pref:
        if any(ch.isalpha() for ch in phrase) and 3 <= len(phrase) <= 36:
            title = re.sub(r"\b([a-z])", lambda m: m.group(1).upper() if m.start()==0 or phrase[m.start()-1]==" " else m.group(1), phrase)
            title = title.replace("Mrna","mRNA").replace("Dna","DNA").replace("Rna","RNA")
            if title.lower() not in {l.lower() for l in labels} and len(labels) < 4:
                labels.append(title)
    # if still short, use prompt-derived placeholders (never the word "Generic")
    if len(labels) < 4:
        base = [w.capitalize() for w in tokens_nostop(prompt)[:4]]
        for b in base:
            if b and b.lower() not in {x.lower() for x in labels} and len(labels) < 4:
                labels.append(b)
    # final pad to exactly 4
    i = 1
    while len(labels) < 4:
        pad = f"{tokens_nostop(prompt)[:1][0].capitalize() if tokens_nostop(prompt) else 'Topic'} {i}"
        if pad.lower() not in {x.lower() for x in labels}:
            labels.append(pad)
        i += 1
    return labels[:4]

def map_statements_to_bins(statements: List[str], labels: List[str]) -> Dict[str,str]:
    mapping = {}
    low_labels = [l.lower() for l in labels]
    for s in statements:
        s_low = s.lower()
        # choose the label whose words appear most in s
        best_i, best_score = 0, -1
        for i, lab in enumerate(low_labels):
            score = sum(1 for w in lab.split() if w in s_low)
            if score > best_score:
                best_score, best_i = score, i
        mapping[s] = labels[best_i]
    return mapping

# -------- Extract causal stems (application tone) --------
CAUSE_EFFECT_PATTERNS = [
    (r"\b(increase|elevat\w*|upregulat\w*|enhanc\w*|stabiliz\w*)\b.+\b(rate|level|amount|production|flux|stability|binding|interaction|initiation|elongation)\b", "increase"),
    (r"\b(decrease|reduc\w*|downregulat\w*|destabiliz\w*|impair\w*|inhibit\w*)\b.+\b(rate|level|amount|production|flux|stability|binding|interaction|initiation|elongation)\b", "decrease"),
    (r"\b(accumulat\w*)\b.+\b(intermediate|unprocessed|misfolded|aggregate|fragment|pre-?mrna|nascent)\b", "accumulate"),
    (r"\b(required|essential|necessary)\b.+\b(for)\b", "decrease"),
]

def extract_relations_from_sentence(sent: str) -> List[Tuple[str,str,str]]:
    outs = []
    s_low = sent.lower()
    for pat, key in CAUSE_EFFECT_PATTERNS:
        if re.search(pat, s_low):
            noun = ""
            m = re.search(r"(intermediate|unprocessed|misfolded proteins?|aggregates?|pre-?mrna|nascent)", s_low)
            if m: noun = m.group(0)
            stem = re.sub(r"\s+", " ", sent).strip()
            if not stem.endswith((".", "?")): stem += "."
            stem = f"When this condition holds: {stem} What would you expect to happen?"
            outs.append((stem, key if key in {"increase","decrease","accumulate"} else "no_change", noun))
    if re.search(r"\bleads? to\b|\bresults? in\b|\bcauses?\b", s_low):
        stem = re.sub(r"\s+", " ", sent).strip()
        if not stem.endswith((".", "?")): stem += "."
        stem = f"When this relationship applies: {stem} The immediate effect would ______."
        outs.append((stem, "increase", ""))
    return outs

# -------- Lenient grading --------
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

# -------- Build activities from matched sentences --------
def build_fitb(matched_sents: List[str], prompt: str, rng: random.Random) -> List[Dict[str,str]]:
    pool = []
    for s in matched_sents[:800]:
        pool.extend(extract_relations_from_sentence(s))
    # If mining is thin, template from prompt tokens (still topic-specific; never generic)
    if len(pool) < 4:
        key = "increase"
        topic_words = tokens_nostop(prompt)[:3] or ["output"]
        stems = [
            f"When {topic_words[0]} availability rises, the immediate {topic_words[-1]} would ______.",
            f"When a step central to {topic_words[0]} is slowed, the {topic_words[-1]} would ______.",
            f"When processing related to {topic_words[-1]} is blocked, intermediate forms would ______.",
            f"When an upstream factor essential for {topic_words[0]} is missing, the {topic_words[-1]} would ______.",
        ]
        pool.extend([(s, key, topic_words[-1]) for s in stems])

    # de-dup, shuffle, sample 4
    seen = set(); uniq = []
    for stem, key, noun in pool:
        k = stem.lower()
        if k not in seen:
            uniq.append((stem,key,noun)); seen.add(k)
    rng.shuffle(uniq)
    pick = uniq[:4]
    return [{"stem": s, "key": k, "noun": n} for (s,k,n) in pick]

def build_drag(matched_sents: List[str], prompt: str, rng: random.Random) -> Tuple[List[str], List[str], Dict[str,str]]:
    # short statements from matched text
    statements = []
    for s in matched_sents[:600]:
        short = re.sub(r"\s+", " ", s).strip()
        short = re.sub(r"^Figure\s*\d+[:\.\-]\s*", "", short, flags=re.IGNORECASE)
        if len(short) > 100: short = short[:97] + "…"
        if 28 <= len(short) <= 100:
            statements.append(short)
    if len(statements) < 6:
        # prompt-derived process phrases (topic-specific placeholders; never "Generic")
        words = tokens_nostop(prompt) or ["Topic"]
        base = words[0].capitalize()
        statements += [
            f"{base}: upstream factor/change",
            f"{base}: key process step",
            f"{base}: intermediate/evidence observed",
            f"{base}: immediate output",
            f"{base}: rate/throughput",
            f"{base}: completion/finishing step",
        ]
    # dedupe, shuffle, sample 6–8
    statements = list(dict.fromkeys(statements))
    rng.shuffle(statements)
    bank = statements[: max(6, min(8, len(statements))) ]

    # infer labels from matched sentences + prompt
    labels = infer_bins(matched_sents, prompt)
    answers = map_statements_to_bins(bank, labels)
    return labels, bank, answers

# -------- UI --------
st.title("📘 Prompt → Critical‑Thinking Activities (Universal)")
st.caption("Type any topic. The app searches your slides for that topic and builds fresh, application‑style practice (no trivia).")

prompt = st.text_input("Enter a topic (anything in your slides)", value="", placeholder="e.g., ATP generation, protein structure, membrane transport…")

if st.button("Generate"):
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)

    if not st.session_state.corpus:
        msg = "I couldn't read any slide text. If your slides are PDFs, install **PyPDF2** or **pypdf** on the server."
        if PDF_BACKEND is None:
            st.error(msg)
        else:
            st.warning("No extractable text was found in /slides. Check that PDFs contain selectable text (not just images).")
    rng = random.Random(new_seed())
    matched = collect_candidates(st.session_state.corpus, prompt)
    st.session_state.fitb = build_fitb(matched, prompt, rng)
    st.session_state.drag_labels, st.session_state.drag_bank, st.session_state.drag_answer = build_drag(matched, prompt, rng)
    st.success("Built fresh activities from your slides for this prompt.")

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
        const ANSWERS = {json.dumps(answer)};
        const LABELS = {json.dumps(labels)};

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
          for (const [stmt, want] of Object.entries(ANSWERS)) {{
            total += 1;
            let got = "Bank";
            for (const [label, items] of Object.entries(bins)) {{
              if (items.includes(stmt)) {{ got = label; break; }}
            }}
            if (got === want) correct += 1;
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
    st.components.v1.html(html, height=680, scrolling=True)
