# app.py
# Prompt + Slides → (1) 4× Fill-in-the-Blank (prediction, lenient) + (2) Drag into Topic-Specific Bins
# ----------------------------------------------------------------------------------------------------
# Zero student installs. True drag/drop implemented with pure HTML5 (embedded via st.components.v1.html).
# Activities are generated *from your prompt and slide text* — no hardcoded trivia.
#
# Run:
#   pip install streamlit
#   # For PDFs, choose one that already works in your env:
#   pip install PyPDF2   # or: pip install pypdf
#   streamlit run app.py
# ----------------------------------------------------------------------------------------------------

import os, re, json, pathlib
import streamlit as st
from typing import List, Tuple

# ---------------- App config ----------------
st.set_page_config(page_title="Study Mode", page_icon="📘", layout="wide")
SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm"}

# ---------------- PDF backends (optional) ----------------
PDF_BACKEND = None
try:
    import PyPDF2
    PDF_BACKEND = "PyPDF2"
except Exception:
    try:
        import pypdf
        PDF_BACKEND = "pypdf"
    except Exception:
        PDF_BACKEND = None

# ---------------- File IO ----------------
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
                reader = PyPDF2.PdfReader(f)
                return "\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""
    if PDF_BACKEND == "pypdf":
        try:
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
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

# ---------------- Topic detection (for bin labels & nudges) ----------------
def detect_topic(prompt: str) -> str:
    t = (prompt or "").lower()
    if any(k in t for k in ["transcription","rna pol","promoter","enhancer","tfiid","tbp","splice","mrna","utr"]):
        return "transcription"
    if any(k in t for k in ["translation","ribosome","trna","elongation factor","ef-tu","eftu","ef-g","release factor","shine-dalgarno","kozak"]):
        return "translation"
    if any(k in t for k in ["replication","helicase","primase","ligase","okazaki","leading strand","lagging strand","ssb"]):
        return "replication"
    if any(k in t for k in ["microscope","microscopy","resolution","diffraction","numerical aperture","na","nyquist"]):
        return "microscopy"
    if any(k in t for k in ["rtk","gpcr","erk","mapk","pi3k","akt","ras","raf","phosphatase","kinase"]):
        return "signaling"
    if any(k in t for k in ["cell cycle","cdk","cyclin","checkpoint","apc/c","p53","cohesin","separase"]):
        return "cell_cycle"
    return "generic"

# ---------------- Mining sentences near the prompt ----------------
def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())

def split_sentences(text: str) -> List[str]:
    # Simple split that copes with PDFs reasonably
    parts = re.split(r"(?<=[\.\?\!])\s+|\n+", text)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p and len(p.strip()) > 20]

def relevance(sent: str, q_tokens: List[str]) -> int:
    bag = {}
    for tk in tokenize(sent):
        bag[tk] = bag.get(tk, 0) + 1
    return sum(bag.get(q, 0) for q in q_tokens)

CAUSE_EFFECT_PATTERNS = [
    # (pattern, canonical answer, noun_hint finder)
    (r"\b(increase|elevat\w*|upregulat\w*|activat\w*)\b.+\b(rate|level|amount|production|flux|phosphorylation|initiation)\b", "increase"),
    (r"\b(decrease|reduc\w*|downregulat\w*|inhibit\w*|impair\w*)\b.+\b(rate|level|amount|production|flux|phosphorylation|initiation)\b", "decrease"),
    (r"\b(accumulat\w*)\b.+\b(intermediate|fragment|pre-?mrna|nascent)\b", "accumulate"),
    (r"\b(required|essential|necessary)\b.+\b(for)\b", "decrease"),
]

def extract_relations_from_sentence(sent: str) -> List[Tuple[str,str,str]]:
    """
    Returns list of (stem, canonical_key, noun_hint) derived from sentence.
    """
    outs = []
    s_low = sent.lower()
    # Try directional language
    for pat, key in CAUSE_EFFECT_PATTERNS:
        if re.search(pat, s_low):
            # Make a prediction-style stem from the sentence (light rewrite)
            # Keep it warm: "When X happens, Y would ______."
            # Try to lift a likely noun to serve as 'accumulate' hint
            noun = ""
            m = re.search(r"(pre-?mrna|okazaki fragments?|nascent (?:chains|peptides)|intermediates?)", s_low)
            if m: noun = m.group(0)
            stem = re.sub(r"\s+", " ", sent).strip()
            # soften to “When/If … would ___.” if not already a prediction
            if not stem.endswith(("?", ".")): stem += "."
            stem = f"When this condition holds: {stem} What would you expect to happen?"
            outs.append((stem, "no_change" if key not in {"increase","decrease","accumulate"} else key, noun))
    # If the sentence uses “X leads to Y” style
    if re.search(r"\bleads? to\b|\bresults? in\b|\bcauses?\b", s_low):
        stem = re.sub(r"\s+", " ", sent).strip()
        if not stem.endswith((".", "?")): stem += "."
        stem = f"When this relationship applies: {stem} The immediate effect would ______."
        outs.append((stem, "increase", ""))  # conservative default
    return outs

def mine_prompted_relations(all_text: List[str], prompt: str, max_items=6) -> List[Tuple[str,str,str]]:
    q = tokenize(prompt)
    scored_sents = []
    for doc in all_text:
        for s in split_sentences(doc):
            sc = relevance(s, q)
            if sc > 0:
                scored_sents.append((sc, s))
    scored_sents.sort(reverse=True, key=lambda x: x[0])
    relations = []
    for _, s in scored_sents[:120]:
        relations.extend(extract_relations_from_sentence(s))
        if len(relations) >= max_items:
            break
    return relations[:max_items]

# ---------------- Lenient grading ----------------
UP = {"increase","increases","increased","up","higher","activates","activation","faster","more","↑"}
DOWN = {"decrease","decreases","decreased","down","lower","inhibits","inhibition","slower","less","↓"}
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

# ---------------- Topic-specific drag bins ----------------
def topic_bins(topic: str):
    if topic == "transcription":
        return ["Initiation", "Elongation", "Termination", "Processing"]
    if topic == "translation":
        return ["Initiation", "Elongation", "Termination", "Quality control"]
    if topic == "replication":
        return ["Initiation", "Elongation", "Ligation", "Fork stability"]
    if topic == "microscopy":
        return ["Resolution", "Contrast", "Artifacts", "Sampling"]
    if topic == "signaling":
        return ["Upstream activation", "Pathway inhibition", "Downstream output", "Feedback"]
    if topic == "cell_cycle":
        return ["Entry/Commitment", "Checkpoint hold", "Transition", "Exit/Completion"]
    return ["Increase", "Decrease", "No change", "Accumulates"]

def build_drag_bank_and_answers(topic: str, mined: List[Tuple[str,str,str]]):
    """
    Use mined relations to create concise statements to sort into bins.
    If topic is one of the domain sets, map statements to those conceptual bins.
    Otherwise use Increase/Decrease/No change/Accumulates.
    """
    labels = topic_bins(topic)
    # For domain topics, pre-map some common statements:
    pairs = []
    t = topic
    if t == "transcription":
        pairs.extend([
            ("Promoter clearance efficiency", "Initiation"),
            ("Pol II pause-release rate", "Elongation"),
            ("Readthrough at poly(A) sites", "Termination"),
            ("Unprocessed pre-mRNA species", "Processing"),
        ])
    elif t == "translation":
        pairs.extend([
            ("Start-site selection", "Initiation"),
            ("Elongation speed", "Elongation"),
            ("Release-factor dependent termination", "Termination"),
            ("Incomplete nascent chains", "Quality control"),
        ])
    elif t == "replication":
        pairs.extend([
            ("Origin firing frequency", "Initiation"),
            ("Replication-fork progression", "Elongation"),
            ("Okazaki fragment joining", "Ligation"),
            ("Strand re-annealing at forks", "Fork stability"),
        ])
    elif t == "microscopy":
        pairs.extend([
            ("Minimum resolvable distance (d)", "Resolution"),
            ("Optical sectioning", "Contrast"),
            ("Spherical aberration", "Artifacts"),
            ("Nyquist sampling", "Sampling"),
        ])
    elif t == "signaling":
        pairs.extend([
            ("RTK dimerization/activation", "Upstream activation"),
            ("RAS-GTP lifetime", "Pathway inhibition"),
            ("ERK phosphorylation", "Downstream output"),
            ("Negative feedback to ERK", "Feedback"),
        ])
    elif t == "cell_cycle":
        pairs.extend([
            ("G1→S entry rate", "Entry/Commitment"),
            ("Spindle checkpoint activity", "Checkpoint hold"),
            ("Anaphase onset", "Transition"),
            ("Cohesion removal completion", "Exit/Completion"),
        ])

    # Add mined fallbacks mapped to generic bins if needed
    generic_bins = {"increase":"Increase","decrease":"Decrease","no_change":"No change","accumulate":"Accumulates"}
    for stem, key, noun in mined:
        label = generic_bins.get(key, "No change")
        # Summarize into a short statement for sorting
        short = re.sub(r"^When this.*?:\s*", "", stem, flags=re.IGNORECASE)
        short = re.sub(r"\s+What would you expect.*$", "", short, flags=re.IGNORECASE)
        short = short.strip(". ")
        if len(short) > 80: short = short[:77] + "…"
        pairs.append((short, label))

    # Keep unique statements, cap 6–8 to keep drag task tight
    seen = set(); uniq = []
    for s, lbl in pairs:
        k = (s.lower(), lbl)
        if k not in seen:
            uniq.append((s,lbl)); seen.add(k)

    # choose up to 6–8 items
    bank_pairs = uniq[:8] if len(uniq) >= 6 else uniq
    bank = [s for (s,_) in bank_pairs]
    answers = {s:lbl for (s,lbl) in bank_pairs}
    return labels, bank, answers

# ---------------- UI ----------------
st.title("📘 Prompt → Critical-Thinking Activities")
st.caption("Type a topic. The app searches your slides and builds *fresh* activities each time (no memorization trivia).")

prompt = st.text_input("Enter a topic (e.g., transcription initiation, translation elongation, replication fork dynamics)", value="", placeholder="Type your topic…")

if st.button("Generate"):
    # Load slides once
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)

    all_text = st.session_state.corpus
    topic = detect_topic(prompt)
    mined = mine_prompted_relations(all_text, prompt, max_items=10)

    # Build 4 FITB items from mined relations or topic defaults
    fitb = []
    for (stem, key, noun) in mined:
        fitb.append({"stem": stem, "key": key, "noun": noun})
        if len(fitb) == 4: break

    if len(fitb) < 4:
        # Backfill with topic-appropriate warm stems
        topic_defaults = {
            "transcription": [
                ("When TFIIH helicase activity is reduced, the rate of transcription initiation would ______.", "decrease", ""),
                ("When a strong enhancer contacts the promoter, mRNA output would ______.", "increase", ""),
                ("When Mediator–Pol II contact is disrupted, promoter clearance would ______.", "decrease", ""),
                ("When the poly(A) signal is mutated, unprocessed pre-mRNA species would ______.", "accumulate", "pre-mrna"),
            ],
            "translation": [
                ("When EF-Tu cannot escort aminoacyl-tRNA, the elongation rate would ______.", "decrease", ""),
                ("When the Kozak/Shine-Dalgarno context improves, initiation frequency would ______.", "increase", ""),
                ("When release factors are scarce, termination time per protein would ______.", "increase", ""),
                ("When peptidyl-transferase is blocked, incomplete nascent chains would ______.", "accumulate", "nascent"),
            ],
            "replication": [
                ("When helicase activity is inhibited, replication-fork progression would ______.", "decrease", ""),
                ("When primer supply drops, the number of lagging-strand starts would ______.", "decrease", ""),
                ("When DNA ligase is inactive, unjoined Okazaki fragments would ______.", "accumulate", "okazaki"),
                ("When SSB coverage improves on ssDNA, unwanted re-annealing at forks would ______.", "decrease", ""),
            ],
            "microscopy": [
                ("When numerical aperture (NA) increases, the minimum resolvable distance would ______.", "decrease", ""),
                ("When a shorter wavelength is used, the resolution limit would ______.", "decrease", ""),
                ("When spherical aberration grows, perceived sharpness would ______.", "decrease", ""),
                ("When the confocal pinhole is narrowed moderately, optical sectioning would ______.", "increase", ""),
            ],
            "signaling": [
                ("When an RTK is persistently ligand-bound and dimerized, ERK phosphorylation would ______.", "increase", ""),
                ("When GAP activity on RAS increases, RAS-GTP lifetime would ______.", "decrease", ""),
                ("When PI3K is hyperactive, AKT activation would ______.", "increase", ""),
                ("When a phosphatase targeting ERK is overexpressed, pERK levels would ______.", "decrease", ""),
            ],
            "cell_cycle": [
                ("When APC/C is inhibited by the spindle checkpoint, separase activation would ______.", "decrease", ""),
                ("When Cyclin E–CDK2 activity rises, G1→S entry would ______.", "increase", ""),
                ("When p53 stabilizes after DNA damage, CDK activity would ______.", "decrease", ""),
                ("When cohesion removal is delayed, metaphase duration would ______.", "increase", ""),
            ],
            "generic": [
                ("When a rate-limiting step is enhanced, the amount of product formed would ______.", "increase", ""),
                ("When a core catalytic step is hindered, the overall throughput would ______.", "decrease", ""),
                ("When a terminal processing step is blocked, intermediate species would ______.", "accumulate", "intermediate"),
                ("When a robust compensation route is active, the net change would be ______.", "no_change", ""),
            ],
        }
        for stem, key, noun in topic_defaults.get(topic, topic_defaults["generic"]):
            if len(fitb) == 4: break
            fitb.append({"stem": stem, "key": key, "noun": noun})

    st.session_state.fitb = fitb

    # Build drag bank + answers (topic-specific bins; statements from mined + topic)
    labels, bank, answers = build_drag_bank_and_answers(topic, mined)
    st.session_state.drag_labels = labels
    st.session_state.drag_bank = bank
    st.session_state.drag_answer = answers

    st.success(f"Generated fresh activities for **{topic.replace('_',' ').title()}**.")

# ---------------- Render Activity 1: FITB ----------------
if "fitb" in st.session_state:
    st.markdown("## Activity 1 — Predict the immediate effect")
    st.caption("Answer with **increase**, **decrease**, **no change**, or **accumulates**. Lenient matching is OK.")
    for i, item in enumerate(st.session_state.fitb, start=1):
        u = st.text_input(f"{i}. {item['stem']}", key=f"fitb_{i}")
        if st.button(f"Check {i}", key=f"fitb_check_{i}"):
            ok = matches(u, item["key"], item.get("noun",""))
            st.success("Nice — that matches the expected immediate effect.") if ok else st.error("Not quite — think causally and immediately.")

# ---------------- Render Activity 2: TRUE Drag (HTML5, no installs) ----------------
if all(k in st.session_state for k in ["drag_labels","drag_bank","drag_answer"]):
    st.markdown("---")
    st.markdown("## Activity 2 — Drag the statements into the correct bin")

    labels = st.session_state.drag_labels
    bank   = st.session_state.drag_bank
    answer = st.session_state.drag_answer

    # Build a minimal HTML5 DnD UI with in-iframe scoring
    html_payload = f"""
    <div style="font-family: ui-sans-serif, system-ui; line-height:1.35;">
      <style>
        .bank, .bin {{
          border: 2px dashed #bbb; border-radius: 10px; padding: 10px; min-height: 100px;
          background: #fafafa; margin-bottom: 14px;
        }}
        .bin {{ background: #f6faff; }}
        .title {{ font-weight: 600; margin-bottom: 6px; }}
        .card {{
          background: white; border: 1px solid #ddd; border-radius: 8px;
          padding: 8px 10px; margin: 6px 0; cursor: grab;
          box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        }}
        .zone {{ display:flex; gap:16px; }}
        .left {{ flex: 1; }}
        .right {{ flex: 2; display:grid; grid-template-columns: repeat(2, 1fr); gap:16px; }}
        .ok   {{ color:#0a7; font-weight:600; }}
        .bad  {{ color:#b00; font-weight:600; }}
        button {{
          border-radius: 8px; border: 1px solid #ddd; background:#fff; padding:8px 12px; cursor:pointer;
        }}
      </style>

      <div class="zone">
        <div class="left">
          <div class="title">Bank</div>
          <div id="bank" class="bank" ondrop="drop(event)" ondragover="allow(event)"></div>
        </div>
        <div class="right">
          {"".join([f'<div class="bin" ondrop="drop(event)" ondragover="allow(event)"><div class="title">{lbl}</div><div id="bin_{i}"></div></div>' for i,lbl in enumerate(labels)])}
        </div>
      </div>

      <div style="margin-top:10px;">
        <button onclick="checkBins()">Check bins</button>
        <span id="score" style="margin-left:10px;"></span>
      </div>

      <script>
        const BANK_ITEMS = {json.dumps(bank)};
        const ANSWERS = {json.dumps(answer)};
        function allow(ev) {{ ev.preventDefault(); }}
        function drag(ev)  {{ ev.dataTransfer.setData("text/plain", ev.target.id); }}
        function drop(ev) {{
          ev.preventDefault();
          const id = ev.dataTransfer.getData("text/plain");
          const card = document.getElementById(id);
          const target = ev.target.closest('.bank, .bin');
          if (target) target.appendChild(card);
        }}
        function renderBank() {{
          const b = document.getElementById("bank");
          b.innerHTML = "";
          BANK_ITEMS.forEach((txt, idx) => {{
            const c = document.createElement("div");
            c.className = "card";
            c.id = "card_" + idx;
            c.draggable = true;
            c.ondragstart = drag;
            c.textContent = txt;
            b.appendChild(c);
          }});
        }}
        function currentBins() {{
          const bins = {{}};
          const titles = {json.dumps(labels)};
          titles.forEach((t, i) => {{
            const holder = document.getElementById("bin_" + i);
            const items = [];
            holder.querySelectorAll(".card").forEach(c => items.push(c.textContent));
            bins[t] = items;
          }});
          return bins;
        }}
        function checkBins() {{
          const bins = currentBins();
          let total = 0, correct = 0;
          for (const [stmt, want] of Object.entries(ANSWERS)) {{
            total += 1;
            let got = "Bank";
            for (const [label, items] of Object.entries(bins)) {{
              if (items.includes(stmt)) {{ got = label; break; }}
            }}
            if (got === want) correct += 1;
          }}
          const score = document.getElementById("score");
          if (total === 0) {{
            score.innerHTML = "<span class='bad'>Drag items into bins first.</span>";
          }} else if (correct === total) {{
            score.innerHTML = "<span class='ok'>All bins correct! 🎉</span>";
          }} else {{
            score.innerHTML = "<span class='bad'>" + correct + "/" + total + " correct — adjust and try again.</span>";
          }}
        }}
        renderBank();
      </script>
    </div>
    """
    st.components.v1.html(html_payload, height=520, scrolling=True)
