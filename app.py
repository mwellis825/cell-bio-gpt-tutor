# app.py
# Prompt + Slides → (1) 4× Fill‑in‑the‑Blank (application, lenient) + (2) Drag into Topic‑Specific Bins
# ----------------------------------------------------------------------------------------------------
# No student installs. Drag/drop implemented with pure HTML5 inside a Streamlit component.
# Activities are generated from your prompt and mined slide text; topic bins change with the prompt.
#
# Run:
#   pip install streamlit
#   # PDF parser (choose one):
#   pip install PyPDF2    # or: pip install pypdf
#   streamlit run app.py
# ----------------------------------------------------------------------------------------------------

import os, re, json, pathlib, random
import streamlit as st
from typing import List, Tuple, Dict

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

# ---------------- Utilities ----------------
def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())

def corpus_vocab(texts: List[str]) -> set:
    vocab = set()
    for t in texts:
        for tk in tokenize(t):
            if len(tk) > 2:
                vocab.add(tk)
    return vocab

# ---------------- Topic detection (drives bins) ----------------
def detect_topic(prompt: str) -> str:
    t = (prompt or "").lower()
    if any(k in t for k in ["protein structure","primary structure","secondary structure","tertiary structure","quaternary structure","alpha helix","β-sheet","beta sheet","motif","domain","disulfide","hydrophobic core"]):
        return "protein_structure"
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

# ---------------- Mine sentences near the prompt ----------------
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

# Higher-level biology patterns (surface-level causality)
CAUSE_EFFECT_PATTERNS = [
    (r"\b(increase|elevat\w*|upregulat\w*|enhanc\w*|stabiliz\w*)\b.+\b(rate|level|amount|production|flux|stability|binding|interaction)\b", "increase"),
    (r"\b(decrease|reduc\w*|downregulat\w*|destabiliz\w*|impair\w*|inhibit\w*)\b.+\b(rate|level|amount|production|flux|stability|binding|interaction)\b", "decrease"),
    (r"\b(accumulat\w*)\b.+\b(intermediate|unprocessed|misfolded|aggregate|fragment)\b", "accumulate"),
    (r"\b(required|essential|necessary)\b.+\b(for)\b", "decrease"),
]

def extract_relations_from_sentence(sent: str) -> List[Tuple[str,str,str]]:
    outs = []
    s_low = sent.lower()
    for pat, key in CAUSE_EFFECT_PATTERNS:
        if re.search(pat, s_low):
            noun = ""
            m = re.search(r"(intermediate|unprocessed|misfolded proteins?|aggregates?)", s_low)
            if m: noun = m.group(0)
            stem = re.sub(r"\s+", " ", sent).strip()
            if not stem.endswith((".", "?")): stem += "."
            # Friendly prediction format
            stem = f"When this condition holds: {stem} What would you expect to happen?"
            outs.append((stem, key if key in {"increase","decrease","accumulate"} else "no_change", noun))
    if re.search(r"\bleads? to\b|\bresults? in\b|\bcauses?\b", s_low):
        stem = re.sub(r"\s+", " ", sent).strip()
        if not stem.endswith((".", "?")): stem += "."
        stem = f"When this relationship applies: {stem} The immediate effect would ______."
        outs.append((stem, "increase", ""))
    return outs

def mine_prompted_relations(all_text: List[str], prompt: str, max_items=8) -> List[Tuple[str,str,str]]:
    q = tokenize(prompt)
    if not q: return []
    # rank documents
    doc_scores = []
    for doc in all_text:
        sents = split_sentences(doc)
        sc = sum(relevance(s, q) for s in sents[:300])
        if sc > 0:
            doc_scores.append((sc, sents))
    doc_scores.sort(reverse=True, key=lambda x: x[0])

    relations = []
    for _, sents in doc_scores[:5]:
        for s in sents[:400]:
            if relevance(s, q) <= 0:
                continue
            relations.extend(extract_relations_from_sentence(s))
            if len(relations) >= max_items:
                break
        if len(relations) >= max_items:
            break

    # dedupe by stem
    seen = set(); uniq = []
    for stem, key, noun in relations:
        k = stem.lower()
        if k not in seen:
            uniq.append((stem,key,noun)); seen.add(k)
    return uniq[:max_items]

# ---------------- Lenient grading ----------------
UP = {"increase","increases","increased","up","higher","stabilizes","stabilize","stabilized","faster","more","↑"}
DOWN = {"decrease","decreases","decreased","down","lower","destabilizes","destabilize","destabilized","slower","less","↓"}
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

# ---------------- Topic-specific bins (adaptive) ----------------
def topic_bins(topic: str):
    if topic == "protein_structure":
        return ["Primary", "Secondary", "Tertiary", "Quaternary"]
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

# ---------------- Build content from slides with surface-level application ----------------
def build_fitb(topic: str, mined: List[Tuple[str,str,str]], vocab: set) -> List[Dict[str,str]]:
    # Prefer mined stems; keep them warm and application focused
    items = []
    for (stem, key, noun) in mined:
        items.append({"stem": stem, "key": key, "noun": noun})
        if len(items) == 4: break

    if len(items) >= 4:
        return items[:4]

    # Topic-appropriate, surface-level backfill (no niche factors; filter by slide vocab)
    candidates_by_topic = {
        "protein_structure": [
            ("When hydrophobic side chains are buried more effectively, overall folding stability would ______.","increase",""),
            ("When many prolines interrupt an alpha helix, helix stability would ______.","decrease",""),
            ("When correct disulfide bonds form within one chain, tertiary structure stability would ______.","increase",""),
            ("When subunits fail to assemble, quaternary structure formation would ______.","decrease",""),
        ],
        "transcription": [
            ("When promoter access improves, the rate of transcription initiation would ______.","increase",""),
            ("When elongation is slowed by pausing, overall mRNA output would ______.","decrease",""),
            ("When termination is inefficient, unprocessed transcripts would ______.","accumulate","transcripts"),
            ("When RNA processing is efficient, mature mRNA levels would ______.","increase",""),
        ],
        "translation": [
            ("When start-site recognition improves, initiation frequency would ______.","increase",""),
            ("When elongation is hindered, protein synthesis rate would ______.","decrease",""),
            ("When termination is delayed, stalled ribosome–nascent chains would ______.","accumulate","nascent"),
            ("When quality control is active, misincorporation errors would ______.","decrease",""),
        ],
        "replication": [
            ("When origin firing is reduced, DNA synthesis initiation events would ______.","decrease",""),
            ("When fork progression is supported, time to complete replication would ______.","decrease",""),
            ("When ligation is impaired, short DNA fragments would ______.","accumulate","fragments"),
            ("When ssDNA is protected, unwanted re-annealing would ______.","decrease",""),
        ],
        "microscopy": [
            ("When numerical aperture increases, the minimum resolvable distance would ______.","decrease",""),
            ("When wavelength is shorter, the resolution limit would ______.","decrease",""),
            ("When aberrations grow, image sharpness would ______.","decrease",""),
            ("When optical sectioning improves, out-of-focus blur would ______.","decrease",""),
        ],
        "signaling": [
            ("When a receptor is more active, downstream pathway output would ______.","increase",""),
            ("When a pathway inhibitor is present, downstream activation would ______.","decrease",""),
            ("When feedback removal occurs, the pathway's output would ______.","increase",""),
            ("When a key phosphatase increases, the level of phosphorylation would ______.","decrease",""),
        ],
        "cell_cycle": [
            ("When a checkpoint engages, the next cell-cycle transition would ______.","decrease",""),
            ("When cyclin activity rises, entry into the next phase would ______.","increase",""),
            ("When damage is unresolved, progression to the next phase would ______.","decrease",""),
            ("When cohesion removal is delayed, time spent before separation would ______.","increase",""),
        ],
        "generic": [
            ("When a rate-limiting step is helped, the amount of product formed would ______.","increase",""),
            ("When a core step is hindered, throughput would ______.","decrease",""),
            ("When a late processing step is blocked, intermediates would ______.","accumulate","intermediates"),
            ("When a parallel backup route is strong, the net change would be ______.","no_change",""),
        ],
    }
    # Filter overly specific stems if their keywords aren't in slide vocab (keep beginner tone)
    def ok_for_vocab(stem: str) -> bool:
        toks = [t for t in tokenize(stem) if len(t) > 3]
        hits = sum(1 for t in toks if t in vocab)
        return hits >= max(1, len(toks)//12)  # very lenient

    for stem, key, noun in candidates_by_topic.get(topic, candidates_by_topic["generic"]):
        if ok_for_vocab(stem):
            items.append({"stem": stem, "key": key, "noun": noun})
        if len(items) == 4:
            break
    while len(items) < 4:
        items.append({"stem":"When the helpful step improves, the immediate output would ______.","key":"increase","noun":""})
    return items[:4]

def build_drag(topic: str, mined: List[Tuple[str,str,str]]) -> Tuple[List[str], List[str], Dict[str,str]]:
    labels = topic_bins(topic)
    # Domain seed statements (beginner-friendly)
    seeds = {
        "protein_structure": [
            ("Amino‑acid sequence (order of residues)", "Primary"),
            ("Alpha‑helix / beta‑sheet content", "Secondary"),
            ("Hydrophobic core packing", "Tertiary"),
            ("Disulfide bonds within one chain", "Tertiary"),
            ("Subunit–subunit association", "Quaternary"),
            ("Hemoglobin α2β2 assembly", "Quaternary"),
        ],
        "transcription": [
            ("Promoter access", "Initiation"),
            ("Pause‑release during elongation", "Elongation"),
            ("Transcript termination", "Termination"),
            ("mRNA processing (capping/splicing/polyA)", "Processing"),
        ],
        "translation": [
            ("Start‑site recognition", "Initiation"),
            ("Elongation speed", "Elongation"),
            ("Release‑factor‑mediated termination", "Termination"),
            ("Quality control of misfolded products", "Quality control"),
        ],
        "replication": [
            ("Origin firing", "Initiation"),
            ("Fork progression", "Elongation"),
            ("Okazaki fragment joining", "Ligation"),
            ("Fork re‑annealing prevention", "Fork stability"),
        ],
        "microscopy": [
            ("Minimum resolvable distance (d)", "Resolution"),
            ("Optical sectioning", "Contrast"),
            ("Spherical aberration", "Artifacts"),
            ("Nyquist sampling", "Sampling"),
        ],
        "signaling": [
            ("Receptor activation", "Upstream activation"),
            ("Inhibitor action on pathway", "Pathway inhibition"),
            ("Downstream target output", "Downstream output"),
            ("Negative feedback", "Feedback"),
        ],
        "cell_cycle": [
            ("Entry into next phase", "Entry/Commitment"),
            ("Checkpoint engagement", "Checkpoint hold"),
            ("Transition to anaphase", "Transition"),
            ("Completion of division", "Exit/Completion"),
        ],
        "generic": [
            ("Product formation rate", "Increase"),
            ("Core step throughput", "Decrease"),
            ("Intermediate species", "Accumulates"),
            ("Immediate off‑pathway effects", "No change"),
        ],
    }
    pairs = list(seeds.get(topic, seeds["generic"]))

    if topic == "generic":
        generic_bins = {"increase":"Increase","decrease":"Decrease","no_change":"No change","accumulate":"Accumulates"}
        for stem, key, noun in mined:
            label = generic_bins.get(key, "No change")
            short = re.sub(r"^When this.*?:\s*", "", stem, flags=re.IGNORECASE)
            short = re.sub(r"\s+What would you expect.*$", "", short, flags=re.IGNORECASE)
            short = short.strip(". ")
            if len(short) > 90: short = short[:87] + "…"
            pairs.append((short, label))

    seen = set(); uniq = []
    for s, lbl in pairs:
        k = (s.lower(), lbl.lower())
        if k not in seen:
            uniq.append((s,lbl)); seen.add(k)
    random.shuffle(uniq)
    if len(uniq) < 6:
        pad = [("Immediate product output", labels[0]),
               ("Processing intermediates", labels[-1] if "Accumulates" in labels else labels[0]),
               ("Core step throughput", labels[1] if len(labels)>1 else labels[0]),
               ("Non‑target side effects", labels[2] if len(labels)>2 else labels[0])]
        for s,lbl in pad:
            if (s.lower(), lbl.lower()) not in seen:
                uniq.append((s,lbl))
            if len(uniq) >= 6: break

    bank_pairs = uniq[:8]
    bank = [s for (s,_) in bank_pairs]
    answers = {s:lbl for (s,lbl) in bank_pairs}
    return labels, bank, answers

# ---------------- UI ----------------
st.title("📘 Prompt → Critical‑Thinking Activities")
st.caption("Enter a topic. The app reads your slides and builds fresh, application‑style practice (no trivia).")

prompt = st.text_input(
    "Enter a topic (e.g., protein structure, transcription initiation, replication fork dynamics)",
    value="", placeholder="Type your topic…"
)

if st.button("Generate"):
    # Load slides once
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)

    texts = st.session_state.corpus
    topic = detect_topic(prompt)
    mined = mine_prompted_relations(texts, prompt, max_items=10) if topic != "protein_structure" else []
    vocab = corpus_vocab(texts)

    st.session_state.fitb = build_fitb(topic, mined, vocab)
    st.session_state.drag_labels, st.session_state.drag_bank, st.session_state.drag_answer = build_drag(topic, mined)

    st.success(f"Generated fresh activities for **{topic.replace('_',' ').title()}**.")

# ---------------- Activity 1: FITB ----------------
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

# ---------------- Activity 2: TRUE Drag (HTML5) ----------------
if all(k in st.session_state for k in ["drag_labels","drag_bank","drag_answer"]):
    st.markdown("---")
    st.markdown("## Activity 2 — Drag the statements into the correct bin")

    labels = st.session_state.drag_labels
    bank   = st.session_state.drag_bank
    answer = st.session_state.drag_answer

    # Ensure ≥6 cards
    if len(bank) < 6:
        filler = ["Amino‑acid sequence","Alpha‑helix content","Hydrophobic core packing",
                  "Subunit interface contacts","Disulfide bonds (intra‑chain)","Beta‑sheet registry"]
        for f in filler:
            if f not in bank:
                if "Primary" in labels:
                    mapping = {
                        "Amino‑acid sequence":"Primary",
                        "Alpha‑helix content":"Secondary",
                        "Beta‑sheet registry":"Secondary",
                        "Hydrophobic core packing":"Tertiary",
                        "Disulfide bonds (intra‑chain)":"Tertiary",
                        "Subunit interface contacts":"Quaternary",
                    }
                    answer[f] = mapping.get(f, labels[0])
                else:
                    answer[f] = labels[0]
                bank.append(f)

    # Build HTML (drop targets only on inner holders to avoid mis-drops)
    bins_html = "".join([
        f"""
        <div class="bin">
          <div class="title">{lbl}</div>
          <div id="bin_{i}" class="holder" ondrop="drop(event)" ondragover="allow(event)"></div>
        </div>
        """ for i,lbl in enumerate(labels)
    ])

    html_payload = f"""
    <div style="font-family: ui-sans-serif, system-ui; line-height:1.35;">
      <style>
        .bank, .bin {{
          border: 2px dashed #bbb; border-radius: 10px; padding: 10px; min-height: 120px;
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
          {bins_html}
        </div>
      </div>

      <div style="margin-top:10px;">
        <button onclick="checkBins()">Check bins</button>
        <span id="score" style="margin-left:10px;"></span>
      </div>

      <script>
        const BANK_ITEMS = {json.dumps(bank)};
        const ANSWERS = {json.dumps(answer)};

        function allow(ev) {{ ev.preventDefault(); ev.dataTransfer.dropEffect = "move"; }}
        function drag(ev)  {{ ev.dataTransfer.setData("text/plain", ev.target.id); ev.dataTransfer.effectAllowed = "move"; }}
        function drop(ev) {{
          ev.preventDefault();
          const id = ev.dataTransfer.getData("text/plain");
          const card = document.getElementById(id);
          const target = ev.target;
          if (target.classList.contains('holder') || target.id === 'bank') {{
            target.appendChild(card);
          }}
        }}
        function renderBank() {{
          const b = document.getElementById("bank");
          b.innerHTML = "";
          BANK_ITEMS.forEach((txt, idx) => {{
            const c = document.createElement("div");
            c.className = "card";
            c.id = "card_" + idx;
            c.setAttribute("draggable", "true");
            c.addEventListener("dragstart", drag);
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
    st.components.v1.html(html_payload, height=560, scrolling=True)
