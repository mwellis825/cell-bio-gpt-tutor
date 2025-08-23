# app.py
# Prompt-based Study Mode — 4x Fill-in-the-Blank + Drag-to-Bins (click assign)
# -----------------------------------------------------------------------------
# Flow:
#   1) User enters a topic prompt.
#   2) App searches ./slides (PDF/txt/md/html) for relevant decks.
#   3) Generates:
#       (A) FOUR lenient fill-in-the-blank items (auto-picked stems & answers)
#       (B) Drag-to-Bins: word bank on LEFT, labeled bins on RIGHT (click-to-assign)
#
# Dependencies:
#   pip install streamlit
#   # For PDF text extraction:
#   pip install PyPDF2   # or: pip install pypdf
#   streamlit run app.py
# -----------------------------------------------------------------------------

import streamlit as st
import os, re, pathlib

st.set_page_config(page_title="Study Mode", page_icon="📘", layout="wide")

SLIDES_DIR = os.path.join(os.getcwd(), "slides")
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".mdx", ".html", ".htm"}

# Try PDF parser(s)
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

# ----------------------- File IO -----------------------
def read_text_file(path: str) -> str:
    for enc in ("utf-8","latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""

def read_pdf_file(path: str) -> str:
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

def load_corpus(slides_dir: str):
    corpus = {}
    if not os.path.isdir(slides_dir): return corpus
    base = pathlib.Path(slides_dir)
    for p in base.rglob("*"):
        if not p.is_file(): continue
        ext = p.suffix.lower()
        rel = str(p.relative_to(base))
        text = ""
        if ext in SUPPORTED_TEXT_EXTS:
            text = read_text_file(str(p))
        elif ext == ".pdf":
            text = read_pdf_file(str(p))
        if text and len(text.strip()) > 20:
            corpus[rel] = text
    return corpus

# ---------------------- Search & utility ----------------------
def tokenize(s: str):
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())

def match_score(text: str, query_tokens):
    tokens = tokenize(text)
    if not tokens: return 0
    counts = {}
    for t in tokens: counts[t] = counts.get(t,0)+1
    return sum(counts.get(q,0) for q in query_tokens)

def normalize(s: str):
    s = (s or "").strip().lower()
    s = re.sub(r"\s+"," ", s)
    s = s.replace("’","'")
    return s

# ---------------------- LO extraction ----------------------
VERBS = ["explain","predict","justify","compare","contrast","distinguish","design","interpret",
         "evaluate","diagnose","infer","propose","calculate","model","identify","classify",
         "analyze","synthesize","outline","formulate","apply","assess"]

def extract_los(text: str):
    lines = [re.sub(r"\s+"," ", ln).strip() for ln in text.splitlines()]
    los = []
    for ln in lines:
        low = ln.lower()
        if not ln or len(ln) < 6: 
            continue
        bullet = low.startswith(("-","•","—","*")) or re.match(r"^\d+[\).\]]\s", low)
        starts_verb = any(re.match(rf"^\s*({v})\b", low) for v in VERBS)
        mentions_lo = any(k in low for k in ["learning objective","objective:","outcome:"])
        if bullet or starts_verb or mentions_lo:
            cleaned = re.sub(r"^(\d+[\).\]]\s*|[-•—*]\s*)","", ln).strip()
            if len(cleaned) >= 10:
                los.append(cleaned)
    if not los:
        # Fallback: long-ish sentences
        chunks = re.split(r"(?<=[\.\?\!])\s+", text)
        los = [c.strip() for c in chunks if len(c.strip()) > 60][:8]
    # Deduplicate
    seen=set(); uniq=[]
    for x in los:
        k=x.lower()
        if k not in seen:
            uniq.append(x); seen.add(k)
    return uniq[:12]

# ---------------------- Topic detection ----------------------
def detect_topic(text_or_prompt: str):
    t = (text_or_prompt or "").lower()
    if any(k in t for k in ["dna replication","replication fork","okazaki","helicase","primase","ligase","leading strand","lagging strand"]):
        return "dna_replication"
    if any(k in t for k in ["glycolysis","tca","mitochond","electron transport","oxidative phosphorylation"]):
        return "metabolism"
    if any(k in t for k in ["cell cycle","cdk","cyclin","p53","meiosis","mitosis","checkpoint"]):
        return "cell_cycle"
    if any(k in t for k in ["actin","microtubule","kinesin","dynein","focal adhesion","myosin"]):
        return "cytoskeleton"
    if any(k in t for k in ["rtk","gpcr","pi3k","akt","mapk","erk","ras","second messenger"]):
        return "signaling"
    return "generic"

# ---------------------- Generators ----------------------
def make_fitb_set(topic: str):
    # Return 4 (prompt, accepted_answers[]) items, tailored when possible
    items = []
    if topic == "dna_replication":
        items = [
            ("Enzyme that unwinds the parental DNA duplex at the replication fork:", ["helicase"]),
            ("Enzyme that synthesizes short RNA primers to initiate DNA synthesis:", ["primase","rna primase"]),
            ("Fragments produced on the lagging strand during discontinuous synthesis:", ["okazaki fragments","okazaki"]),
            ("Enzyme that seals nicks between adjacent DNA fragments:", ["dna ligase","ligase"]),
        ]
    elif topic == "metabolism":
        items = [
            ("Complex that accepts electrons from NADH at the start of the ETC:", ["complex i","nadH dehydrogenase","nadh dehydrogenase"]),
            ("Ion whose gradient powers ATP synthase in mitochondria:", ["proton","h+","hydrogen ion"]),
            ("Cycle that oxidizes acetyl‑coa to co2 in mitochondria:", ["tca cycle","citric acid cycle","krebs cycle"]),
            ("Mitochondrial enzyme that synthesizes ATP using the proton-MOTIVE force:", ["atp synthase","f0f1 atp synthase","fof1 atp synthase"]),
        ]
    elif topic == "cell_cycle":
        items = [
            ("CDK/cyclin complex that promotes G1→S transition:", ["cyclin e-cdk2","cdk2-cyclin e","cyclin e cdk2"]),
            ("Checkpoint that ensures all chromosomes are attached to the spindle:", ["spindle assembly checkpoint","sac"]),
            ("Ubiquitin ligase that targets securin to trigger anaphase:", ["apc/c","apc c","anaphase promoting complex"]),
            ("Tumor suppressor that halts the cell cycle in response to dna damage:", ["p53"]),
        ]
    elif topic == "cytoskeleton":
        items = [
            ("Motor protein that generally walks toward microtubule plus ends:", ["kinesin"]),
            ("Motor protein that generally walks toward microtubule minus ends:", ["dynein"]),
            ("Actin‑binding protein that crosslinks actin into contractile structures with myosin:", ["alpha‑actinin","alpha actinin","α‑actinin","alpha actinins"]),
            ("Drug that stabilizes microtubules and disrupts mitotic spindle dynamics:", ["taxol","paclitaxel"]),
        ]
    else:  # generic fallback
        items = [
            ("Name one key regulator of this process:", ["regulator","mechanism"]),
            ("Identify the rate‑limiting step:", ["rate-limiting","bottleneck"]),
            ("Name a negative feedback element:", ["feedback","negative feedback"]),
            ("Name a rescue/recovery mechanism:", ["recovery","redundancy"]),
        ]
    return items[:4]

def make_drag_bins(topic: str):
    # Returns (bins: list of (label, correct_word), bank: list of words)
    if topic == "dna_replication":
        bins = [
            ("Unwinds duplex DNA", "helicase"),
            ("Lays down RNA primers", "primase"),
            ("Continuous synthesis strand", "leading strand"),
            ("Discontinuous fragments", "okazaki fragments"),
            ("Seals nicks", "dna ligase"),
        ]
        bank = ["helicase","primase","leading strand","lagging strand","okazaki fragments",
                "dna ligase","topoisomerase","ssb"]
    elif topic == "metabolism":
        bins = [
            ("Accepts electrons from NADH", "complex i"),
            ("Accepts electrons from FADH2", "complex ii"),
            ("Pumps protons to IMS", "complex iii"),
            ("Reduces oxygen to water", "complex iv"),
            ("Synthesizes ATP", "atp synthase"),
        ]
        bank = ["complex i","complex ii","complex iii","complex iv","atp synthase","coq","cytc"]
    else:
        bins = [
            ("Initiates process", "initiator"),
            ("Rate‑limiting step", "rate-limiting"),
            ("Stabilizing feedback", "feedback"),
            ("Recovery pathway", "recovery"),
        ]
        bank = ["initiator","rate-limiting","feedback","recovery","inhibitor","activator"]
    return bins, sorted(list(set(bank)), key=lambda x: x.lower())

# ---------------------- UI ----------------------
st.title("📘 Study Mode — Prompt + Slides")
st.caption("Enter a topic; the app searches your slides and generates 4x Fill‑in‑the‑Blank, then Drag‑to‑Bins.")

prompt = st.text_input("Topic prompt", value="DNA Replication")
if st.button("Generate"):
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus(SLIDES_DIR)
    corpus = st.session_state.corpus

    if not corpus:
        st.error("No slides found or unable to read PDFs. Add slides to `./slides` and/or install PyPDF2/pypdf.")
    else:
        q_tokens = tokenize(prompt)
        ranked = sorted(corpus.items(), key=lambda kv: match_score(kv[1], q_tokens), reverse=True)
        text_best = ranked[0][1] if ranked else ""
        topic = detect_topic(text_best + " " + prompt)
        st.session_state.topic = topic

        # Build FITB
        fitb_items = make_fitb_set(topic)
        st.session_state.fitb = [{"prompt": p, "accepted": a} for (p,a) in fitb_items]

        # Build Drag bins
        bins, bank = make_drag_bins(topic)
        st.session_state.drag_bins = [{"label": lbl, "correct": ans, "current": None} for (lbl, ans) in bins]
        st.session_state.drag_bank = bank
        st.success(f"Generated activities for topic: **{topic.replace('_',' ').title()}**")

# Render activities if available
if "fitb" in st.session_state and "drag_bins" in st.session_state:
    # ---------- Activity A: 4x Fill-in-the-Blank ----------
    st.markdown("## Activity 1: Fill‑in‑the‑Blank")
    for i, item in enumerate(st.session_state.fitb, start=1):
        colq, cola = st.columns([2.5, 1.5])
        with colq:
            st.write(f"**Q{i}.** {item['prompt']}")
        with cola:
            ans = st.text_input("Answer", key=f"fitb_ans_{i}")
            if st.button("Check", key=f"fitb_check_{i}"):
                user = normalize(ans)
                accepted = [normalize(x) for x in item["accepted"]]
                ok = any((user == a) or (a in user) or (user in a) for a in accepted if a)
                st.success("Correct!") if ok else st.error("Not quite.")

    st.markdown("---")

    # ---------- Activity B: Drag-to-Bins (click assign) ----------
    st.markdown("## Activity 2: Drag‑to‑Bins")
    left, right = st.columns([1, 2])

    # LEFT: Word bank (click one to select for placement)
    with left:
        st.subheader("Word bank")
        selected = st.session_state.get("drag_selected")
        st.caption("Click a word to select it, then click a bin label to place it.")
        for idx, w in enumerate(st.session_state.drag_bank):
            if st.button(("🔘 " if selected == w else "⚪ ") + w, key=f"bank_{idx}"):
                st.session_state.drag_selected = w
                selected = w
        if st.button("Reset bank/bins"):
            # move all placed items back to bank
            for b in st.session_state.drag_bins:
                if b["current"]:
                    st.session_state.drag_bank.append(b["current"])
                    b["current"] = None
            st.session_state.drag_bank = sorted(list(set(st.session_state.drag_bank)), key=lambda x: x.lower())
            st.session_state.drag_selected = None
            st.experimental_rerun()

    # RIGHT: Labeled bins
    with right:
        st.subheader("Bins")
        cols = st.columns(2)
        for i, b in enumerate(st.session_state.drag_bins):
            col = cols[i % 2]
            with col:
                label_btn = st.button(f"🗂️ {b['label']}", key=f"bin_label_{i}")
                if label_btn and st.session_state.get("drag_selected"):
                    # place selected word here (swap if occupied)
                    sel = st.session_state.drag_selected
                    if b["current"]:
                        # return current to bank first
                        st.session_state.drag_bank.append(b["current"])
                    b["current"] = sel
                    # remove from bank
                    st.session_state.drag_bank = [x for x in st.session_state.drag_bank if x != sel]
                    st.session_state.drag_selected = None
                    st.experimental_rerun()
                # show current content with a clear button
                st.write("Placed:", b["current"] if b["current"] else "—")
                if b["current"]:
                    if st.button("Remove", key=f"bin_rm_{i}"):
                        st.session_state.drag_bank.append(b["current"])
                        b["current"] = None
                        st.session_state.drag_bank = sorted(list(set(st.session_state.drag_bank)), key=lambda x: x.lower())
                        st.experimental_rerun()

        if st.button("Check bins"):
            placed = sum(1 for b in st.session_state.drag_bins if b["current"])
            if placed < len(st.session_state.drag_bins):
                st.warning(f"Place items in all {len(st.session_state.drag_bins)} bins before checking.")
            else:
                correct = sum(1 for b in st.session_state.drag_bins if normalize(b["current"]) == normalize(b["correct"]))
                if correct == len(st.session_state.drag_bins):
                    st.success("All bins correct! 🎉")
                else:
                    st.warning(f"{correct}/{len(st.session_state.drag_bins)} correct. Keep refining.")

