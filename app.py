
import os
import re
import io
import json
import time
import random
import requests
from typing import List, Dict, Any, Tuple
import streamlit as st

# ============================================================
# App shell — preserves UX (one prompt + Generate + two activities)
# ============================================================
st.set_page_config(page_title="Let's Practice Biology!", page_icon="🎓", layout="wide")
st.title("Let's Practice Biology!")

# ---------------- GitHub locations (no env vars) ----------------
GITHUB_USER   = "mwellis825"
GITHUB_REPO   = "cell-bio-gpt-tutor"
GITHUB_BRANCH = "main"
SLIDES_DIR_GH = "slides"   # slides live here
EXAMS_DIR_GH  = "exams"    # optional prior exams for style imitation

# ---------------- Small utils ----------------
def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_json_block(s: str) -> str:
    """Pull the first valid {...} or [...] block from model text."""
    s = _strip_code_fences(s)
    a_start, a_end = s.find("["), s.rfind("]")
    o_start, o_end = s.find("{"), s.rfind("}")
    if a_start != -1 and a_end != -1 and a_end > a_start:
        cand = s[a_start:a_end+1]
        try:
            json.loads(cand); return cand
        except Exception:
            pass
    if o_start != -1 and o_end != -1 and o_end > o_start:
        cand = s[o_start:o_end+1]
        try:
            json.loads(cand); return cand
        except Exception:
            pass
    return s


def _lex_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb: return 0.0
    return len(ta & tb) / float(len(ta | tb))

def _valid_bins_terms(bins, terms, mapping) -> bool:
    # bins 3-4; terms 6-8; every term mapped; no bin label lexically overlapping with term too much
    if not (isinstance(bins, list) and 3 <= len(bins) <= 4): return False
    if not (isinstance(terms, list) and 6 <= len(terms) <= 8): return False
    if not (isinstance(mapping, dict) and all(t in mapping for t in terms)): return False
    # each bin has at least 2 terms
    counts = {b:0 for b in bins}
    for t in terms:
        b = mapping.get(t)
        if b not in counts: return False
        counts[b] += 1
        # lexical overlap constraint
        for lbl in bins:
            if b == lbl and _lex_overlap(lbl, t) > 0.4:  # too similar
                return False
    if min(counts.values()) < 2: return False
    return True


def new_seed() -> int:
    return int(time.time() * 1000) ^ random.randint(0, 1_000_000)

# ---------------- GitHub fetchers ----------------
def _gh_list(user: str, repo: str, path: str, branch: str) -> List[Dict[str, Any]]:
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{path}?ref={branch}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []

def _gh_fetch_raw(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def _read_pdf_bytes(pdf_bytes: bytes) -> str:
    # Try pypdf then PyPDF2
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        return "\\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception:
        try:
            import PyPDF2  # type: ignore
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            return "\\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception:
            return ""

@st.cache_data(show_spinner=False)
def load_corpus_from_github(user: str, repo: str, path: str, branch: str) -> List[str]:
    corpus = []
    try:
        items = _gh_list(user, repo, path, branch)
    except Exception:
        return corpus
    for it in items:
        if it.get("type") != "file":
            continue
        name = (it.get("name") or "").lower()
        raw_url = it.get("download_url") or f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}/{it.get('name')}"
        try:
            if name.endswith(".pdf"):
                txt = _read_pdf_bytes(_gh_fetch_raw(raw_url))
            elif name.endswith((".txt",".md",".html",".htm")):
                txt = (_gh_fetch_raw(raw_url)).decode("utf-8", "ignore")
            else:
                txt = ""
        except Exception:
            txt = ""
        if txt and len(txt.strip()) > 20:
            corpus.append(txt)
    return corpus

# ---------------- Retrieval to build a slide scope ----------------
STOP = {
    "the","and","for","that","with","this","from","into","are","was","were","has","have","had","can","will","would","could","should",
    "a","an","of","in","on","to","by","as","at","or","be","is","it","its","their","our","your","if","when","then","than","but",
    "we","you","they","which","these","those","there","here","such","may","might","also","very","much","many","most","more","less"
}
def _tokens_nostop(s: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9']+", (s or "").lower()) if t not in STOP and len(t) > 2]

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\\.!?])\\s+|\\n+", text or "")
    return [re.sub(r"\\s+"," ",p).strip() for p in parts if p and len(p.strip()) > 30]

def _relevance(sent: str, q_tokens: List[str]) -> int:
    bag = {}
    for tk in re.findall(r"[A-Za-z0-9']+", sent.lower()):
        bag[tk] = bag.get(tk,0)+1
    score = sum(bag.get(q,0) for q in q_tokens)
    s_low = sent.lower()
    if len(q_tokens) >= 2 and " ".join(q_tokens[:2]) in s_low:
        score += 2
    return score

def collect_prompt_matched(corpus: List[str], prompt: str, top_docs=6, max_sents=1200) -> List[str]:
    q = _tokens_nostop(prompt)
    if not q:
        return []
    doc_scores = []
    for doc in corpus:
        sents = _split_sentences(doc)
        sc = sum(_relevance(s, q) for s in sents[:max_sents])
        if sc > 0:
            doc_scores.append((sc, sents))
    doc_scores.sort(reverse=True, key=lambda x: x[0])
    matched = []
    for _, sents in doc_scores[:top_docs]:
        for s in sents:
            if _relevance(s, q) > 0:
                matched.append(s)
    return matched[:200]

def build_scope(corpus: List[str], prompt: str, limit_chars: int = 6000) -> str:
    sents = collect_prompt_matched(corpus, prompt, top_docs=6, max_sents=1200)
    scope = "\\n".join(sents)[:limit_chars]
    return scope

# ---------------- Topic classification (for fallback generators) ----------------
def classify_topic(prompt: str) -> str:
    p = (prompt or "").lower()
    if "organelle" in p: return "organelle function"
    if "replicat" in p: return "dna replication"
    if "transcription" in p: return "transcription"
    if "translation" in p: return "translation"
    if "glycolysis" in p: return "glycolysis"
    if "membrane" in p or "transport" in p: return "membrane transport"
    if "protein sorting" in p or "signal sequence" in p or "er signal" in p: return "protein sorting"
    if "cell cycle" in p or "mitosis" in p: return "cell cycle"
    if "bond" in p or "bonds" in p: return "chemical bonds"
    if "dna repair" in p or "repair" in p: return "dna repair"
    if "dna" in p: return "dna"
    return (p.split(",")[0].split(";")[0] or "this topic").strip()

# ---------------- Optional OpenAI client ----------------
def _openai_client():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=key)
    except Exception:
        return None

def _chat(client, system, user, max_tokens=900, temperature=0.1, seed=42) -> str:
    return client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"),
        temperature=temperature, max_tokens=max_tokens, seed=seed,
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    ).choices[0].message.content or ""

# ---------------- LLM Generators (strict JSON) + Fallbacks ----------------

def gen_dnd_from_scope(scope: str, prompt: str):
    client = _openai_client()
    if client is None or not scope.strip():
        return None

    def _ask(require_strict: bool, seed_val: int):
        sys_prompt = "You create rigorous drag-and-drop activities grounded ONLY in the provided slide excerpts. Never reveal answers."
        # No f-strings to avoid brace escaping issues
        user_prompt = (
            "Create ONE classification drag-and-drop activity from the slide excerpts.\n\n"
            "Slide excerpts:\n\"\"\"\n" + scope + "\n\"\"\"\n\n"
            "Student prompt: \"" + prompt + "\"\n\n"
            "Design constraints (must follow ALL):\n"
            "- 3–4 bins with CONCEPTUAL labels (e.g., mechanism, role, evidence, regulation). Avoid using any term's words in bin labels.\n"
            "- 6–8 draggable ITEMS as short phrases (4–12 words) that require reasoning; prefer statements, mini-scenarios, or definitions—not single words.\n"
            "- Include at least one confusable pair that tests nuance.\n"
            "- Every item maps to exactly one bin.\n"
            "- Provide a short, non-revealing hint for each item.\n"
            "- Use ONLY information present in the slide excerpts.\n"
            "- " + ("Bins must NOT be substrings/overlaps of any item (strict)." if require_strict else "Avoid label–item lexical overlap.") + "\n\n"
            "Return STRICT JSON (no markdown):\n"
            "{\n"
            "  \"title\": \"string\",\n"
            "  \"bins\": [\"string\", ...],              // 3-4 conceptual labels\n"
            "  \"terms\": [\"string\", ...],             // 6-8 short phrases\n"
            "  \"mapping\": {\"TERM\":\"BIN\"},          // every term maps to a listed bin\n"
            "  \"hints\": {\"TERM\":\"one short hint\"}\n"
            "}\n"
        )
        raw = _chat(client, sys_prompt, user_prompt, max_tokens=900, temperature=0.35, seed=seed_val)
        return raw

    tries = 2
    for attempt in range(tries):
        seed_val = (new_seed() % 1_000_000)
        raw = _ask(require_strict=(attempt == 1), seed_val=seed_val)
        try:
            raw = _extract_json_block(raw)
            data = json.loads(raw)
        except Exception:
            data = {}
        bins  = data.get("bins") or []
        terms = data.get("terms") or []
        mapping = data.get("mapping") or {}
        hints = data.get("hints") or {}
        # Basic presence check
        if not bins or not terms or not mapping:
            continue
        # Enforce variety/rigor
        if not _valid_bins_terms(bins, terms, mapping):
            continue
        # Build UI payload
        title = data.get("title") or "Concept classification"
        instr = "Drag each statement into the best-fitting conceptual category."
        labels = bins
        draggables = terms
        answer = {t: mapping.get(t) for t in terms}
        hint_map = {t: (hints.get(t) or "Focus on the distinctive feature that fits this category.") for t in terms}
        return title, instr, labels, draggables, answer, hint_map

    # If all attempts fail, return None to trigger fallback
    return None


def gen_fitb_from_scope(scope: str, prompt: str):
    client = _openai_client()
    if client is None or not scope.strip():
        return None

    def _ask(seed_val: int):
        sys_prompt = "You create rigorous fill-in-the-blank items grounded ONLY in the provided slide excerpts. Never reveal answers."
        user_prompt = (
            "From the slide excerpts, create 4–6 FITB items that require understanding (not recall of a single word).\n\n"
            "Slide excerpts:\n\"\"\"\n" + scope + "\n\"\"\"\n\n"
            "Student prompt: \"" + prompt + "\"\n\n"
            "Design constraints (must follow ALL):\n"
            "- Each item is a sentence of 10–25 words with 1–2 blanks (use 4+ underscores per blank, e.g., ______).\n"
            "- Answers must be recoverable from the excerpts (terms, phrases, or names actually present).\n"
            "- Mix specificity (e.g., enzyme names) with conceptual phrases (e.g., rate-limiting step) when present in slides.\n"
            "- Avoid trivial clozes (e.g., repeating the bin label). Avoid meta/boilerplate phrasing.\n"
            "- Provide one concise, non-revealing hint per item.\n\n"
            "Return STRICT JSON array (no markdown). Each item:\n"
            "{\"stem\":\"A 10–25 word sentence with ______ blank(s)\",\"answers\":[\"a\",\"b\"],\"hint\":\"one short hint\"}\n"
        )
        raw = _chat(client, sys_prompt, user_prompt, max_tokens=900, temperature=0.35, seed=seed_val)
        return raw

    tries = 2
    best = None
    for attempt in range(tries):
        seed_val = (new_seed() % 1_000_000)
        raw = _ask(seed_val)
        try:
            raw = _extract_json_block(raw)
            items = json.loads(raw)
        except Exception:
            items = []
        # Normalize and filter
        out = []
        for it in (items if isinstance(items, list) else []):
            stem = (it.get("stem","") or "").strip()
            ans  = it.get("answers", [])
            hint = (it.get("hint","") or "Use the exact term from the slides.").strip()
            if isinstance(stem,str) and isinstance(ans,list) and "____" in stem and not _is_boilerplate(stem):
                # keep 1–2 blanks only
                blanks = _count_blanks(stem)
                if 1 <= blanks <= 2 and 1 <= len(ans) <= 5:
                    out.append({"stem": stem, "answers": [a for a in ans if isinstance(a,str) and a.strip()], "hint": hint})
        # Validate richer requirements
        if _valid_fitb_items(out):
            best = out[:6]
            break

    return best
def build_dnd_activity(topic: str) -> Tuple[str, List[str], List[str], Dict[str,str], Dict[str,str]]:
    rng = random.Random(new_seed())
    options = {
        "organelle function": (["Nucleus","Mitochondrion","Golgi"], ["Houses DNA","ATP production","Protein sorting"], {"Houses DNA":"Nucleus","ATP production":"Mitochondrion","Protein sorting":"Golgi"}),
        "glycolysis": (["Hexokinase","PFK-1","Pyruvate kinase"], ["First phosphorylation","Commitment step","ATP at end"], {"First phosphorylation":"Hexokinase","Commitment step":"PFK-1","ATP at end":"Pyruvate kinase"}),
        "transcription": (["Pol II","Spliceosome","Capping enzymes"], ["mRNA synthesis","Remove introns","Add 5' cap"], {"mRNA synthesis":"Pol II","Remove introns":"Spliceosome","Add 5' cap":"Capping enzymes"}),
        "chemical bonds": (["Covalent","Ionic","Hydrogen bond"], ["Electron sharing","Charge attraction","Polar interaction"], {"Electron sharing":"Covalent","Charge attraction":"Ionic","Polar interaction":"Hydrogen bond"}),
        "dna repair": (["BER","NER","MMR"], ["Base-specific removal","Bulky lesion removal","Mismatch correction"], {"Base-specific removal":"BER","Bulky lesion removal":"NER","Mismatch correction":"MMR"}),
        "dna": (["DNA","Nucleotide","Phosphodiester bond"], ["Genetic polymer","Monomer","Backbone link"], {"Genetic polymer":"DNA","Monomer":"Nucleotide","Backbone link":"Phosphodiester bond"}),
    }
    labels, terms, mapping = options.get(topic, (["Category A","Category B"], ["Item 1","Item 2"], {"Item 1":"Category A","Item 2":"Category B"}))
    rng.shuffle(terms)
    title = f"Match items for {topic}"
    instr = "Match each **item** to its **category**."
    hints = {t: "Focus on the defining feature mentioned in class." for t in terms}
    return title, instr, labels, terms, mapping, hints

def build_fitb(topic: str, rng: random.Random) -> List[Dict[str,Any]]:
    items = []
    if "glycolysis" in topic:
        items.append({"stem":"The end product of glycolysis is ______.","answers":["pyruvate"],"hint":"Three-carbon product."})
        items.append({"stem":"Glycolysis occurs in the ______.","answers":["cytosol","cytoplasm"],"hint":"Not an organelle lumen."})
    elif "transcription" in topic:
        items.append({"stem":"Eukaryotic mRNA is made by ______ polymerase II.","answers":["rna"],"hint":"Pol II."})
    elif "dna repair" in topic:
        items.append({"stem":"Bulky adducts (e.g., thymine dimers) are removed by ______.","answers":["nucleotide excision repair","ner"],"hint":"The pathway that excises an oligonucleotide."})
    else:
        items.append({"stem":f"{topic.title()} primarily involves the ______.","answers":["key term"],"hint":"Use the exact term used in lecture."})
    rng.shuffle(items)
    return items[:4]

# ---------------- Exam-style (optional) ----------------
@st.cache_data(show_spinner=False)
def load_exam_corpus() -> List[str]:
    return load_corpus_from_github(GITHUB_USER, GITHUB_REPO, EXAMS_DIR_GH, GITHUB_BRANCH)

def extract_exam_style(exam_corpus: List[str]) -> Dict[str,Any] | None:
    client = _openai_client()
    if client is None or not exam_corpus:
        return None
    sample = "\\n\\n".join(exam_corpus)[:12000]
    system = "You distill exam style. Return strict JSON only."
    user = (
        "From the prior exams below, extract a compact style profile.\\n\\n"
        "PRIOR EXAMS (snippets):\\n\"\"\"" + sample + "\"\"\"\\n\\n"
        "Return STRICT JSON:\\n"
        "{\\n"
        "  \"preferred_types\": [\"mcq\",\"short_answer\"],\\n"
        "  \"mcq_options\": 4,\\n"
        "  \"tone\": \"succinct|formal|clinical|conversational\",\\n"
        "  \"length\": \"short|medium|long\",\\n"
        "  \"constraints\": [\"single-best-answer\"],\\n"
        "  \"rationale_required\": true\\n"
        "}\\n"
    )
    try:
        raw = _chat(client, system, user, max_tokens=400, temperature=0.0)
        raw = _extract_json_block(raw)
        prof = json.loads(raw)
        prof.setdefault("preferred_types", ["mcq","short_answer"])
        prof.setdefault("mcq_options", 4)
        prof.setdefault("tone", "succinct")
        prof.setdefault("length", "medium")
        prof.setdefault("constraints", ["single-best-answer"])
        prof.setdefault("rationale_required", True)
        return prof
    except Exception:
        return None

def gen_exam_question(scope: str, style: Dict[str,Any], user_prompt: str) -> Dict[str,Any] | None:
    client = _openai_client()
    if client is None or not scope.strip():
        return None
    pref = (style or {}).get("preferred_types", ["mcq","short_answer"])
    want_mcq = bool(pref and pref[0] == "mcq")
    mcq_n = max(3, min(int((style or {}).get("mcq_options", 4)), 5))
    system = "You write exam questions grounded ONLY in the provided slide excerpts. Return strict JSON only."
    style_json = json.dumps(style or {}, ensure_ascii=False)
    user = (
        "SLIDE EXCERPTS (authoritative; cite page numbers if present):\\n\"\"\"\\n" + scope + "\\n\"\"\"\\n\\n"
        "STUDENT PROMPT: \"" + user_prompt + "\"\\n\\n"
        "STYLE PROFILE:\\n" + style_json + "\\n\\n"
        "Create ONE exam-style question faithful to the slides.\\n\\n"
        "Schema:\\n"
        "{\\n"
        "  \"type\": \"mcq\" | \"short_answer\",\\n"
        "  \"stem\": \"string (<= 70 words; tone per style)\",\\n"
        "  \"options\": [\"A\",\"B\",\"C\",\"D\"],      // MCQ only, " + str(mcq_n) + " options\\n"
        "  \"answer\": \"B\",                     // letter for MCQ; text for SA\\n"
        "  \"rationale\": \"why correct; <= 60 words\",\\n"
        "  \"bloom\": \"Remember|Understand|Apply|Analyze|Evaluate|Create\",\\n"
        "  \"difficulty\": 1,                   // 1-5\\n"
        "  \"slide_refs\": [int]\\n"
        "}\\n\\n"
        "- Distractors must be plausible and present in scope.\\n"
        "- Cite 1–3 slide pages in \"slide_refs\".\\n"
        "- Prefer type: " + ("MCQ" if want_mcq else "short_answer") + " if it fits.\\n"
    )
    try:
        raw = _chat(client, system, user, max_tokens=700, temperature=0.1)
        raw = _extract_json_block(raw)
        q = json.loads(raw)
        if q.get("type") not in ("mcq","short_answer"): return None
        if not isinstance(q.get("stem",""), str) or len(q["stem"].split()) < 3: return None
        if q["type"] == "mcq":
            opts = q.get("options", [])
            if not (isinstance(opts, list) and 3 <= len(opts) <= 5 and isinstance(q.get("answer",""), str)):
                return None
        else:
            if not isinstance(q.get("answer",""), str) or not q["answer"].strip():
                return None
        if isinstance(q.get("slide_refs"), list):
            q["slide_refs"] = [int(x) for x in q["slide_refs"] if str(x).isdigit() or isinstance(x,int)]
        else:
            q["slide_refs"] = []
        return q
    except Exception:
        return None

# ---------------- UI: prompt + Generate (unchanged) ----------------
prompt = st.text_input(
    "Enter a topic for review and press generate:",
    value="",
    placeholder="e.g., organelle function, glycolysis regulation, DNA repair…",
    label_visibility="visible",
)

# ---------------- Generate: LLM-first grounded, then fallback ----------------
if st.button("Generate"):
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus_from_github(GITHUB_USER, GITHUB_REPO, SLIDES_DIR_GH, GITHUB_BRANCH)
    if "exam_corpus" not in st.session_state:
        st.session_state.exam_corpus = load_corpus_from_github(GITHUB_USER, GITHUB_REPO, EXAMS_DIR_GH, GITHUB_BRANCH)

    topic = classify_topic(prompt) or "this topic"
    st.session_state.topic = topic
    scope = build_scope(st.session_state.corpus or [], prompt, limit_chars=6000)

    # Activity 1: Drag & Drop
    dnd = gen_dnd_from_scope(scope, prompt)
    if dnd is None:
        title, instr, labels, terms, answer, hint_map = build_dnd_activity(topic)
        st.session_state.dnd_source  = "Fallback"
    else:
        title, instr, labels, terms, answer, hint_map = dnd
        st.session_state.dnd_source  = "LLM"
    st.session_state.dnd_title = title
    st.session_state.dnd_instr = instr
    st.session_state.drag_labels = labels
    st.session_state.drag_bank   = terms
    st.session_state.drag_answer = answer
    st.session_state.dnd_hints   = hint_map

    # Activity 2: FITB
    rng = random.Random(new_seed())
    fitb_items = gen_fitb_from_scope(scope, prompt)
    if fitb_items is None:
        fitb_items = build_fitb(topic, rng)
        st.session_state.fitb_source = "Fallback"
    else:
        st.session_state.fitb_source = "LLM"
    st.session_state.fitb = fitb_items

    # Exam-style question (optional)
    style = extract_exam_style(st.session_state.exam_corpus or [])
    exam_q = gen_exam_question(scope, style or {}, prompt) if style else None
    st.session_state.exam_q = exam_q
    st.session_state.exam_source = "LLM" if exam_q else "Unavailable"

# ---------------- Render Activity 1: Drag & Drop ----------------
if all(k in st.session_state for k in ["drag_labels","drag_bank","drag_answer","dnd_instr"]):
    st.markdown("## Activity 1: Drag and Drop")
    st.caption(f"Source: {st.session_state.get('dnd_source','') or 'Fallback'}")
    st.markdown(f"**{st.session_state.get('dnd_title','Match items')}**")
    st.markdown(st.session_state.dnd_instr)

    labels = st.session_state.drag_labels
    terms  = st.session_state.drag_bank
    answer = st.session_state.drag_answer
    hint_map = st.session_state.dnd_hints

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
        body {{ margin:0; padding:0; }}
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
        }}
        .ghost {{ opacity: 0.5; }}
        .chosen {{ outline: 2px solid #7aa2f7; }}
        .zone {{ display:flex; gap:14px; }}
        .left {{ flex: 1; }}
        .right {{ flex: 2; display:grid; grid-template-columns: repeat({cols_count}, 1fr); gap:14px; }}
        .title {{ font-weight: 600; margin-bottom: 6px; }}
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
            score.innerHTML = "<span class='bad'>" + correct + "/" + total + " correct — try again.</span>";
          }}
        }});
      </script>
    </body>
    </html>
    """
    st.components.v1.html(html, height=560, scrolling=True)

    c1, c2 = st.columns([1,3])
    with c1:
        chosen_item = st.selectbox("Hint for:", ["(choose…)"] + terms, index=0, key="dnd_hint_select")
    with c2:
        if chosen_item != "(choose…)":
            fb = hint_map.get(chosen_item, "Focus on the distinctive clue in this item.")
            st.info(fb)

# ---------------- Render Activity 2: Fill in the Blank ----------------
if "fitb" in st.session_state:
    st.markdown("---")
    st.markdown("## Activity 2: Fill in the Blank")
    st.caption(f"Source: {st.session_state.get('fitb_source','') or 'Fallback'}")
    topic_name = st.session_state.get("topic","this topic")
    st.markdown(f"Use your knowledge of **{topic_name}** to answer the following.")

    rng = random.Random(new_seed())
    for idx, item in enumerate(st.session_state.fitb):
        u = st.text_input(item["stem"], key=f"fitb_{idx}")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Hint", key=f"hint_{idx}"):
                st.info(item.get("hint","Focus on the exact term used in lecture."))
        with col2:
            if st.button("Check", key=f"check_{idx}"):
                ok = False
                ans_list = item.get("answers", [])
                norm = lambda s: re.sub(r"[^a-z0-9]+","", s.lower())
                if isinstance(ans_list, list):
                    ok = norm(u) in {norm(a) for a in ans_list}
                st.success("That’s right! 🎉") if ok else st.warning("Not quite. Try again or use the hint.")
        with col3:
            if st.button("Reveal", key=f"rev_{idx}"):
                ans = item.get("answers", [])
                st.info(", ".join(ans) if ans else "(no stored answer)")

# ---------------- Render Exam-style Question (optional) ----------------
if st.session_state.get("exam_q"):
    st.markdown("---")
    st.markdown("## Exam-style Question")
    st.caption(f"Source: {st.session_state.get('exam_source','Unavailable')}")
    q = st.session_state.exam_q
    st.write(q.get("stem","(no stem)"))
    if q.get("type") == "mcq":
        opts = q.get("options", [])
        try:
            idx_default = 0 if opts else None
        except Exception:
            idx_default = None
        choice = st.radio("Choose one:", opts, index=idx_default, key="exam_choice")
        if st.button("Check exam answer"):
            correct = q.get("answer","")
            if isinstance(correct, str) and choice and choice.strip().upper().startswith(correct.strip().upper()):
                st.success("Correct ✅")
            else:
                st.error(f"Not quite. Correct answer: {correct}")
            st.caption(f"Bloom: {q.get('bloom','?')} • Difficulty: {q.get('difficulty','?')} • Slides: {', '.join(str(x) for x in q.get('slide_refs',[]))}")
            if q.get("rationale"):
                st.info(q["rationale"])
    else:
        user_sa = st.text_input("Your short answer:", key="exam_sa")
        if st.button("Check short answer"):
            correct = q.get("answer","")
            n = lambda s: re.sub(r"[^a-z0-9]+","", s.lower())
            if n(user_sa) == n(correct):
                st.success("Correct ✅")
            else:
                st.error(f"Expected: {correct}")
            st.caption(f"Bloom: {q.get('bloom','?')} • Difficulty: {q.get('difficulty','?')} • Slides: {', '.join(str(x) for x in q.get('slide_refs',[]))}")
            if q.get("rationale"):
                st.info(q["rationale"])
