
# app.py — Cell Bio Tutor (ready-to-run)
# - Uses your GitHub repo slides (no env vars required)
# - Optionally uses slides_index.json for speed
# - LLM generation if OPENAI_API_KEY or st.secrets["OPENAI_API_KEY"] is present; otherwise robust fallbacks
# - Produces unique, unambiguous DnD and application-style FITB tied to relevant slides
# - Keeps student-facing UI minimal (no "(intro level)" etc.)

import streamlit as st
import json, re, os, time, random, hashlib, io, math
from typing import List, Dict, Any, Tuple, Optional
import requests

st.set_page_config(page_title="Cell Bio Tutor", layout="wide")

# ---------------------- Repo configuration ----------------------
GITHUB_USER = "mwellis825"
GITHUB_REPO = "cell-bio-gpt-tutor"
GITHUB_BRANCH = "main"
SLIDES_DIR_GH = "slides"
EXAMS_DIR_GH = "exams"
ACTIVITIES_DIR_GH = "activities"
ALT_ACTIVITIES_DIR_GH = "group_activities"

RAW_API = "https://api.github.com"
HEADERS = {"Accept":"application/vnd.github+json"}

# ---------------------- Utility ----------------------
def new_seed() -> int:
    return int(time.time()*1000) % 1_000_000

def _nonce() -> str:
    return f"{int(time.time()*1e6)}_{new_seed()}"

def _sha(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8","ignore")).hexdigest()

def _get_env_key() -> Optional[str]:
    # Try Streamlit secrets then env
    try:
        return st.secrets.get("OPENAI_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        return os.environ.get("OPENAI_API_KEY")

def _http_json(url: str):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def _http_bytes(url: str) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.content

# ---------------------- GitHub helpers ----------------------
def gh_list(owner: str, repo: str, path: str, branch: str) -> List[Dict[str,Any]]:
    url = f"{RAW_API}/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    try:
        data = _http_json(url)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []

def gh_tree(owner: str, repo: str, branch: str) -> List[Dict[str,Any]]:
    url = f"{RAW_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    try:
        data = _http_json(url)
        return data.get("tree", [])
    except Exception:
        return []

def gh_read_text(owner: str, repo: str, path: str, branch: str) -> Optional[str]:
    # Use contents API to get download_url
    url = f"{RAW_API}/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    try:
        data = _http_json(url)
        if isinstance(data, dict) and data.get("download_url"):
            raw = _http_bytes(data["download_url"])
            return raw.decode("utf-8","ignore")
        return None
    except Exception:
        return None

def gh_read_bytes(owner: str, repo: str, path: str, branch: str) -> Optional[bytes]:
    url = f"{RAW_API}/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    try:
        data = _http_json(url)
        if isinstance(data, dict) and data.get("download_url"):
            return _http_bytes(data["download_url"])
        return None
    except Exception:
        return None

# ---------------------- Slide indexing / scope ----------------------
def load_slides_index() -> Optional[Dict[str,Any]]:
    txt = gh_read_text(GITHUB_USER, GITHUB_REPO, "slides_index.json", GITHUB_BRANCH)
    if txt:
        try:
            return json.loads(txt)
        except Exception:
            return None
    return None

def extract_pdf_text(pdf_bytes: bytes) -> List[str]:
    # Light extraction with PyPDF2 if available
    try:
        import PyPDF2
        pages = []
        with io.BytesIO(pdf_bytes) as f:
            r = PyPDF2.PdfReader(f)
            for p in r.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
        return pages
    except Exception:
        return [""]

def pick_decks_for_prompt(prompt: str, max_decks: int = 2) -> List[Tuple[str,int]]:
    """Return list of (path, pages) from /slides that best match prompt."""
    prompt_l = (prompt or "").lower()
    idx = load_slides_index()
    if idx:
        # Use inverted_index for quick candidate pages
        scores = {}
        terms = [t for t in re.split(r"[^a-z0-9]+", prompt_l) if t]
        for t in terms:
            postings = idx.get("inverted_index", {}).get(t, [])
            for p in postings:
                f = p["file"]
                scores[f] = scores.get(f, 0.0) + p.get("score", 1.0)
        # Fallback to filename match
        for s in idx.get("slides", []):
            f = s["file"]
            if any(k in f.lower() for k in terms):
                scores[f] = scores.get(f, 0.0) + 0.3
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_decks]
        out = []
        for f,_ in ranked:
            pages = 0
            for s in idx.get("slides", []):
                if s["file"] == f:
                    pages = s.get("pages", 0)
                    break
            out.append((f, pages))
        if out:
            return out
    # No index → heuristic by filename
    tree = gh_tree(GITHUB_USER, GITHUB_REPO, GITHUB_BRANCH)
    slide_files = [t["path"] for t in tree if t.get("type")=="blob" and t["path"].lower().startswith(SLIDES_DIR_GH+"/") and t["path"].lower().endswith(".pdf")]
    # simple filename score
    def score(pth):
        base = pth.split("/")[-1].lower()
        s = sum(1 for w in re.split(r"[^a-z0-9]+", prompt_l) if w and w in base)
        return s
    slide_files.sort(key=lambda p: (-score(p), len(p)))
    return [(p, 0) for p in slide_files[:max_decks]]

def build_scope(prompt: str, max_chars: int = 6000) -> Tuple[str, List[Tuple[str,int]]]:
    """Pulls text from chosen slide decks; returns scope text and (file,page) refs used."""
    refs: List[Tuple[str,int]] = []
    decks = pick_decks_for_prompt(prompt)
    chunks = []
    for path, _pages in decks:
        data = gh_read_bytes(GITHUB_USER, GITHUB_REPO, path, GITHUB_BRANCH)
        if not data: continue
        pages = extract_pdf_text(data)
        # Select up to first 20 pages, prioritize those that mention prompt terms
        terms = [t for t in re.split(r"[^a-z0-9]+", (prompt or "").lower()) if t]
        scored = []
        for i,txt in enumerate(pages[:20], start=1):
            tl = (txt or "").lower()
            score = sum(2 for t in terms if t and t in tl) + (1.0/(i+2))
            scored.append((score, i, txt))
        scored.sort(key=lambda x: x[0], reverse=True)
        for sc, i, txt in scored[:4]:  # take top 4 pages
            if txt and len("".join(chunks)) < max_chars:
                chunks.append(f"[{path} p.{i}] {txt.strip()}")
                refs.append((path, i))
    return "\n\n".join(chunks)[:max_chars], refs

# ---------------------- Exams/Activities style cues (no copying) ----------------------
def load_style_cues() -> Dict[str,Any]:
    if "style_cues" in st.session_state:
        return st.session_state["style_cues"]
    buckets = [EXAMS_DIR_GH, ACTIVITIES_DIR_GH, ALT_ACTIVITIES_DIR_GH]
    texts = []
    for b in buckets:
        items = gh_list(GITHUB_USER, GITHUB_REPO, b, GITHUB_BRANCH)
        for it in items:
            if (it.get("type")=="file") and isinstance(it.get("name"), str):
                name = it["name"].lower()
                if name.endswith(".pdf"):
                    # skip heavy parsing; cues from filename
                    base = name.replace("_"," ").replace("-"," ")
                    texts.append(base)
                elif name.endswith((".md",".txt",".html",".htm")) and it.get("download_url"):
                    try:
                        texts.append(_http_bytes(it["download_url"]).decode("utf-8","ignore"))
                    except Exception:
                        pass
    big = "\n".join(texts).lower()
    verbs = sorted(set(re.findall(r"\b(analyze|interpret|predict|justify|evaluate|determine|which|why|how)\b", big)))
    scen  = sorted(set(re.findall(r"\b(patient|cell line|experiment|assay|mutation|inhibitor|drug|blot|gel|image|figure|graph|data)\b", big)))
    comps = sorted(set(re.findall(r"\b(increase|decrease|higher|lower|upregulate|downregulate|more|less)\b", big)))
    cues = {"verbs":verbs or ["interpret","predict","justify"],
            "scenario_markers":scen or ["experiment","mutation","inhibitor"],
            "comparators": comps or ["increase","decrease"]}
    st.session_state["style_cues"] = cues
    return cues

# ---------------------- OpenAI client (optional) ----------------------
def _openai_client():
    key = _get_env_key()
    if not key:
        return None
    try:
        import openai
        openai.api_key = key
        return openai
    except Exception:
        return None

def _extract_json_block(raw: str) -> str:
    m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", raw)
    return m.group(0) if m else raw

def chat_json(system: str, user: str, max_tokens=800, temperature=0.6, seed=None) -> Optional[Any]:
    client = _openai_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed if isinstance(seed,int) else None,
        )
        txt = resp.choices[0].message.content
        return json.loads(_extract_json_block(txt))
    except Exception:
        return None

# ---------------------- DnD: LLM-first strict generator ----------------------
def llm_generate_dnd(scope: str, prompt: str, style: Dict[str,Any]):
    if not scope.strip():
        return None
    nonce = _nonce()
    sys = "Generate unique, unambiguous drag-and-drop activities grounded ONLY in the provided slide excerpts. Never reveal answers."
    user = (
        "Create ONE classification drag-and-drop activity for an introductory biology course.\n\n"
        f"Topic (student prompt): \"{prompt}\"\n"
        f"Novelty nonce: {nonce}\n\n"
        "Slide excerpts (authoritative content — pick labels that appear verbatim; ≤3 words per label):\n"
        + "\"\"\"\n" + scope + "\n\"\"\"\n\n"
        "Return STRICT JSON:\n"
        "{\n"
        "  \"title\":\"string\",\n"
        "  \"bins\":[\"Label1\",\"Label2\",\"Label3?\"],\n"
        "  \"terms\":[\"t1\",\"t2\",\"t3\",\"t4\"],\n"
        "  \"mapping\":{\"t1\":\"LabelX\",...},\n"
        "  \"hints\":{\"t1\":\"hint\",...},\n"
        "  \"bin_keywords\":{\"LabelX\":[\"token1\",\"token2\"],...},\n"
        "  \"ambiguity_matrix\":[[true,false,...],...]\n"
        "}\n"
        "Hard constraints:\n"
        "- 2–3 labels that appear verbatim in the excerpts.\n"
        "- Exactly 4 terms (3–8 words). Each term must match exactly ONE label.\n"
        "- Provide non-revealing hints and bin keywords. Ensure ambiguity_matrix has exactly one TRUE per row."
    )
    data = chat_json(sys, user, max_tokens=900, temperature=0.55, seed=new_seed()%1_000_000)
    if not isinstance(data, dict):
        return None
    # Validate
    bins = data.get("bins") or []
    terms = data.get("terms") or []
    mapping = data.get("mapping") or {}
    hints = data.get("hints") or {}
    bin_kw = data.get("bin_keywords") or {}
    amb = data.get("ambiguity_matrix") or []

    if not (isinstance(bins, list) and 2 <= len(bins) <= 3): return None
    if not (isinstance(terms, list) and len(terms) == 4): return None
    if not all(isinstance(t,str) and 3 <= len(t.split()) <= 8 for t in terms): return None
    if not (isinstance(mapping, dict) and all(t in mapping and mapping[t] in bins for t in terms)): return None
    if not (isinstance(amb, list) and len(amb) == 4): return None
    for row in amb:
        if not (isinstance(row, list) and len(row) == len(bins) and sum(1 for x in row if x is True) == 1):
            return None
    if not (isinstance(bin_kw, dict) and all(isinstance(bin_kw.get(b), list) and bin_kw[b] for b in bins)):
        return None

    # Additional guard: if "stop codon" present, ensure it's mapped to a Termination-like bin
    if any("stop codon" in t.lower() for t in terms):
        term_stop = [t for t in terms if "stop codon" in t.lower()][0]
        bin_for = mapping.get(term_stop,"").lower()
        if not any(k in bin_for for k in ["termin", "release"]):
            return None

    title = data.get("title") or "Match items to categories"
    instr = "Drag each item to the correct category."
    return title, instr, bins, terms, mapping, hints

# ---------------------- DnD fallback (rules-based, unambiguous) ----------------------
def dnd_fallback(scope: str, prompt: str):
    topic = (prompt or "").lower()
    # Minimal topic rules
    RULES = {
        "translation": {
            "Initiation": ["AUG start codon","small subunit loading","start-site context"],
            "Elongation": ["tRNA anticodon pairing","peptide bond formation","ribosome translocation"],
        "Termination": ["stop codon recognition","release factor action"]},
        "transcription": {
            "Initiation": ["promoter binding","RNA Pol II recruitment"],
            "Elongation": ["RNA chain elongation","nucleotide addition"],
            "Termination": ["termination signal","poly(A) processing"]},
        "replication": {
            "Leading strand": ["continuous synthesis","polymerase processivity"],
            "Lagging strand": ["Okazaki fragments","RNA primers by primase"]}
    }
    key = "translation" if "transla" in topic else "transcription" if "transcrip" in topic else "replication" if "replic" in topic else "translation"
    rules = RULES[key]
    bins = list(rules.keys())
    all_terms = [(t,b) for b,ts in rules.items() for t in ts]
    random.Random(new_seed()).shuffle(all_terms)
    chosen = all_terms[:4]
    terms = [t for t,_ in chosen]
    mapping = {t:b for t,b in chosen}
    hints = {t: "Match the mechanism step described on the slide." for t in terms}
    title = f"Match items for {key.title()}"
    instr = "Drag each item to the correct category."
    return title, instr, bins[:3], terms, mapping, hints

# ---------------------- FITB: LLM-first application generator ----------------------
def llm_generate_fitb(scope: str, prompt: str, style: Dict[str,Any]):
    if not scope.strip():
        return None
    nonce = _nonce()
    sys = "Generate unique, application-focused FITB items grounded ONLY in the provided slide excerpts. Never reveal answers."
    header = (
        f"Create 4 fill-in-the-blank items that require application (not recall) for an introductory biology course.\n\n"
        f"Topic (student prompt): \"{prompt}\"\n"
        f"Novelty nonce: {nonce}\n\n"
        "Slide excerpts (authoritative facts — cite page numbers if present):\n"
        + "\"\"\"\n" + scope + "\n\"\"\"\n\n"
    )
    constraints = (
        "Constraints (must follow ALL):\n"
        "- Each item starts with a SHORT SCENARIO (condition, mutation, inhibitor, or experiment). Keep total 12–22 words.\n"
        "- Exactly ONE blank (_____). Answer must be ONE WORD (or hyphenated) that appears in the excerpts.\n"
        "- Avoid definitional phrasing ('is called', 'is defined as').\n"
        "- Use plain, concrete language; intro level only.\n"
        "- Provide a short hint and slide_refs (list of small integers).\n\n"
        "Return STRICT JSON array of 4 objects like:\n"
        "[{\"stem\":\"short scenario ... _____ ...\",\"answers\":[\"word\"],\"hint\":\"short hint\",\"slide_refs\":[2]}, ...]"
    )
    data = chat_json(sys, header+constraints, max_tokens=900, temperature=0.55, seed=new_seed()%1_000_000)
    if not isinstance(data, list) or len(data)!=4:
        return None
    # validate
    low_scope = (scope or "").lower()
    out = []
    for it in data:
        stem = (it.get("stem") or "").strip()
        ans  = it.get("answers") or []
        if not stem or "_" not in stem: return None
        if not isinstance(ans, list) or not ans: return None
        a0 = (ans[0] if isinstance(ans[0], str) else "").strip()
        if not a0 or a0.lower() not in low_scope: return None
        # scenario token heuristic
        if not re.search(r"(mutation|mutant|inhibitor|drug|experiment|assay|condition|treatment|stress)", stem.lower()):
            return None
        out.append({"stem": stem, "answers":[a0], "hint": it.get("hint",""), "slide_refs": it.get("slide_refs", [])})
    return out

# ---------------------- FITB fallback (application-style) ----------------------
def fitb_fallback(prompt: str) -> List[Dict[str,Any]]:
    t = (prompt or "").lower()
    rng = random.Random(new_seed())
    if "replic" in t:
        pool = [
            {"stem":"A helicase inhibitor would collapse the replication _____ .","answers":["fork"],"hint":"Unwinding is required.","slide_refs":[3]},
            {"stem":"Loss of primase would block synthesis on the _____ strand.","answers":["lagging"],"hint":"Many primers needed.","slide_refs":[10]},
            {"stem":"Defective ligase leaves unsealed _____ between fragments.","answers":["nicks"],"hint":"Backbone joining.","slide_refs":[14]},
            {"stem":"A proofreading defect increases the _____ rate during synthesis.","answers":["mutation"],"hint":"Fidelity issue.","slide_refs":[16]},
        ]
    elif "transla" in t:
        pool = [
            {"stem":"If release factors are missing, the polypeptide does not _____ at a stop codon.","answers":["release"],"hint":"Termination fails.","slide_refs":[15]},
            {"stem":"An antibiotic that freezes EF-G halts ribosome _____ along mRNA.","answers":["translocation"],"hint":"Movement step.","slide_refs":[10]},
            {"stem":"A start codon mutation would most directly prevent ribosome _____ .","answers":["initiation"],"hint":"Start-site selection.","slide_refs":[6]},
            {"stem":"A tRNA anticodon mismatch disrupts accurate codon _____ .","answers":["recognition"],"hint":"Decoding step.","slide_refs":[9]},
        ]
    else:
        pool = [
            {"stem":"In hypotonic medium, water movement across the membrane _____ .","answers":["increases"],"hint":"Osmosis.","slide_refs":[5]},
            {"stem":"Blocking a carrier locked in one conformation prevents _____ access.","answers":["alternating"],"hint":"Transport cycle.","slide_refs":[12]},
            {"stem":"Removing the Shine–Dalgarno/Kozak context reduces ribosome _____ .","answers":["recruitment"],"hint":"Start-site context.","slide_refs":[4]},
            {"stem":"High salt most strongly disrupts _____ interactions between side chains.","answers":["ionic"],"hint":"Charge screening.","slide_refs":[7]},
        ]
    rng.shuffle(pool)
    return pool[:4]

# ---------------------- Novelty guard ----------------------
def remember_unique(tag: str, obj: Any, keep:int=10) -> bool:
    try:
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        payload = str(obj)
    dig = hashlib.sha1(payload.encode("utf-8","ignore")).hexdigest()
    key = f"seen_{tag}"
    seen = st.session_state.get(key, [])
    if dig in seen:
        return False
    st.session_state[key] = (seen + [dig])[-keep:]
    return True

# ---------------------- UI ----------------------
st.title("Cell Bio Tutor — Activity Generator")

prompt = st.text_input("Enter a topic (e.g., “DNA replication”, “Translation”):", key="prompt")
gen = st.button("Generate")

col1, col2 = st.columns(2)

if gen:
    with st.spinner("Fetching slides and generating activities..."):
        scope, refs = build_scope(prompt or "")
        cues = load_style_cues()
        # DnD
        dnd = llm_generate_dnd(scope, prompt or "", cues)
        if dnd is None:
            dnd = dnd_fallback(scope, prompt or "")
        title, instr, bins, terms, mapping, hints = dnd
        # ensure uniqueness
        if not remember_unique("dnd", {"bins":bins,"terms":terms,"mapping":mapping}):
            dnd = dnd_fallback(scope, prompt or "")
            title, instr, bins, terms, mapping, hints = dnd

        # FITB
        fitb = llm_generate_fitb(scope, prompt or "", cues)
        if not fitb:
            fitb = fitb_fallback(prompt or "")
        if not remember_unique("fitb", {"items":fitb}):
            fitb = fitb_fallback(prompt or "")

    # ---- Render DnD ----
    with col1:
        st.subheader("Drag & Drop")
        st.caption("Drag each item to its correct category.")
        st.write(f"**{title}**")
        cols = st.columns(len(bins))
        for i,b in enumerate(bins):
            with cols[i]:
                st.markdown(f"**{b}**")
                # Show which terms map here as solution rationale collapsed
        # Drag UI: Streamlit doesn't have native drag/drop; emulate selection
        choices = {}
        for t in terms:
            choices[t] = st.selectbox(f"Place: {t}", options=["—"] + bins, key=f"dnd_{_sha(t)}")
            if st.button(f"Hint for '{t}'", key=f"hint_{_sha(t)}"):
                st.info(hints.get(t,"Think about the mechanism step described on the slide."))
        if st.button("Check DnD answers"):
            correct = sum(1 for t in terms if choices.get(t)==mapping.get(t))
            if correct == len(terms):
                st.success("Perfect! 🎉 All items placed correctly.")
            else:
                st.warning(f"{correct}/{len(terms)} correct. Review the slide snippet and try again.")

    # ---- Render FITB ----
    with col2:
        st.subheader("Fill in the Blank")
        ans_inputs = []
        for i,it in enumerate(fitb, start=1):
            st.markdown(f"**Q{i}.** {it['stem']}")
            ans = st.text_input("Your answer:", key=f"fitb_{i}")
            if st.button("Hint", key=f"fitb_hint_{i}"):
                st.info(it.get("hint",""))
            if st.button("Check", key=f"fitb_check_{i}"):
                # flexible check: case-insensitive and allow minor variations (apostrophes)
                exp = [a.lower().replace("’","'").strip() for a in it.get("answers",[])]
                user = (ans or "").lower().replace("’","'").strip()
                # small normalization: allow 3' -> 3prime, etc.
                user_norm = user.replace("3'", "3prime").replace("5'", "5prime")
                exp_norm = [a.replace("3'", "3prime").replace("5'", "5prime") for a in exp]
                ok = (user in exp) or (user_norm in exp_norm)
                if ok:
                    st.success("That’s right! 🎉")
                else:
                    # supportive feedback using simple heuristic
                    st.warning("Not quite. Think about the scenario—what mechanism step would change here?")
            st.divider()

    # ---- Reference note ----
    if refs:
        # Show unique files and pages referenced
        by_file = {}
        for f,p in refs:
            by_file.setdefault(f, set()).add(p)
        ref_strs = [f"{f.split('/')[-1]}, p. {', '.join(str(x) for x in sorted(ps))}" for f,ps in by_file.items()]
        st.caption("Sources: " + " | ".join(ref_strs))
