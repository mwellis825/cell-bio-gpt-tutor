import streamlit as st
import json, re, os, time, random, hashlib, io, base64
from typing import List, Dict, Any, Tuple, Optional
import requests

# ──────────────────────────── Page ────────────────────────────
st.set_page_config(page_title="Course Tutor — Activities", layout="wide")
st.title("Course Tutor — Activities")

# ──────────────────────────── GitHub slides ────────────────────────────
GITHUB_USER   = "mwellis825"
GITHUB_REPO   = "cell-bio-gpt-tutor"
GITHUB_BRANCH = "main"
SLIDES_DIR    = "slides"
API = "https://api.github.com"
HEADERS = {"Accept":"application/vnd.github+json"}

# ──────────────────────────── Caching helpers ────────────────────────────
@st.cache_data(show_spinner=False)
def http_json(url: str):
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def http_bytes(url: str) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=35)
    r.raise_for_status()
    return r.content

@st.cache_data(show_spinner=False)
def gh_tree():
    try:
        data = http_json(f"{API}/repos/{GITHUB_USER}/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1")
        return data.get("tree", [])
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def gh_read_bytes(path: str) -> Optional[bytes]:
    try:
        data = http_json(f"{API}/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}")
        if isinstance(data, dict) and data.get("download_url"):
            return http_bytes(data["download_url"])
    except Exception:
        pass
    return None

# ──────────────────────────── Helpers ────────────────────────────
def new_seed() -> int:
    return int(time.time()*1000) % 1_000_000

def nonce() -> str:
    return f"{int(time.time()*1e6)}_{new_seed()}"

def canon(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def sha(obj: Any) -> str:
    try:
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        payload = str(obj)
    return hashlib.sha1(payload.encode("utf-8","ignore")).hexdigest()

@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_bytes_b64: str) -> List[str]:
    try:
        import PyPDF2
        b = base64.b64decode(pdf_bytes_b64.encode())
        pages = []
        with io.BytesIO(b) as f:
            r = PyPDF2.PdfReader(f)
            for p in r.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
        return pages
    except Exception:
        return [""]

def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text or "")
    return [re.sub(r"\s+"," ",p).strip() for p in parts if p and len(p.strip()) > 25]

# ──────────────────────────── Scope building ────────────────────────────
def pick_decks(prompt: str, k:int=2) -> List[str]:
    prompt_l = (prompt or "").lower()
    words = [w for w in re.findall(r"[a-z0-9]+", prompt_l) if len(w)>2]
    tree = gh_tree()
    files = [t["path"] for t in tree if t.get("type")=="blob" and t["path"].lower().startswith(SLIDES_DIR+"/") and t["path"].lower().endswith(".pdf")]
    def score(p):
        base = p.split("/")[-1].lower()
        return sum(1 for w in words if w in base)
    files.sort(key=lambda p:(-score(p), len(p)))
    return files[:k] or files[:2]

def build_scope(prompt: str, max_chars:int=6500) -> Tuple[str, List[Tuple[str,int]]]:
    decks = pick_decks(prompt, k=3)
    refs = []
    chunks = []
    terms = [w for w in re.findall(r"[a-z0-9]+",(prompt or "").lower()) if len(w)>2]
    rng = random.Random(new_seed())
    for path in decks:
        b = gh_read_bytes(path)
        if not b: continue
        b64 = base64.b64encode(b).decode()
        pages = extract_pdf_text(b64)
        scored = []
        for i,txt in enumerate(pages[:28], start=1):
            tl = (txt or "").lower()
            s = sum(3 for t in terms if t in tl) + (1/(i+2))
            scored.append((s,i,txt))
        scored.sort(key=lambda x:x[0], reverse=True)
        for s,i,txt in scored[:5]:
            if txt and len("".join(chunks)) < max_chars:
                chunks.append(f"[{path} p.{i}] {txt.strip()}")
                refs.append((path,i))
    # Shuffle chunks slightly to avoid determinism in first generation
    rng.shuffle(chunks)
    return "\n\n".join(chunks)[:max_chars], refs

# ──────────────────────────── OpenAI (optional) ────────────────────────────
def get_openai():
    key = os.environ.get("OPENAI_API_KEY","").strip()
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY","")  # type: ignore[attr-defined]
        except Exception:
            key = ""
    if not key: return None
    try:
        import openai
        openai.api_key = key
        return openai
    except Exception:
        return None

def _json_extract(raw: str) -> Any:
    m = re.search(r"\[[\s\S]*\]|\{[\s\S]*\}", raw)
    return json.loads(m.group(0)) if m else json.loads(raw)

def chat_json(system: str, user: str, *, max_tokens=800, temperature=0.45, seed=None) -> Optional[Any]:
    client = get_openai()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"),
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=max_tokens,   # lower for speed
            temperature=temperature,
            seed=seed if isinstance(seed,int) else None,
        )
        txt = resp.choices[0].message.content
        return _json_extract(txt)
    except Exception:
        return None

# ──────────────────────────── Domain-agnostic DnD validator ────────────────────────────
def dnd_self_consistency_check(scope: str, bins: List[str], terms: List[str], mapping: Dict[str,str]) -> Optional[Dict[str,Any]]:
    """Ask the LLM to audit the mapping using slide excerpts + general intro knowledge.
       Returns dict with corrected_mapping; None if invalid/ambiguous."""
    sys = "You are validating a classification task for an introductory course. Be strict and correct."
    user = f"""Slides (authoritative excerpts):
\"\"\"
{scope}
\"\"\"

Bins: {json.dumps(bins, ensure_ascii=False)}
Terms: {json.dumps(terms, ensure_ascii=False)}
Proposed mapping: {json.dumps(mapping, ensure_ascii=False)}

For EACH term, choose the SINGLE correct bin from the list. If more than one could reasonably fit, mark AMBIGUOUS.
Use both the slides and standard introductory knowledge. If the proposed mapping is wrong, correct it.

Return STRICT JSON:
{{
  "ok": true|false,
  "ambiguous_terms": ["t?"],
  "corrected_mapping": {{"term":"Bin", ...}}
}}"""
    data = chat_json(sys, user, temperature=0.15, max_tokens=550, seed=int(time.time())%1_000_000)
    if not isinstance(data, dict): return None
    if data.get("ok") is False: return None
    if data.get("ambiguous_terms"): return None
    corr = data.get("corrected_mapping", {})
    if set(corr.keys()) != set(terms): return None
    if not all(v in bins for v in corr.values()): return None
    return data

# ──────────────────────────── FITB generation ────────────────────────────
def llm_fitb(scope: str, prompt: str, avoid: Optional[set]=None) -> Optional[List[Dict[str,Any]]]:
    if not scope.strip(): return None
    n = nonce()
    sys = "Generate unique, application-focused FITB items grounded ONLY in the provided slides. Never reveal answers."
    user = f"""Create 6 Fill-in-the-Blank items for an INTRO course.

PROMPT: "{prompt}"
SLIDE EXCERPTS (authoritative):
\"\"\"
{scope}
\"\"\"

Constraints (ALL) — id:{n}
- Each item is a SHORT SCENARIO (perturbation/condition/experiment/observation) that requires inference.
- Exactly ONE blank (_____). Answer is ONE WORD (or hyphenated).
- 12–22 words; plain language; no definition phrasing.
- Provide a short helpful HINT per item.
Return STRICT JSON array of 6 objects: {{"stem":"...", "answers":["word"], "hint":"..."}}"""
    data = chat_json(sys, user, temperature=0.55, max_tokens=700, seed=int(time.time())%1_000_000)
    if not isinstance(data, list): return None
    out = []
    seen_stems = set()
    avoid = avoid or set()
    for it in data:
        stem = (it.get("stem") or "").strip()
        ans  = [a for a in (it.get("answers",[]) or []) if isinstance(a,str) and a.strip()]
        hint = (it.get("hint","") or "Focus on the mechanism implied by the scenario.").strip()
        if not stem or "____" not in stem: continue
        if stem.lower() in seen_stems: continue
        if sha(stem) in avoid: continue  # avoid repeats across clicks
        wc = len(re.findall(r"[A-Za-z0-9']+", stem))
        if not (12 <= wc <= 22): continue
        if re.search(r"\bis\b|\bare\b|\bmeans\b|\bdefined as\b", stem.lower()): continue
        if not ans: continue
        out.append({"stem": stem, "answers":[ans[0]], "hint": hint})
        seen_stems.add(stem.lower())
    if len(out) < 4: return None
    # pick 4 randomly to increase variety
    rng = random.Random(new_seed())
    rng.shuffle(out)
    picked = out[:4]
    # track hashes to avoid duplicates on next click
    used = st.session_state.get("_fitb_used_hashes", set())
    for x in picked: used.add(sha(x["stem"]))
    st.session_state["_fitb_used_hashes"] = used
    return picked

# ──────────────────────────── DnD generation with strong non-triviality ────────────────────────────
BANNED_VAGUE_LABELS = {"interaction","process","function","role","effect","feature","key term","core feature","thing","concept"}
TRIVIAL_TERM_PAT = re.compile(r"\b(function|role|about|in\s+\w+|is\s+when|is\s+the)\b", re.I)

def keyword_index_from_scope(bins: List[str], scope: str) -> Dict[str, set]:
    idx = {b: set() for b in bins}
    sents = split_sentences(scope)
    for b in bins:
        b_low = b.lower()
        for s in sents:
            if b_low in s.lower():
                toks = [t for t in re.findall(r"[A-Za-z0-9\-']+", s.lower()) if len(t) > 2]
                for t in toks:
                    idx[b].add(t)
        for tk in re.findall(r"[A-Za-z0-9\-']+", b.lower()):
            if len(tk) > 2:
                idx[b].add(tk)
    # prune ubiquitous tokens
    all_t = {}
    for b in bins:
        for t in idx[b]:
            all_t[t] = all_t.get(t,0)+1
    for b in bins:
        idx[b] = {t for t in idx[b] if all_t.get(t,0) <= max(1, len(bins)//2)}
    return idx

def _non_trivial_term(term: str, bins: List[str]) -> bool:
    """Reject terms that simply restate a bin or use trivial phrasing."""
    if TRIVIAL_TERM_PAT.search(term): 
        return False
    tl = re.findall(r"[A-Za-z0-9]{3,}", term.lower())
    for b in bins:
        bl = re.findall(r"[A-Za-z0-9]{3,}", b.lower())
        overlap = set(tl) & set(bl)
        if overlap:  # term contains bin words → trivial tautology
            return False
    return True

def llm_dnd(scope: str, prompt: str) -> Optional[Tuple[str,str,List[str],List[str],Dict[str,str],Dict[str,str]]]:
    if not scope.strip(): return None
    sys = "Generate one unambiguous drag-and-drop activity grounded ONLY in the provided slide excerpts. Never reveal answers."
    n = nonce()
    user = f"""Create ONE classification drag-and-drop for an intro student.

PROMPT: "{prompt}"
SLIDE EXCERPTS:
\"\"\"
{scope}
\"\"\"

Strict constraints — id:{n}
- BINS: exactly 3 concrete labels copied verbatim from excerpts (<=4 words). No vague labels.
- TERMS: exactly 4 short items (3–8 words). Each must include a DISCRIMINATING cue from the excerpts that favors exactly ONE bin.
- Avoid trivial terms that merely restate a bin name (e.g., "Role of tRNAs"). Use informative cues.
- Provide short non-revealing hints per item.

Return STRICT JSON only:
{{
  "title":"string",
  "bins":["B1","B2","B3"],
  "terms":["t1","t2","t3","t4"],
  "mapping":{{"t1":"B?","t2":"B?","t3":"B?","t4":"B?"}},
  "hints":{{"t1":"hint","t2":"hint","t3":"hint","t4":"hint"}}
}}"""
    data = chat_json(sys, user, temperature=0.5, max_tokens=700, seed=int(time.time())%1_000_000)
    if not isinstance(data, dict): return None
    bins  = data.get("bins") or []
    terms = data.get("terms") or []
    mapping = data.get("mapping") or {}
    hints = data.get("hints") or {}

    # Structure checks
    if not (isinstance(bins,list) and len(bins)==3): return None
    if not all(isinstance(b,str) and 1<=len(b.split())<=4 and canon(b) not in BANNED_VAGUE_LABELS for b in bins): return None
    if not (isinstance(terms,list) and len(terms)==4 and all(isinstance(t,str) and 3<=len(t.split())<=8 for t in terms)): return None
    if not (isinstance(mapping,dict) and all(t in mapping and mapping[t] in bins for t in terms)): return None

    # Triviality/tautology filter for terms
    if not all(_non_trivial_term(t, bins) for t in terms):
        return None

    # LLM self-consistency validator (fast)
    audit = dnd_self_consistency_check(scope, bins, terms, mapping)
    if audit is None:
        return None
    mapping = audit["corrected_mapping"]

    # Exclusivity with scope keywords (ties allowed)
    idx = keyword_index_from_scope(bins, scope)
    def bin_score(term: str, b: str) -> int:
        tl = canon(term)
        return sum(1 for k in idx.get(b,[]) if k in tl)
    for t in terms:
        mapped = mapping.get(t,"")
        ms = bin_score(t, mapped)
        others = [bin_score(t, b) for b in bins if b != mapped]
        if not (ms >= 0 and all(ms >= o for o in others)):
            return None

    # Uniqueness per click and across recent runs
    payload = {"bins":bins,"terms":terms,"mapping":mapping}
    dig = sha(payload)
    seen = st.session_state.get("_seen_dnd", set())
    if dig in seen: 
        return None
    seen.add(dig); st.session_state["_seen_dnd"] = seen

    title = data.get("title") or "Match items to slide-based labels"
    instr = "Drag each item to the correct category."
    return title, instr, bins, terms, mapping, hints

# ──────────────────────────── DnD evaluation ────────────────────────────
def evaluate_dnd(choices: Dict[str,str], mapping: Dict[str,str]) -> Tuple[int,int,List[str]]:
    total = len(mapping); wrong = []
    for t, chosen in choices.items():
        gold = mapping.get(t, "")
        if canon(chosen) == canon(gold):
            continue
        wrong.append(t)
    return (total - len(wrong)), total, wrong

# ──────────────────────────── FITB evaluation ────────────────────────────
def check_fitb(user_answer: str, expected: List[str]) -> bool:
    user = (user_answer or "").lower().replace("’","'").strip()
    user_n = user.replace("5'", "5prime").replace("3'", "3prime")
    norm = [a.lower().replace("’","'").strip() for a in expected]
    norm_n = [a.replace("5'", "5prime").replace("3'", "3prime") for a in norm]
    return (user in norm) or (user_n in norm_n)

# ──────────────────────────── UI ────────────────────────────
prompt = st.text_input("Enter a topic")
generate = st.button("Generate")

if generate:
    # Fresh run uniqueness sets
    st.session_state["_seen_fitb"] = set()
    st.session_state["_seen_dnd"] = set()
    st.session_state["prompt"] = prompt or ""

    with st.spinner("Building activities..."):
        scope, refs = build_scope(prompt or "")
        if not scope.strip():
            st.error("No slide text found from GitHub /slides.")
            st.stop()

        # FITB: supply avoid-set to reduce repeats across clicks
        avoid = st.session_state.get("_fitb_used_hashes", set())
        fitb = None
        for _ in range(2):  # try twice for speed
            fitb = llm_fitb(scope, prompt or "", avoid=avoid)
            if fitb: break

        # DnD: try up to 5 times but quit quickly on first valid (avoid the "bad first" issue)
        dnd = None
        for _ in range(5):
            out = llm_dnd(scope, prompt or "")
            if out:
                dnd = out; break

        if not fitb and not dnd:
            st.error("Could not generate activities. Please try a more specific prompt.")
            st.stop()

    col1, col2 = st.columns(2)

    # Activity 1: FITB (first)
    with col1:
        st.subheader("Activity 1: Fill in the Blank")
        if not fitb:
            st.info("FITB unavailable for this prompt.")
        else:
            for i,it in enumerate(fitb, start=1):
                st.markdown(f"**Q{i}.** {it['stem']}")
                ans = st.text_input("Your answer:", key=f"fitb_{i}")
                if st.button(f"Hint for Q{i}", key=f"fitb_hint_{i}"):
                    st.info(it.get("hint",""))
                if st.button(f"Check Q{i}", key=f"fitb_check_{i}"):
                    ok = check_fitb(ans, it.get("answers",[]))
                    st.success("That’s right! 🎉") if ok else st.warning("Close—re-read the scenario and the slide cues.")
                st.divider()

    # Activity 2: DnD (second)
    with col2:
        st.subheader("Activity 2: Drag & Drop")
        if not dnd:
            st.info("Drag-and-drop unavailable for this prompt.")
        else:
            title, instr, bins, terms, mapping, hints = dnd
            st.markdown(f"**{title}**")
            st.caption(instr)
            placements = {}
            for t in terms:
                placements[t] = st.selectbox(f"Place: {t}", options=["—"]+bins, key=f"dnd_{sha(t)}")
                if st.button(f"Hint: {t}", key=f"hint_{sha(t)}"):
                    st.info(hints.get(t,"Re-read the slide text."))
            if st.button("Check bins"):
                correct, total, wrong = evaluate_dnd(placements, mapping)
                if correct == total:
                    st.success("All bins correct! 🎉")
                else:
                    st.error(f"{correct}/{total} correct — try again.")
                    if wrong:
                        st.caption("Consider these again: " + ", ".join(wrong))

    # References
    if refs:
        by = {}
        for f,p in refs:
            by.setdefault(f, set()).add(p)
        refs_str = " | ".join(f"{f.split('/')[-1]} p.{', '.join(str(x) for x in sorted(ps))}" for f,ps in by.items())
        st.caption("Source slides: " + refs_str)
