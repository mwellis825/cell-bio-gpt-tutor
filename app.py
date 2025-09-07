import streamlit as st
import json, re, os, time, random, hashlib, io
from typing import List, Dict, Any, Tuple, Optional
import requests

# ===================== Page =====================
st.set_page_config(page_title="Course Tutor — Activities", layout="wide")
st.title("Course Tutor — Activities")

# ===================== Repo (slides from GitHub) =====================
GITHUB_USER   = "mwellis825"
GITHUB_REPO   = "cell-bio-gpt-tutor"
GITHUB_BRANCH = "main"
SLIDES_DIR    = "slides"
API = "https://api.github.com"
HEADERS = {"Accept":"application/vnd.github+json"}

def http_json(url: str):
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.json()

def http_bytes(url: str) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=35)
    r.raise_for_status()
    return r.content

def gh_tree():
    try:
        data = http_json(f"{API}/repos/{GITHUB_USER}/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1")
        return data.get("tree", [])
    except Exception:
        return []

def gh_read_bytes(path: str) -> Optional[bytes]:
    try:
        data = http_json(f"{API}/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}")
        if isinstance(data, dict) and data.get("download_url"):
            return http_bytes(data["download_url"])
    except Exception:
        pass
    return None

# ===================== Helpers =====================
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

def extract_pdf_text(pdf_bytes: bytes) -> List[str]:
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

# ===================== Scope building =====================
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

def build_scope(prompt: str, max_chars:int=8000) -> Tuple[str, List[Tuple[str,int]]]:
    decks = pick_decks(prompt)
    refs = []
    chunks = []
    terms = [w for w in re.findall(r"[a-z0-9]+",(prompt or "").lower()) if len(w)>2]
    for path in decks:
        b = gh_read_bytes(path)
        if not b: continue
        pages = extract_pdf_text(b)
        scored = []
        for i,txt in enumerate(pages[:30], start=1):
            tl = (txt or "").lower()
            s = sum(2 for t in terms if t in tl) + (1/(i+2))
            scored.append((s,i,txt))
        scored.sort(key=lambda x:x[0], reverse=True)
        for s,i,txt in scored[:5]:
            if txt and len("".join(chunks)) < max_chars:
                chunks.append(f"[{path} p.{i}] {txt.strip()}")
                refs.append((path,i))
    return "\n\n".join(chunks)[:max_chars], refs

# ===================== Sentence/heading extraction =====================
HEADING_RE = re.compile(r"^(?:[A-Z][A-Z0-9 \-:\/]{3,}|[A-Z][a-z]+(?: [A-Z][a-z]+){0,4}:?)$")
def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text or "")
    return [re.sub(r"\s+"," ",p).strip() for p in parts if p and len(p.strip()) > 25]

def candidate_bins_from_scope(scope: str, n:int=3) -> List[str]:
    lines = [ln.strip() for ln in scope.splitlines()]
    heads = []
    for ln in lines:
        ln2 = re.sub(r"^\[[^\]]+\]\s*", "", ln).strip()
        if HEADING_RE.match(ln2) and 3 <= len(ln2.split()) <= 4:
            heads.append(ln2.strip(":"))
    if len(heads) < n:
        caps = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b", scope)
        freq = {}
        for c in caps:
            freq[c] = freq.get(c,0)+1
        more = [x for x,_ in sorted(freq.items(), key=lambda kv: (-kv[1], len(kv[0])) )]
        for m in more:
            if m not in heads and 2 <= len(m.split()) <= 4:
                heads.append(m)
                if len(heads) >= n: break
    uniq = []
    seen = set()
    for h in heads:
        k = canon(h)
        if k not in seen:
            seen.add(k); uniq.append(h)
        if len(uniq) == n: break
    return uniq[:n]

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
    return idx

def sentence_snippets(scope: str) -> List[str]:
    sents = split_sentences(scope)
    seen = set(); out = []
    for s in sents:
        k = s.lower().strip()
        if k not in seen:
            seen.add(k); out.append(s)
    return out[:120]

# ===================== UI =====================
prompt = st.text_input("Enter a topic")
generate = st.button("Generate")

if generate:
    st.session_state["_seen_fitb"] = set()
    st.session_state["_seen_dnd"] = set()
    st.session_state["prompt"] = prompt or ""

    with st.spinner("Building activities..."):
        scope, refs = build_scope(prompt or "")
        if not scope.strip():
            st.error("No slide text found.")
            st.stop()
        bins_probe = candidate_bins_from_scope(scope, n=3)
        st.session_state["_scope_idx"] = keyword_index_from_scope(bins_probe, scope)

        # Placeholder: FITB & DnD generation would be here
        st.write("Scope built with references:", refs)
