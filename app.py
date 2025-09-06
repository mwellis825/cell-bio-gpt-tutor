
import streamlit as st
import json, re, os, time, random, hashlib, io
from typing import List, Dict, Any, Tuple, Optional
import requests

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Cell Bio Tutor", layout="wide")
st.title("Cell Bio Tutor — Activities")

# ---------------- GITHUB REPO ----------------
GITHUB_USER   = "mwellis825"
GITHUB_REPO   = "cell-bio-gpt-tutor"
GITHUB_BRANCH = "main"
SLIDES_DIR    = "slides"
API = "https://api.github.com"
HEADERS = {"Accept":"application/vnd.github+json"}

# ---------------- UTILITIES ----------------
def nonce() -> str:
    return f"{int(time.time()*1e6)}_{int(time.time()*1000)%1_000_000}"

def canon(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def sha(obj: Any) -> str:
    try:
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        payload = str(obj)
    return hashlib.sha1(payload.encode("utf-8","ignore")).hexdigest()

def remember(tag: str, obj: Any, keep:int=12) -> bool:
    d = sha(obj)
    key = f"_seen_{tag}"
    seen = st.session_state.get(key, [])
    if d in seen:
        return False
    st.session_state[key] = (seen + [d])[-keep:]
    return True

def http_json(url: str):
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.json()

def http_bytes(url: str) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.content

def gh_list(path: str):
    try:
        return http_json(f"{API}/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}")
    except Exception:
        return []

def gh_tree():
    try:
        data = http_json(f"{API}/repos/{GITHUB_USER}/{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1")
        return data.get("tree", [])
    except Exception:
        return []

def gh_read_text(path: str) -> Optional[str]:
    try:
        data = http_json(f"{API}/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}")
        if isinstance(data, dict) and data.get("download_url"):
            return http_bytes(data["download_url"]).decode("utf-8","ignore")
    except Exception:
        pass
    return None

def gh_read_bytes(path: str) -> Optional[bytes]:
    try:
        data = http_json(f"{API}/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}")
        if isinstance(data, dict) and data.get("download_url"):
            return http_bytes(data["download_url"])
    except Exception:
        pass
    return None

# ---------------- SLIDES SCOPE ----------------
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

def pick_decks(prompt: str, k:int=2) -> List[str]:
    prompt_l = (prompt or "").lower()
    words = [w for w in re.findall(r"[a-z0-9]+", prompt_l) if len(w)>2]
    tree = gh_tree()
    files = [t["path"] for t in tree if t.get("type")=="blob" and t["path"].lower().startswith(SLIDES_DIR+"/") and t["path"].lower().endswith(".pdf")]
    def score(p):
        base = p.split("/")[-1].lower()
        return sum(1 for w in words if w in base)
    files.sort(key=lambda p:(-score(p), len(p)))
    return files[:k]

def build_scope(prompt: str, max_chars:int=6500) -> Tuple[str, List[Tuple[str,int]]]:
    decks = pick_decks(prompt)
    refs = []
    chunks = []
    terms = [w for w in re.findall(r"[a-z0-9]+",(prompt or "").lower()) if len(w)>2]
    for path in decks:
        b = gh_read_bytes(path)
        if not b: continue
        pages = extract_pdf_text(b)
        scored = []
        for i,txt in enumerate(pages[:18], start=1):
            tl = (txt or "").lower()
            s = sum(2 for t in terms if t in tl) + (1/(i+2))
            scored.append((s,i,txt))
        scored.sort(key=lambda x:x[0], reverse=True)
        for s,i,txt in scored[:4]:
            if txt and len("".join(chunks)) < max_chars:
                chunks.append(f"[{path} p.{i}] {txt.strip()}")
                refs.append((path,i))
    return "\n\n".join(chunks)[:max_chars], refs

# ---------------- OPENAI (optional) ----------------
def get_openai():
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        key = os.environ.get("OPENAI_API_KEY")
    if not key: return None
    try:
        import openai
        openai.api_key = key
        return openai
    except Exception:
        return None

def extract_json(raw: str) -> Any:
    m = re.search(r"\[[\s\S]*\]|\{[\s\S]*\}", raw)
    return json.loads(m.group(0)) if m else json.loads(raw)

def chat_json(system: str, user: str, *, max_tokens=800, temperature=0.45, seed=None) -> Optional[Any]:
    client = get_openai()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed if isinstance(seed, int) else None,
        )
        txt = resp.choices[0].message.content
        return extract_json(txt)
    except Exception:
        return None

# ---------------- DnD GENERATION ----------------
VAGUE = {"process","interaction","function","role","effect","site","location","component","feature","category"}

def label_in_scope(label: str, scope: str) -> bool:
    return canon(label) in canon(scope)

def is_vague(label: str) -> bool:
    L = canon(label)
    return any(v in L for v in VAGUE) or len(L.split())>4

def topic_of(prompt: str) -> Optional[str]:
    t = (prompt or "").lower()
    if "transla" in t: return "translation"
    if "transcrip" in t: return "transcription"
    if "replic" in t: return "replication"
    if "bond" in t: return "chemical bonds"
    return None

# Guardrail keywords for evaluation/fallback
CANON = {
    "translation": {
        "Initiation":  ["initiation","start codon","aug","small subunit","shine-dalgarno","kozak","ribosome assembly"],
        "Elongation":  ["elongation","peptidyl transferase","translocation","ef-tu","ef-g","codon-anticodon","aa-trna"],
        "Termination": ["termination","stop codon","release factor","rf1","rf2","rf3"]
    },
    "transcription": {
        "Initiation":  ["initiation","promoter","tata","pol ii"],
        "Elongation":  ["elongation","rna chain","synthesis","nucleotide addition"],
        "Termination": ["termination","poly(a)","cleavage","termination signal"]
    },
    "replication": {
        "Leading strand": ["leading","continuous synthesis","polymerase epsilon","processivity"],
        "Lagging strand": ["lagging","okazaki","discontinuous","rna primer","primase"],
        "Origin":         ["origin","ori","replication bubble","helicase loading"]
    },
    "chemical bonds": {
        "Covalent bond": ["covalent","peptide","disulfide","electron sharing","shared electrons"],
        "Ionic bond":    ["ionic","electrostatic","charge attraction","salt bridge","exchanging electrons","electron transfer"],
        "Hydrogen bond": ["hydrogen","partial charge","polar interaction","h-bond"]
    }
}

def llm_dnd(scope: str, prompt: str) -> Optional[Tuple[str,str,List[str],List[str],Dict[str,str],Dict[str,str]]]:
    if not scope.strip(): return None
    # Encourage uniqueness with per-run nonce
    n = nonce()
    sys = "Generate one unambiguous drag-and-drop activity grounded ONLY in the provided slide excerpts. Never reveal answers."
    topic = topic_of(prompt)
    topic_bins_hint = ", ".join(CANON.get(topic, {}).keys()) if topic else ""
    user = f"""Create ONE classification drag-and-drop for an intro biology student.

STUDENT PROMPT: "{prompt}"
SLIDE EXCERPTS (exact labels/items MUST come from here):
\"\"\"
{scope}
\"\"\"

Constraints (all required) — id:{n}
- BINS: choose 3 concrete labels that appear literally in the excerpts (prefer headings/terms). Avoid vague labels.
- TERMS: choose exactly 4 short items (3–8 words) that each maps to exactly ONE bin.
- Provide one short, non-revealing hint per item.
- If topic known, preferred bins: {topic_bins_hint}

Return STRICT JSON only:
{{"title":"string",
  "bins":["B1","B2","B3"],
  "terms":["t1","t2","t3","t4"],
  "mapping":{{"t1":"B?","t2":"B?","t3":"B?","t4":"B?"}},
  "hints":{{"t1":"hint","t2":"hint","t3":"hint","t4":"hint"}}
}}"""
    data = chat_json(sys, user, temperature=0.4, seed=int(time.time())%1_000_000)
    if not isinstance(data, dict): return None
    bins  = data.get("bins") or []
    terms = data.get("terms") or []
    mapping = data.get("mapping") or {}
    hints = data.get("hints") or {}
    # structural guards
    if not (isinstance(bins,list) and len(bins)==3): return None
    if any(is_vague(b) for b in bins): return None
    if not all(label_in_scope(b, scope) for b in bins): return None
    if not (isinstance(terms,list) and len(terms)==4 and all(isinstance(t,str) and 3<=len(t.split())<=8 for t in terms)): return None
    if not (isinstance(mapping,dict) and all(t in mapping and mapping[t] in bins for t in terms)): return None
    # topic "never wrong" guards
    if topic=="translation":
        for t in terms:
            tl = canon(t)
            dest = canon(mapping.get(t,""))
            if "stop codon" in tl and "termin" not in dest and "release" not in dest:
                return None
            if ("start codon" in tl or "aug" in tl) and "init" not in dest:
                return None
            if any(k in tl for k in ["peptidyl transferase","translocation","ef-tu","ef g","ef-g"]) and "elong" not in dest:
                return None
    payload = {"bins":bins,"terms":terms,"mapping":mapping}
    if not remember("dnd", payload):
        return None
    return data.get("title") or "Match items to slide-based labels", "Drag each item to the correct category.", bins, terms, mapping, hints

def dnd_fallback(prompt: str) -> Tuple[str,str,List[str],List[str],Dict[str,str],Dict[str,str]]:
    topic = topic_of(prompt) or "translation"
    rules = CANON.get(topic, {})
    bins = list(rules.keys())[:3]
    rng = random.Random(int(time.time()*1000)%1_000_000)
    # choose one or two phrases per bin; then sample to 4 total
    pool = []
    for b, kws in rules.items():
        # prefer multi-word phrases
        picks = [k for k in kws if " " in k] or kws
        rng.shuffle(picks)
        for k in picks[:2]:
            pool.append((b, k))
    rng.shuffle(pool)
    chosen = pool[:4]
    terms = [t for _,t in chosen]
    mapping = {t:b for b,t in chosen}
    hints = {t:"Use the literal phrase from the slide." for t in terms}
    return f"Match items for {topic.title()}", "Drag each item to the correct category.", bins, terms, mapping, hints

# ---------------- DnD EVALUATION (no false negatives) ----------------
def build_keyword_index(bins: List[str], topic: Optional[str]) -> Dict[str,List[str]]:
    idx = {}
    for b in bins:
        idx[b] = []
        if topic and topic in CANON and b in CANON[topic]:
            idx[b] += CANON[topic][b]
    return idx

def evaluate_dnd(choices: Dict[str,str], mapping: Dict[str,str], bins: List[str], topic: Optional[str]) -> Tuple[int,int,List[str]]:
    """
    Compare with normalization + keyword fallback. Returns (correct, total, wrong_terms)
    A term is marked correct if:
      - chosen bin == mapping bin after normalization, OR
      - chosen bin keywords match the term more strongly than other bins (topic guard)
    """
    total = len(mapping)
    wrong = []
    # normalize baseline mapping
    nmapping = {canon(t): canon(b) for t,b in mapping.items()}
    nbins = {canon(b): b for b in bins}
    kw = build_keyword_index(bins, topic)
    for t, chosen in choices.items():
        ct = canon(t)
        cb = canon(chosen)
        mb = nmapping.get(ct)
        if mb == cb:
            continue
        # keyword fallback
        score = {}
        tl = canon(t)
        for b, klist in kw.items():
            score[b] = sum(1 for k in klist if canon(k) in tl)
        best = None
        if score:
            best = max(score, key=lambda x: score[x])
        if best and canon(best) == cb and score[best] > 0:
            # accept as correct if selected bin has strictly highest keyword score
            if list(sorted(score.values()))[-1] > (list(sorted(score.values()))[-2] if len(score)>1 else -1):
                continue
        wrong.append(t)
    return (total - len(wrong)), total, wrong

# ---------------- FITB GENERATION ----------------
def llm_fitb(scope: str, prompt: str) -> Optional[List[Dict[str,Any]]]:
    if not scope.strip(): return None
    n = nonce()
    sys = "Generate unique, application-focused FITB items grounded ONLY in the slides. Never reveal answers."
    user = f"""Create 4 Fill-in-the-Blank items for intro biology.

STUDENT PROMPT: "{prompt}"
SLIDE EXCERPTS (authoritative):
\"\"\"
{scope}
\"\"\"

Constraints (ALL) — id:{n}
- Each item is a SHORT SCENARIO (mutation/inhibitor/condition/experiment) that requires application, not recall.
- Exactly ONE blank (_____). Answer is ONE WORD (or hyphenated) that appears in the excerpts.
- 11–20 words; no commas/semicolons.
- Provide a short hint per item.
Return STRICT JSON array of 4 objects: {{"stem":"...", "answers":["word"], "hint":"..."}}"""
    data = chat_json(sys, user, temperature=0.45, seed=int(time.time())%1_000_000)
    if not isinstance(data, list): return None
    out = []
    low = (scope or "").lower()
    for it in data:
        stem = (it.get("stem") or "").strip()
        ans  = [a for a in (it.get("answers",[]) or []) if isinstance(a,str) and a.strip()]
        hint = (it.get("hint","") or "Use the term implied by the scenario.").strip()
        if not stem or "____" not in stem: return None
        wc = len(re.findall(r"[A-Za-z0-9']+", stem))
        if not (11 <= wc <= 22): return None
        if not ans or not any(a.lower() in low for a in ans): return None
        out.append({"stem": stem, "answers":[ans[0]], "hint": hint})
    if len(out) != 4: return None
    if not remember("fitb", out): return None
    return out

def fitb_fallback(prompt: str) -> List[Dict[str,Any]]:
    t = (prompt or "").lower()
    rng = random.Random(int(time.time()*1000)%1_000_000)
    if "replic" in t:
        pool = [
            {"stem":"A helicase inhibitor collapses the replication _____ under stress.","answers":["fork"],"hint":"Unwinding is blocked."},
            {"stem":"Without primase, synthesis stalls on the _____ strand.","answers":["lagging"],"hint":"Requires many primers."},
            {"stem":"Ligase loss leaves unsealed _____ after fragment synthesis.","answers":["nicks"],"hint":"Backbone joining."},
            {"stem":"If proofreading falters, the _____ rate rises in S phase.","answers":["mutation"],"hint":"Fidelity issue."},
        ]
    elif "transla" in t:
        pool = [
            {"stem":"Release factors absent? The nascent chain will not _____ at a stop codon.","answers":["release"],"hint":"Termination step."},
            {"stem":"EF-G frozen by drug: ribosome _____ stops on mRNA.","answers":["translocation"],"hint":"Movement step."},
            {"stem":"A start-codon mutation most directly prevents ribosome _____ .","answers":["initiation"],"hint":"Start-site selection."},
            {"stem":"Wrong anticodon disrupts accurate codon _____ during decoding.","answers":["recognition"],"hint":"Matching step."},
        ]
    elif "bond" in t:
        pool = [
            {"stem":"In high salt, _____ interactions between charged side chains weaken.","answers":["ionic"],"hint":"Charge screening."},
            {"stem":"Heating a protein mostly weakens _____ bonds stabilizing secondary structure.","answers":["hydrogen"],"hint":"Helices/sheets."},
            {"stem":"A peptide linkage is a type of _____ bond.","answers":["covalent"],"hint":"Backbone link."},
            {"stem":"DNA’s backbone is connected by _____ linkages.","answers":["phosphodiester"],"hint":"Nucleotide connection."},
        ]
    else:
        pool = [
            {"stem":"If a channel is blocked, _____ diffusion falls across the membrane.","answers":["facilitated"],"hint":"Through pores."},
            {"stem":"ATP depletion halts primary active _____ by pumps.","answers":["transport"],"hint":"Energy dependence."},
            {"stem":"A carrier stuck in one state cannot undergo _____ access.","answers":["alternating"],"hint":"Transport cycle."},
            {"stem":"Hypertonic medium causes water _____ from the cell.","answers":["efflux"],"hint":"Osmosis direction."},
        ]
    rng.shuffle(pool)
    return pool[:4]

# ---------------- UI ----------------
prompt = st.text_input("Enter a topic (e.g., DNA replication, Translation, Chemical bonds)")
if st.button("Generate"):
    with st.spinner("Building slide scope and activities..."):
        scope, refs = build_scope(prompt or "")
        # DnD
        dnd = llm_dnd(scope, prompt or "") or dnd_fallback(prompt or "")
        title, instr, bins, terms, mapping, hints = dnd
        # FITB
        fitb = llm_fitb(scope, prompt or "") or fitb_fallback(prompt or "")

    # Render DnD
    st.subheader("Activity 1: Drag and Drop")
    st.markdown(f"**{title}**")
    st.caption("Drag each item to the correct category.")
    cols = st.columns(len(bins))
    placements = {}
    for t in terms:
        placements[t] = st.selectbox(f"Place: {t}", options=["—"]+bins, key=f"dnd_{sha(t)}")
        if st.button(f"Hint: {t}", key=f"hint_{sha(t)}"):
            st.info(hints.get(t,"Re-read the slide text."))
    if st.button("Check bins"):
        correct, total, wrong = evaluate_dnd(placements, mapping, bins, topic_of(prompt))
        if correct == total:
            st.success("All bins correct! 🎉")
        else:
            st.error(f"{correct}/{total} correct — try again.")
            if wrong:
                st.caption("Consider these again: " + ", ".join(wrong))

    st.divider()

    # Render FITB
    st.subheader("Activity 2: Fill in the Blank")
    for i,it in enumerate(fitb, start=1):
        st.markdown(f"**Q{i}.** {it['stem']}")
        ans = st.text_input("Your answer:", key=f"fitb_{i}")
        if st.button(f"Hint for Q{i}", key=f"fitb_hint_{i}"):
            st.info(it.get("hint",""))
        if st.button(f"Check Q{i}", key=f"fitb_check_{i}"):
            exp = [a.lower().replace("’","'").strip() for a in it.get("answers",[])]
            user = (ans or "").lower().replace("’","'").strip()
            # small normalization for 5'/3'
            user_n = user.replace("5'", "5prime").replace("3'", "3prime")
            exp_n  = [a.replace("5'", "5prime").replace("3'", "3prime") for a in exp]
            ok = (user in exp) or (user_n in exp_n)
            if ok:
                st.success("That’s right! 🎉")
            else:
                st.warning("Not quite. Think about the scenario and which mechanism step changes.")

    # References
    if refs:
        by = {}
        for f,p in refs:
            by.setdefault(f, set()).add(p)
        refs_str = " | ".join(f"{f.split('/')[-1]} p.{', '.join(str(x) for x in sorted(ps))}" for f,ps in by.items())
        st.caption("Source slides: " + refs_str)
