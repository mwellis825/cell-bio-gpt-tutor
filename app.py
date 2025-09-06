
import streamlit as st
import json, re, os, time, random, hashlib, io
from typing import List, Dict, Any, Tuple, Optional
import requests

# ===================== Page =====================
st.set_page_config(page_title="Cell Bio Tutor — Activities", layout="wide")
st.title("Cell Bio Tutor — Activities")

# ===================== Repo =====================
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

def build_scope(prompt: str, max_chars:int=7000) -> Tuple[str, List[Tuple[str,int]]]:
    decks = pick_decks(prompt)
    refs = []
    chunks = []
    terms = [w for w in re.findall(r"[a-z0-9]+",(prompt or "").lower()) if len(w)>2]
    for path in decks:
        b = gh_read_bytes(path)
        if not b: continue
        pages = extract_pdf_text(b)
        scored = []
        for i,txt in enumerate(pages[:24], start=1):
            tl = (txt or "").lower()
            s = sum(2 for t in terms if t in tl) + (1/(i+2))
            scored.append((s,i,txt))
        scored.sort(key=lambda x:x[0], reverse=True)
        for s,i,txt in scored[:4]:
            if txt and len("".join(chunks)) < max_chars:
                chunks.append(f"[{path} p.{i}] {txt.strip()}")
                refs.append((path,i))
    return "\n\n".join(chunks)[:max_chars], refs

# ===================== OpenAI (optional) =====================
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

def extract_json(raw: str) -> Any:
    m = re.search(r"\[[\s\S]*\]|\{[\s\S]*\}", raw)
    return json.loads(m.group(0)) if m else json.loads(raw)

def chat_json(system: str, user: str, *, max_tokens=900, temperature=0.45, seed=None) -> Optional[Any]:
    client = get_openai()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"),
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed if isinstance(seed,int) else None,
        )
        txt = resp.choices[0].message.content
        return extract_json(txt)
    except Exception:
        return None

# ===================== Topic mapping & guards =====================
def topic_of(prompt: str) -> Optional[str]:
    t = (prompt or "").lower()
    if "transla" in t: return "translation"
    if "transcrip" in t: return "transcription"
    if "replic" in t: return "replication"
    if "bond" in t: return "chemical bonds"
    return None

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
        "Covalent bond": ["covalent","shared electron pair","peptide bond","disulfide bond","strong bond"],
        "Ionic bond":    ["ionic","electron transfer","charge attraction","salt bridge","electrostatic attraction","cation-anion"],
        "Hydrogen bond": ["hydrogen bond","partial charges","h-bond","between n-h and o","backbone hydrogen bonds"]
    }
}

# ===================== LLM: DnD =====================
def llm_dnd(scope: str, prompt: str) -> Optional[Tuple[str,str,List[str],List[str],Dict[str,str],Dict[str,str]]]:
    if not scope.strip(): return None
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
    if not all(isinstance(b,str) and len(b.split())<=4 for b in bins): return None
    if not (isinstance(terms,list) and len(terms)==4 and all(isinstance(t,str) and 3<=len(t.split())<=8 for t in terms)): return None
    if not (isinstance(mapping,dict) and all(t in mapping and mapping[t] in bins for t in terms)): return None
    # topic guardrails
    if topic=="translation":
        for t in terms:
            tl = canon(t); dest = canon(mapping.get(t,""))
            if "stop codon" in tl and "termin" not in dest and "release" not in dest: return None
            if ("start codon" in tl or "aug" in tl) and "init" not in dest: return None
            if any(k in tl for k in ["peptidyl transferase","translocation","ef-tu","ef g","ef-g"]) and "elong" not in dest: return None
    if topic=="chemical bonds":
        for t in terms:
            tl = canon(t); dest = canon(mapping.get(t,""))
            if any(k in tl for k in ["electron transfer","cation","anion","salt bridge","electrostatic"]) and "ionic" not in dest: return None
            if any(k in tl for k in ["shared electron pair","peptide bond","disulfide"]) and "covalent" not in dest: return None
            if any(k in tl for k in ["hydrogen bond","h-bond","partial charge","partial charges","n-h","backbone hydrogen"]) and "hydrogen" not in dest: return None
    # uniqueness
    payload = {"bins":bins,"terms":terms,"mapping":mapping}
    dig = sha(payload)
    seen = st.session_state.get("_seen_dnd", set())
    if dig in seen: return None
    seen.add(dig); st.session_state["_seen_dnd"] = seen

    title = data.get("title") or "Match items to slide-based labels"
    instr = "Drag each item to the correct category."
    return title, instr, bins, terms, mapping, hints

def dnd_fallback(prompt: str) -> Tuple[str,str,List[str],List[str],Dict[str,str],Dict[str,str]]:
    topic = topic_of(prompt) or "translation"
    rules = CANON.get(topic, {})
    bins = list(rules.keys())[:3]
    rng = random.Random(new_seed())
    if topic == "chemical bonds":
        curated = {
            "Covalent bond": ["shared electron pair","peptide bond","disulfide bond"],
            "Ionic bond": ["electron transfer","salt bridge","cation-anion attraction"],
            "Hydrogen bond": ["partial charges attraction","backbone hydrogen bonds","between N-H and O"]
        }
        pool = [(b, t) for b, ts in curated.items() for t in ts]
    else:
        pool = []
        for b, kws in rules.items():
            picks = [k for k in kws if " " in k] or kws
            rng.shuffle(picks)
            for k in picks[:3]:
                pool.append((b, k))
    rng.shuffle(pool)
    chosen = pool[:4]
    terms = [t for _,t in chosen]
    mapping = {t:b for b,t in chosen}
    hints = {t:"Use the literal concept from the slides." for t in terms}
    return f"Match items for {topic.title()}", "Drag each item to the correct category.", bins, terms, mapping, hints

def evaluate_dnd(choices: Dict[str,str], mapping: Dict[str,str]) -> Tuple[int,int,List[str]]:
    total = len(mapping); wrong = []
    nmapping = {canon(t): canon(b) for t,b in mapping.items()}
    for t, chosen in choices.items():
        if canon(chosen) != nmapping.get(canon(t)):
            wrong.append(t)
    return (total - len(wrong)), total, wrong

# ===================== LLM: FITB =====================
def llm_fitb(scope: str, prompt: str) -> Optional[List[Dict[str,Any]]]:
    if not scope.strip(): return None
    n = nonce()
    sys = "Generate unique, application-focused FITB items grounded ONLY in the slides. Never reveal answers."
    user = f"""Create 4 Fill-in-the-Blank items for introductory biology.

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
    data = chat_json(sys, user, temperature=0.5, seed=int(time.time())%1_000_000)
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
    dig = sha(out)
    seen = st.session_state.get("_seen_fitb", set())
    if dig in seen: return None
    seen.add(dig); st.session_state["_seen_fitb"] = seen
    return out

def fitb_fallback(prompt: str) -> List[Dict[str,Any]]:
    t = (prompt or "").lower()
    rng = random.Random(new_seed())
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
            {"stem":"High salt strengthens _____ interactions between cations and anions in crystals.","answers":["ionic"],"hint":"Electrostatic attraction."},
            {"stem":"A peptide linkage is a type of _____ bond between atoms.","answers":["covalent"],"hint":"Shared electron pair."},
            {"stem":"Secondary structure in proteins is stabilized largely by _____ bonds.","answers":["hydrogen"],"hint":"Backbone interactions."},
            {"stem":"Breaking a disulfide requires disrupting a _____ bond.","answers":["covalent"],"hint":"S–S linkage."},
        ]
    else:
        pool = [
            {"stem":"If a channel is blocked, _____ diffusion falls across the membrane.","answers":["facilitated"],"hint":"Through pores."},
            {"stem":"ATP depletion halts primary active _____ by pumps.","answers":["transport"],"hint":"Energy dependence."},
            {"stem":"A carrier stuck in one state cannot undergo _____ access.","answers":["alternating"],"hint":"Transport cycle."},
            {"stem":"Hypertonic medium causes water _____ from the cell.","answers":["efflux"],"hint":"Osmosis direction."},
        ]
    rng.shuffle(pool)
    out = pool[:4]
    dig = sha(out)
    seen = st.session_state.get("_seen_fitb", set())
    if dig in seen:
        rng.shuffle(pool)
        out = pool[:4]
    seen.add(sha(out)); st.session_state["_seen_fitb"] = seen
    return out

# ===================== UI =====================
prompt = st.text_input("Enter a topic (e.g., DNA replication, Translation, Chemical bonds)")
generate = st.button("Generate")

if generate:
    # Reset per-click so a completely new activity is attempted every time
    st.session_state["_seen_fitb"] = set()
    st.session_state["_seen_dnd"] = set()

    with st.spinner("Building slide scope and activities..."):
        scope, refs = build_scope(prompt or "")
        # Try LLM first, else safe fallback — FITB FIRST
        fitb = llm_fitb(scope, prompt or "") or fitb_fallback(prompt or "")
        # DnD SECOND
        dnd  = llm_dnd(scope, prompt or "") or dnd_fallback(prompt or "")
        title, instr, bins, terms, mapping, hints = dnd

    col1, col2 = st.columns(2)

    # ---- Activity 1: FITB ----
    with col1:
        st.subheader("Activity 1: Fill in the Blank")
        for i,it in enumerate(fitb, start=1):
            st.markdown(f"**Q{i}.** {it['stem']}")
            ans = st.text_input("Your answer:", key=f"fitb_{i}")
            if st.button(f"Hint for Q{i}", key=f"fitb_hint_{i}"):
                st.info(it.get("hint",""))
            if st.button(f"Check Q{i}", key=f"fitb_check_{i}"):
                exp = [a.lower().replace("’","'").strip() for a in it.get("answers",[])]
                user = (ans or "").lower().replace("’","'").strip()
                user_n = user.replace("5'", "5prime").replace("3'", "3prime")
                exp_n  = [a.replace("5'", "5prime").replace("3'", "3prime") for a in exp]
                ok = (user in exp) or (user_n in exp_n)
                if ok:
                    st.success("That’s right! 🎉")
                else:
                    st.warning("Not quite. Think about the scenario and which mechanism changes.")
            st.divider()

    # ---- Activity 2: DnD ----
    with col2:
        st.subheader("Activity 2: Drag & Drop")
        st.markdown(f"**{title}**")
        st.caption("Drag each item to the correct category.")
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

    # ---- References ----
    if refs:
        by = {}
        for f,p in refs:
            by.setdefault(f, set()).add(p)
        refs_str = " | ".join(f"{f.split('/')[-1]} p.{', '.join(str(x) for x in sorted(ps))}" for f,ps in by.items())
        st.caption("Source slides: " + refs_str)
