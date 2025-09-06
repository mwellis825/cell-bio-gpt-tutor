
import os
import re
import io
import json
import time
import random
import requests
from typing import List, Dict, Any, Tuple
import streamlit as st

st.set_page_config(page_title="Let's Practice Biology!", page_icon="🎓", layout="wide")
st.title("Let's Practice Biology!")

GITHUB_USER   = "mwellis825"
GITHUB_REPO   = "cell-bio-gpt-tutor"
GITHUB_BRANCH = "main"
SLIDES_DIR_GH = "slides"
EXAMS_DIR_GH  = "exams"

# ---------- Utils ----------
def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_json_block(s: str) -> str:
    s = _strip_code_fences(s)
    a_start, a_end = s.find("["), s.rfind("]")
    o_start, o_end = s.find("{"), s.rfind("}")
    if a_start != -1 and a_end != -1 and a_end > a_start:
        cand = s[a_start:a_end+1]
        try: json.loads(cand); return cand
        except Exception: pass
    if o_start != -1 and o_end != -1 and o_end > o_start:
        cand = s[o_start:o_end+1]
        try: json.loads(cand); return cand
        except Exception: pass
    return s

def new_seed() -> int:
    return int(time.time()*1000) ^ random.randint(0, 1_000_000)

# ---------- GitHub ----------
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
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception:
        try:
            import PyPDF2  # type: ignore
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            return "\n".join([(p.extract_text() or "") for p in reader.pages])
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
                txt = (_gh_fetch_raw(raw_url)).decode("utf-8","ignore")
            else:
                txt = ""
        except Exception:
            txt = ""
        if txt and len(txt.strip()) > 20:
            corpus.append(txt)
    return corpus

# ---------- Retrieval ----------
STOP = {"the","and","for","that","with","this","from","into","are","was","were","has","have","had","can","will","would","could","should",
"a","an","of","in","on","to","by","as","at","or","be","is","it","its","their","our","your","if","when","then","than","but",
"we","you","they","which","these","those","there","here","such","may","might","also","very","much","many","most","more","less"}

def _tokens_nostop(s: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z0-9']+", (s or "").lower()) if t not in STOP and len(t) > 2]

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text or "")
    return [re.sub(r"\s+"," ",p).strip() for p in parts if p and len(p.strip()) > 30]

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
    return "\n".join(sents)[:limit_chars]

# ---------- Topic fallback ----------
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

# ---------- OpenAI ----------
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

# ---------- Intro helpers ----------
BANNED_VAGUE_LABELS = {
    "process","processes","function","functions","interaction","interactions","role","roles",
    "category","categories","type","types","concept","concepts","example","examples",
    "misc","miscellaneous","other","others","general","specifics"
}

def _slide_terms(scope: str, max_terms: int = 24) -> list:
    scope = scope or ""
    cands = set()
    for m in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z0-9\-]+){0,3})\b", scope):
        if len(m.split()) >= 2 and len(m) <= 60:
            cands.add(m.strip())
    BIO_HINTS = [
        "covalent bond","ionic bond","hydrogen bond","van der Waals","glycolysis","PFK-1",
        "ATP synthase","electron transport chain","membrane potential","ion channel","carrier protein",
        "nucleotide excision repair","base excision repair","mismatch repair","DNA polymerase",
        "helicase","ligase","promoter","ribosome","microtubule","actin filament","intermediate filament",
        "tight junction","adherens junction","desmosome","gap junction","signal transduction"
    ]
    tl = scope.lower()
    for k in BIO_HINTS:
        if k.lower() in tl:
            cands.add(k)
    freq = []
    for t in cands:
        freq.append((scope.count(t), t))
    freq.sort(reverse=True)
    return [t for _, t in freq[:max_terms]]

def _is_vague_label(s: str) -> bool:
    t = (s or "").strip().lower()
    return (t in BANNED_VAGUE_LABELS) or (len(t.split()) <= 1 and t in {"effect","evidence","mechanism","pathway","feature"})

def _label_in_scope(label: str, scope: str) -> bool:
    return label and (label.lower() in (scope or "").lower())

def _intro_level_ok(stem: str, answers: list) -> bool:
    wc = len(re.findall(r"[A-Za-z0-9']+", stem or ""))
    if wc > 25: return False
    for a in answers or []:
        if len(a.split()) > 2: return False
    return True


# ---------- Answer normalization & feedback ----------
PRIME_SYMS = {
    "′":"'", "”":"\"", "“":"\"", "’":"'", "→":" to ", "–":"-", "—":"-"
}

def _normalize_text(s: str) -> str:
    """Lowercase, unify primes, remove extra punctuation/spaces."""
    s = (s or "").strip()
    for k,v in PRIME_SYMS.items():
        s = s.replace(k, v)
    s = s.lower()
    # words like 5' to 3' -> 5prime to 3prime
    s = re.sub(r"(\d)\s*'?(\s*to\s*|[-–—>→]\s*)(\d)\s*'?", r"\1prime to \3prime", s)
    s = re.sub(r"(\d)\s*'", r"\1prime", s)
    s = re.sub(r"prime\s*-\s*", "prime ", s)
    # hyphen/space normalization
    s = s.replace("semi-conservative","semiconservative").replace("semi conservative","semiconservative")
    s = s.replace("nucleo-some","nucleosome")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _canon_tokens(s: str) -> str:
    """Canonical token string for equality/substring checks."""
    s = _normalize_text(s)
    # remove trivial plurals
    s = re.sub(r"\b(\w+)s\b", r"\1", s)
    return s

def _direction_equiv(user: str, gold: str) -> bool:
    u = _normalize_text(user)
    g = _normalize_text(gold)
    # Accept many 5' to 3' variants
    def any_53(t: str) -> bool:
        return ("5prime to 3prime" in t) or ("5 to 3" in t) or ("5 3" in t) or ("5prime 3prime" in t)
    return any_53(u) and any_53(g)

def _fuzzy_close(a: str, b: str) -> float:
    # Simple token-based similarity
    ta = set(_canon_tokens(a).split())
    tb = set(_canon_tokens(b).split())
    if not ta or not tb: 
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter/union

COMMON_EQUIV = {
    "semiconservative": {"semiconservative","semi-conservative","semi conservative"},
    "nucleosome": {"nucleosome","nucleosomes"},
    "cytoplasm": {"cytoplasm","cytosol"},
    "rna polymerase ii": {"rna polymerase ii","pol ii","polymerase ii"},
}

def _expand_equivalents(ans_list):
    out = set()
    for a in ans_list or []:
        ca = _canon_tokens(a)
        out.add(ca)
        for base, alts in COMMON_EQUIV.items():
            if ca == base or ca in alts:
                out.update(alts)
    return out

def check_fitb_answer(user_answer: str, answers: list) -> tuple[bool, bool]:
    """Return (is_correct, is_close). is_close means near miss like 'histones' for 'nucleosome' or partial direction."""
    if not isinstance(answers, list):
        return (False, False)
    ca_set = _expand_equivalents(answers)
    ua = _canon_tokens(user_answer)
    if not ua:
        return (False, False)
    # Exact canonical match
    if ua in ca_set:
        return (True, False)
    # Direction equivalence (5' to 3')
    for a in answers:
        if _direction_equiv(user_answer, a):
            return (True, False)
    # Substring near matches and fuzzy Jaccard >= 0.5
    for a in ca_set:
        if ua in a or a in ua or _fuzzy_close(ua, a) >= 0.5:
            return (False, True)
    return (False, False)

def llm_feedback_for_fitb(scope: str, stem: str, answers: list, user_answer: str) -> str:
    """Ask the LLM for a short, encouraging feedback. Fallback to rule-based."""
    client = _openai_client()
    if client is None:
        # Rule-based fallback
        is_ok, is_close = check_fitb_answer(user_answer, answers)
        if is_ok:
            return "Great job — that matches the slide terminology."
        if is_close:
            return "Very close — refine the term to match the exact slide phrase."
        return "Not quite — re-read the slide line and look for the precise term."
    sys = "You are a supportive biology tutor. Respond in one sentence, 8–22 words. Be encouraging; never reveal the exact answer."
    ans_json = json.dumps(answers, ensure_ascii=False)
    prompt = f"""SLIDE EXCERPTS (authoritative):
\"\"\"
{scope}
\"\"\"

FITB prompt: {stem}
Student answered: "{user_answer}"
Correct answers (reference-only): {ans_json}

Give one short, encouraging feedback. If student is close (synonym/partial), say why and nudge them without revealing the answer. Avoid jargon and quotes."""
    try:
        reply = _chat(client, sys, prompt, max_tokens=80, temperature=0.4, seed=new_seed()%1_000_000)
        return (reply or "Keep going — think of the exact term used on the slide.").strip()
    except Exception:
        return "Keep going — think of the exact term used on the slide."


# ---------- Topic nudges & strict validators ----------
TOPIC_NUDGES = {
    "translation": {
        "bins_hint": ["Initiation", "Elongation", "Termination"],
        "terms_hint": ["AUG start codon", "peptidyl transferase activity", "stop codon recognition", "tRNA anticodon pairing"]
    },
    "transcription": {
        "bins_hint": ["Initiation", "Elongation", "Termination"],
        "terms_hint": ["promoter binding", "RNA synthesis", "polyadenylation signal", "termination signal"]
    },
    "dna repair": {
        "bins_hint": ["Base excision repair", "Nucleotide excision repair", "Mismatch repair"],
        "terms_hint": ["glycosylase removes base", "thymine dimer excision", "MutS recognition", "DNA ligase seals nick"]
    },
    "chemical bonds": {
        "bins_hint": ["Covalent bond", "Ionic bond"],
        "terms_hint": ["electron sharing", "charge attraction", "polar interaction", "hydrogen bond"]
    },
    "membrane transport": {
        "bins_hint": ["Channel", "Carrier", "Pump"],
        "terms_hint": ["passive ion flow", "alternating access", "ATP-dependent transport", "electrochemical gradient"]
    }
}

FORBID_IN_LABEL = set("""where when how occurs occur occurred differing differs different than because although whereas mainly mostly very kind type types role roles process processes function functions interaction interactions example examples other misc miscellaneous""".split())

def _bin_ok(lbl: str, scope: str) -> bool:
    if not isinstance(lbl, str): return False
    t = lbl.strip()
    if not t or len(t.split()) > 4: return False
    tl = t.lower()
    if any(w in FORBID_IN_LABEL for w in re.findall(r"[a-z']+", tl)): return False
    # Must appear in scope literally (case-insensitive) as word boundary
    if t.lower() not in (scope or "").lower():
        return False
    # Heuristic: prefer capitalized multiword or proper bio term
    return True

def _term_ok(term: str) -> bool:
    if not isinstance(term, str): return False
    t = term.strip()
    if not (3 <= len(t) <= 80): return False
    if "," in t or ";" in t: return False
    # Avoid vague starters
    if re.match(r"^(where|when|how|why|which|that)\b", t.lower()):
        return False
    # Prefer short phrase 3-10 words
    wc = len(re.findall(r"[A-Za-z0-9']+", t))
    return 3 <= wc <= 12
# ---------- Generators with triple-quoted f-strings ----------

def gen_dnd_from_scope(scope: str, prompt: str):
    client = _openai_client()
    if client is None or not scope.strip():
        return None

    topic = classify_topic(prompt)
    nudges = TOPIC_NUDGES.get(topic, {})
    anchor_terms = _slide_terms(scope, max_terms=24)
    allowed_hint = ", ".join(anchor_terms[:16])
    bins_hint = ", ".join(nudges.get("bins_hint", []))
    terms_hint = ", ".join(nudges.get("terms_hint", []))

    def _ask(seed_val: int, strict: bool):
        sys_prompt = "You create slide-anchored drag-and-drop activities. Use only the excerpt content. Never reveal answers."
        user_prompt = f"""Create ONE classification drag-and-drop activity for intro biology.

Slide excerpts:
\"\"\"
{scope}
\"\"\"

Student prompt: "{prompt}"

Design constraints (MUST follow ALL):
- BINS: choose 2–3 concrete labels that appear verbatim in the excerpts (slide headings or key terms). Avoid abstract words.
- TERMS: write exactly 4 short phrases (3–10 words) that each match exactly one bin. Avoid commas/semicolons.
- Keep language plain and literal; no trick wording.
- Provide one short hint per term (non-revealing).
- Allowed label pool (may use): {allowed_hint}
- Topic-specific nudge (optional): {bins_hint if bins_hint else "n/a"}; terms to consider: {terms_hint if terms_hint else "n/a"}
{("- Enforce: every BIN label must literally occur in the excerpts." if strict else "")}

Return STRICT JSON (no markdown):
{{
  "title": "string",
  "bins": ["string", ...],           // 2 or 3 labels
  "terms": ["string", "string", "string", "string"],  // 4 items
  "mapping": {{"TERM":"BIN"}},
  "hints": {{"TERM":"one short hint"}}
}}"""
        raw = _chat(client, sys_prompt, user_prompt, max_tokens=800, temperature=0.28 if strict else 0.32, seed=seed_val)
        return raw

    for attempt in range(2):
        raw = _ask(seed_val=(new_seed()%1_000_000), strict=(attempt==1))
        try:
            data = json.loads(_extract_json_block(raw))
        except Exception:
            data = {}
        bins  = data.get("bins") or []
        terms = data.get("terms") or []
        mapping = data.get("mapping") or {}
        hints = data.get("hints") or {}

        if not (isinstance(bins, list) and 2 <= len(bins) <= 3): continue
        if not (isinstance(terms, list) and len(terms) == 4): continue
        if not (isinstance(mapping, dict) and all(t in mapping for t in terms)): continue
        # Validate bins and terms
        if any(not _bin_ok(lbl, scope) for lbl in bins): continue
        if any(not _term_ok(t) for t in terms): continue
        # Ensure mapping uses only given bins
        if any(mapping.get(t) not in bins for t in terms): continue

        title = data.get("title") or "Match items to slide-based labels"
        instr = "Drag each item to the correct category."
        labels = bins
        draggables = terms
        answer = {t: mapping.get(t) for t in terms}
        hint_map = {t: (hints.get(t) or "Re-read the exact slide phrase.") for t in terms}
        return title, instr, labels, draggables, answer, hint_map

    return None


    anchor_terms = _slide_terms(scope, max_terms=24)
    allowed_hint = ", ".join(anchor_terms[:16])

    def _ask(seed_val: int, strict: bool):
        sys_prompt = "You create intro-level drag-and-drop activities grounded ONLY in the provided slide excerpts. Never reveal answers."
        user_prompt = f"""Create ONE classification drag-and-drop activity for first-year biology.

Slide excerpts:
\"\"\"
{scope}
\"\"\"

Student prompt: "{prompt}"

Design constraints (MUST follow ALL):
- BINS: choose exactly 3 concrete labels that appear verbatim as slide headings or key terms in the excerpts.
- TERMS: write exactly 6 short phrases (3–8 words) that clearly fit exactly one bin; avoid commas/semicolons and multi-clause wording.
- Difficulty: Introductory (Bloom: Understand). Avoid trick wording.
- Provide one short hint per term (non-revealing).
- Allowed label pool (choose from, if relevant): {allowed_hint}
{("- Enforce: each BIN label must literally appear in the excerpts; reject abstract bins." if strict else "")}

Return STRICT JSON (no markdown):
{{
  "title": "string",
  "bins": ["string", "string", "string"],
  "terms": ["string", ...],                 // exactly 6
  "mapping": {{"TERM":"BIN"}},              // each term -> one of the bins
  "hints": {{"TERM":"one short hint"}}
}}"""
        raw = _chat(client, sys_prompt, user_prompt, max_tokens=900, temperature=0.33, seed=seed_val)
        return raw

    for attempt in range(2):
        raw = _ask(seed_val=(new_seed()%1_000_000), strict=(attempt==1))
        try:
            data = json.loads(_extract_json_block(raw))
        except Exception:
            data = {}
        bins  = data.get("bins") or []
        terms = data.get("terms") or []
        mapping = data.get("mapping") or {}
        hints = data.get("hints") or {}

        if not (isinstance(bins, list) and len(bins) == 3): continue
        if not (isinstance(terms, list) and len(terms) == 6): continue
        if not (isinstance(mapping, dict) and all(t in mapping for t in terms)): continue
        if any(_is_vague_label(lbl) or not _label_in_scope(lbl, scope) for lbl in bins):
            continue
        counts = {b:0 for b in bins}
        ok_terms = True
        for t in terms:
            b = mapping.get(t)
            if b not in counts:
                ok_terms = False; break
            counts[b] += 1
        if not ok_terms or min(counts.values()) < 2:
            continue

        title = data.get("title") or "Classify based on slide concepts"
        instr = "Drag each statement to the correct slide-based category ."
        labels = bins
        draggables = terms
        answer = {t: mapping.get(t) for t in terms}
        hint_map = {t: (hints.get(t) or "Re-read the relevant slide line.") for t in terms}
        return title, instr, labels, draggables, answer, hint_map

    return None

def gen_fitb_from_scope(scope: str, prompt: str):
    client = _openai_client()
    if client is None or not scope.strip():
        return None

    def _ask(seed_val: int):
        sys_prompt = "You create intro-level fill-in-the-blank items grounded ONLY in the provided slide excerpts. Never reveal answers."
        user_prompt = f"""From the slide excerpts, create 4 FITB items appropriate for an introductory biology course.

Slide excerpts:
\"\"\"
{scope}
\"\"\"

Student prompt: "{prompt}"

Design constraints (MUST follow ALL):
- Exactly 4 items; each is 10–16 words with one blank (use 5+ underscores: _____).
- Answers must be present in the excerpts and be a single word (or hyphenation).
- Avoid commas/semicolons in stems; keep language simple and concrete.
- Provide one short, non-revealing hint per item.

Return STRICT JSON array (no markdown). Each item:
{{"stem":"A concise sentence with _____ one blank","answers":["answer"],"hint":"short hint"}}"""
        raw = _chat(client, sys_prompt, user_prompt, max_tokens=700, temperature=0.28, seed=seed_val)
        return raw

    best = None
    for attempt in range(2):
        raw = _ask(seed_val=(new_seed()%1_000_000))
        try:
            items = json.loads(_extract_json_block(raw))
        except Exception:
            items = []

        out = []
        for it in (items if isinstance(items, list) else []):
            stem = (it.get("stem","") or "").strip()
            ans  = [a for a in (it.get("answers",[]) or []) if isinstance(a,str) and a.strip()]
            hint = (it.get("hint","") or "Use the exact term from the slides.").strip()
            if not stem or "____" not in stem:
                continue
            if not ans or any(len(a.split())>1 and "-" not in a for a in ans):
                continue
            wc = len(re.findall(r"[A-Za-z0-9']+", stem))
            if not (10 <= wc <= 16):
                continue
            if ("," in stem or ";" in stem):
                continue
            out.append({"stem": stem, "answers": ans[:2], "hint": hint})

        if len(out) == 4 and all(_intro_level_ok(i["stem"], i["answers"]) for i in out):
            best = out
            break

    
    return best

def ensure_four_fitb(fitb_items, topic: str):
    items = list(fitb_items or [])
    if len(items) >= 4:
        return items[:4]
    need = 4 - len(items)
    add_pool = build_fitb(topic, random.Random(new_seed()))
    # Avoid duplicates by stem text
    have = { (i.get("stem","")).strip().lower() for i in items }
    for it in add_pool:
        if (it.get("stem","")).strip().lower() not in have:
            items.append(it)
            have.add((it.get("stem","")).strip().lower())
            if len(items) == 4: break
    return items[:4]


# ---------- Fallbacks ----------
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
    instr = "Drag each **item** to its **category**."
    hints = {t: "Focus on the defining feature mentioned in class." for t in terms}
    return title, instr, labels, terms, mapping, hints


def build_fitb(topic: str, rng: random.Random) -> List[Dict[str,Any]]:
    items = []
    t = (topic or "").lower()
    if "glycolysis" in t:
        items.append({"stem":"The end product of glycolysis is _____ .","answers":["pyruvate"],"hint":"Three-carbon product."})
        items.append({"stem":"Glycolysis occurs in the _____ .","answers":["cytosol","cytoplasm"],"hint":"Not an organelle lumen."})
        items.append({"stem":"The committed step of glycolysis is catalyzed by _____ .","answers":["pfk-1","pfk1","phosphofructokinase"],"hint":"Allosterically regulated enzyme."})
        items.append({"stem":"ATP is generated by substrate-level phosphorylation at _____ kinase.","answers":["pyruvate","phosphoglycerate"],"hint":"Final steps."})
    elif "transcription" in t:
        items.append({"stem":"In eukaryotes, mRNA is synthesized by _____ polymerase II.","answers":["rna"],"hint":"Pol II."})
        items.append({"stem":"Transcription starts at the _____ region of DNA.","answers":["promoter"],"hint":"TATA box may be here."})
        items.append({"stem":"The introns are removed from pre-mRNA by the _____ .","answers":["spliceosome"],"hint":"snRNP complex."})
        items.append({"stem":"A 7-methylguanosine cap is added to the _____ end.","answers":["5'","5-prime","5prime","5’"],"hint":"The end that emerges first."})
    elif "translation" in t:
        items.append({"stem":"Protein synthesis occurs on the _____ .","answers":["ribosome"],"hint":"Large and small subunits."})
        items.append({"stem":"Translation begins at the start codon, typically _____ .","answers":["aug"],"hint":"Also codes for methionine."})
        items.append({"stem":"The anticodon resides on the _____ molecule.","answers":["trna","t-rna"],"hint":"Adapter between codon and amino acid."})
        items.append({"stem":"Peptide bond formation is catalyzed in the _____ center.","answers":["peptidyl-transferase","peptidyl transferase","peptidyltransferase"],"hint":"Activity of the large subunit rRNA."})
    elif "dna repair" in t:
        items.append({"stem":"Bulky adducts, like thymine dimers, are removed by _____ .","answers":["nucleotide-excision-repair","ner","nucleotide excision repair"],"hint":"Excises an oligonucleotide."})
        items.append({"stem":"Cytosine deamination is corrected by _____ excision repair.","answers":["base"],"hint":"Removes damaged base then sugar-phosphate."})
        items.append({"stem":"Replication errors are fixed by _____ repair.","answers":["mismatch"],"hint":"MutS recognizes distortion."})
        items.append({"stem":"DNA ligase seals a remaining _____ in the backbone.","answers":["nick"],"hint":"Between adjacent nucleotides."})
    else:
        # Generic but meaningful fallbacks
        items.append({"stem":"DNA is composed of repeating _____ units called nucleotides.","answers":["monomer"],"hint":"Basic building block."})
        items.append({"stem":"A gene is transcribed into _____ before translation.","answers":["mrna","messenger-rna","messenger rna"],"hint":"RNA type that carries coding sequence."})
        items.append({"stem":"Proteins are synthesized from amino acids during _____ .","answers":["translation"],"hint":"Occurs on ribosomes."})
        items.append({"stem":"The cell membrane is a lipid _____ with embedded proteins.","answers":["bilayer"],"hint":"Two leaflets."})
    rng.shuffle(items)
    return items[:4]


# ---------- Exam (optional) ----------
@st.cache_data(show_spinner=False)
def load_exam_corpus() -> List[str]:
    return load_corpus_from_github(GITHUB_USER, GITHUB_REPO, EXAMS_DIR_GH, GITHUB_BRANCH)

def extract_exam_style(exam_corpus: List[str]) -> Dict[str,Any] | None:
    client = _openai_client()
    if client is None or not exam_corpus:
        return None
    sample = "\n\n".join(exam_corpus)[:12000]
    system = "You distill exam style. Return strict JSON only."
    user = f"""From the prior exams below, extract a compact style profile.

PRIOR EXAMS (snippets):
\"\"\"
{sample}
\"\"\"

Return STRICT JSON:
{{
  "preferred_types": ["mcq","short_answer"],
  "mcq_options": 4,
  "tone": "succinct|formal|clinical|conversational",
  "length": "short|medium|long",
  "constraints": ["single-best-answer"],
  "rationale_required": true
}}"""
    try:
        prof = json.loads(_extract_json_block(_chat(client, system, user, max_tokens=400, temperature=0.0)))
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

    schema_block = (
        "Schema:\n"
        + "{\n"
        + '  "type": "mcq" | "short_answer",\n'
        + '  "stem": "string (<= 70 words; tone per style)",\n'
        + f'  "options": ["A","B","C","D"],      // MCQ only, {mcq_n} options\n'
        + '  "answer": "B",                     // letter for MCQ; text for SA\n'
        + '  "rationale": "why correct; <= 60 words",\n'
        + '  "bloom": "Remember|Understand|Apply|Analyze|Evaluate|Create",\n'
        + '  "difficulty": 1,                   // 1-5\n'
        + '  "slide_refs": [int]\n'
        + "}\n"
    )

    user = (
        f"SLIDE EXCERPTS (authoritative; cite page numbers if present):\n\"\"\"\n{scope}\n\"\"\"\n\n"
        f'STUDENT PROMPT: "{user_prompt}"\n\n'
        f"STYLE PROFILE:\n{style_json}\n\n"
        "Create ONE exam-style question faithful to the slides.\n\n"
        + schema_block +
        "\n- Distractors must be plausible and present in scope.\n"
        "- Cite 1–3 slide pages in \"slide_refs\".\n"
        f"- Prefer type: {'MCQ' if want_mcq else 'short_answer'} if it fits.\n"
    )

    try:
        q = json.loads(_extract_json_block(_chat(client, system, user, max_tokens=700, temperature=0.1)))
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
# ---------- Render Exam (optional) ----------
def render_exam():
    if st.session_state.get("exam_q"):
        st.markdown("---")
        st.markdown("## Exam-style Question")
        st.caption(f"Source: {st.session_state.get('exam_source','Unavailable')}")
        q = st.session_state.exam_q
        st.write(q.get("stem","(no stem)"))
        if q.get("type") == "mcq":
            opts = q.get("options", [])
            choice = st.radio("Choose one:", opts, index=0 if opts else None, key="exam_choice")
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


# ---------- Main UI flow ----------
prompt_val = st.text_input(
    "Enter a topic for review and press generate:",
    value=st.session_state.get("prompt_value_input",""),
    placeholder="e.g., organelle function, glycolysis regulation, DNA repair…",
    key="prompt_value_input"
)
go = st.button("Generate")

if go:
    if "corpus" not in st.session_state:
        st.session_state.corpus = load_corpus_from_github(GITHUB_USER, GITHUB_REPO, SLIDES_DIR_GH, GITHUB_BRANCH)
    if "exam_corpus" not in st.session_state:
        st.session_state.exam_corpus = load_corpus_from_github(GITHUB_USER, GITHUB_REPO, EXAMS_DIR_GH, GITHUB_BRANCH)

    topic = classify_topic(prompt_val) or "this topic"
    st.session_state.topic = topic
    scope = build_scope(st.session_state.corpus or [], prompt_val, limit_chars=6000)

    dnd = gen_dnd_from_scope(scope, prompt_val)
    if dnd is None:
        title, instr, labels, terms, answer, hint_map = build_dnd_activity(topic)
    else:
        title, instr, labels, terms, answer, hint_map = dnd
    st.session_state.dnd_title = title
    st.session_state.dnd_instr = instr
    st.session_state.drag_labels = labels
    st.session_state.drag_bank   = (terms[:4] if isinstance(terms, list) else terms)
    st.session_state.drag_answer = answer
    st.session_state.dnd_hints   = hint_map
    st.session_state.scope      = scope
    st.session_state.prompt_used = prompt_val

    fitb_items = gen_fitb_from_scope(scope, prompt_val)
    if fitb_items is None:
        rng = random.Random(new_seed())
        fitb_items = build_fitb(topic, random.Random(new_seed()))
    st.session_state.fitb = ensure_four_fitb(fitb_items, st.session_state.get("topic",""))

    style = extract_exam_style(st.session_state.exam_corpus or [])
    st.session_state.exam_q = gen_exam_question(scope, style or {}, prompt_val) if style else None

# Render activities if available

if all(k in st.session_state for k in ["drag_labels","drag_bank","drag_answer","dnd_instr"]):
    st.markdown("## Activity 1: Drag and Drop")
    
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

if "fitb" in st.session_state:
    st.markdown("---")
    st.markdown("## Activity 2: Fill in the Blank")
    
    topic_name = st.session_state.get("topic","this topic")
    st.markdown(f"Use your knowledge of **{topic_name}** to answer the following.")

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+","", (s or "").lower())

    for idx, item in enumerate(st.session_state.fitb):
        u = st.text_input(item["stem"], key=f"fitb_{idx}")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Hint", key=f"hint_{idx}"):
                st.info(item.get("hint","Use the exact term from the slides."))
        with col2:
            if st.button("Check", key=f"check_{idx}"):
                ans_list = item.get("answers", [])
                ok, close = check_fitb_answer(u, ans_list)
                scope_text = st.session_state.get("scope","")
                feedback = llm_feedback_for_fitb(scope_text, item.get("stem",""), ans_list, u)
                if ok:
                    st.success("That’s right! 🎉")
                elif close:
                    st.warning("Almost there.")
                    st.info(feedback)
                else:
                    st.warning("Not quite.")
                    st.info(feedback)
        with col3:
            if st.button("Reveal", key=f"rev_{idx}"):
                ans = item.get("answers", [])
                st.info(", ".join(ans) if ans else "(no stored answer)")

# Optional exam renderer if you keep exam_q
render_exam()
