# app.py — Cell Bio Tutor (Slides-only, LLM-authored, H5P via viewer ?src=)
# If data-URL is too large, uploads the .h5p to GitHub repo and passes a short URL.

import os, io, json, base64, zipfile, pathlib, re, uuid, hashlib, html, urllib.parse
from typing import List, Dict, Optional, Tuple
import streamlit as st
import numpy as np
from pypdf import PdfReader
import requests
import time

st.set_page_config(page_title="Cell Bio Tutor — Slides Only", layout="centered")
st.title("🧬 Cell Bio Tutor — Slides-Only (LLM + H5P)")

ROOT = pathlib.Path(__file__).parent
SLIDES_DIR = ROOT / "slides"
TEMPLATES_DIR = ROOT / "templates"
FIB_MASTER = TEMPLATES_DIR / "rtk_fill_in_blanks_FIXED_blocks.h5p"
DND_MASTER = TEMPLATES_DIR / "cellular_respiration_aligned_course_style.h5p"

VIEWER_QS = "https://mwellis825.github.io/cell-bio-gpt-tutor/viewer.html?src="
DATA_URL_LIMIT = 180_000  # conservative URL-encoded length limit

# --- OpenAI (lazy) ---
EMBED_MODEL = "text-embedding-3-small"

def _get_api_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key and hasattr(st, "secrets"):
        key = st.secrets.get("OPENAI_API_KEY")
    return key

def _get_openai_client():
    key = _get_api_key()
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"Could not initialize OpenAI client: {e}")
        return None

if "openai_client" not in st.session_state:
    st.session_state.openai_client = None

def client():
    if st.session_state.openai_client is None:
        st.session_state.openai_client = _get_openai_client()
    return st.session_state.openai_client

# --- GitHub upload helper ---
def gh_cfg():
    tok = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("GITHUB_REPO", "")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    updir = st.secrets.get("GITHUB_UPLOAD_DIR", "docs/generated")
    return tok, repo, branch, updir

def upload_h5p_to_github(bytes_data: bytes, filename: str) -> Optional[str]:
    tok, repo, branch, updir = gh_cfg()
    if not tok or not repo:
        return None
    path = f"{updir.rstrip('/')}/{filename}"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    content_b64 = base64.b64encode(bytes_data).decode("utf-8")
    payload = {
        "message": f"Add generated H5P {filename}",
        "content": content_b64,
        "branch": branch,
    }
    headers = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    r = requests.put(url, headers=headers, json=payload, timeout=30)
    if r.status_code in (200, 201):
        # raw URL (works for public repos)
        user_repo = repo
        raw = f"https://raw.githubusercontent.com/{user_repo}/{branch}/{path}"
        # tiny wait for GitHub to serve fresh content
        for _ in range(5):
            try:
                rr = requests.head(raw, timeout=5)
                if rr.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.6)
        return raw
    else:
        st.warning(f"GitHub upload failed: {r.status_code} {r.text}")
        return None

# --- Session state ---
def init_state():
    defaults = dict(
        topic="",
        n_items=3,
        gen_mcq=True,
        gen_dnd=True,
        index_ready=False,
        index_chunks=[],
        index_embeds=None,
        run_id=None,
        cache={},  # run_id -> {"fib_bytes":..., "dnd_bytes":..., "mcqs":[...], "fib_url":..., "dnd_url":...}
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# --- Slides ingestion & retrieval ---
def extract_pages(pdf_path: pathlib.Path) -> List[Dict]:
    out = []
    try:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            txt = re.sub(r"\s+", " ", txt).strip()
            if txt:
                out.append({"file": pdf_path.name, "page": i, "text": txt})
    except Exception as e:
        st.warning(f"Could not read {pdf_path.name}: {e}")
    return out

def ensure_index() -> bool:
    if st.session_state.index_ready:
        return True
    if not SLIDES_DIR.exists():
        st.error("slides/ folder not found. Add your course PDFs there.")
        return False
    chunks = []
    for p in sorted(SLIDES_DIR.glob("*.pdf")):
        chunks.extend(extract_pages(p))
    if not chunks:
        st.error("No extractable text found in slides/.")
        return False
    st.session_state.index_chunks = chunks

    cli = client()
    if cli:
        try:
            texts = [c["text"] for c in chunks]
            embeds = []
            BATCH = 64
            for i in range(0, len(texts), BATCH):
                batch = texts[i:i+BATCH]
                resp = cli.embeddings.create(model=EMBED_MODEL, input=batch)
                embeds.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
            st.session_state.index_embeds = np.vstack(embeds).astype(np.float32)
        except Exception as e:
            st.warning(f"Embeddings failed; retrieval will be keyword-only. ({e})")
            st.session_state.index_embeds = None
    else:
        st.info("No OpenAI key: retrieval will be keyword-only.")

    st.session_state.index_ready = True
    return True

def search_chunks(query: str, k: int = 8) -> List[Dict]:
    if not ensure_index():
        return []
    chunks = st.session_state.index_chunks
    M = st.session_state.index_embeds

    if M is None:
        q = query.lower()
        toks = [t for t in re.findall(r"\w+", q) if t]
        scored = []
        for ch in chunks:
            t = ch["text"].lower()
            score = sum(t.count(tok) for tok in toks)
            if score > 0:
                scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[:k]] or chunks[:k]

    try:
        qe = client().embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    except Exception:
        st.warning("Query embedding failed; falling back to keyword retrieval.")
        st.session_state.index_embeds = None
        return search_chunks(query, k)

    qe = np.array(qe, dtype=np.float32)
    sims = (M @ qe) / (np.linalg.norm(M, axis=1) * (np.linalg.norm(qe) + 1e-8) + 1e-8)
    top_idx = np.argsort(-sims)[:k]
    return [chunks[i] for i in top_idx]

def build_context(query: str, max_chars: int = 6000) -> Tuple[str, List[Tuple[str,int]]]:
    hits = search_chunks(query, k=10)
    pieces, sources, used = [], [], 0
    for h in hits:
        tag = f"[{h['file']} p.{h['page']}]"
        snippet = f"{tag} {h['text']}"
        if used + len(snippet) > max_chars:
            break
        pieces.append(snippet)
        sources.append((h['file'], h['page']))
        used += len(snippet)
    return "\n\n".join(pieces), sources

def load_openstax_fallback() -> str:
    return ("Cellular respiration spans glycolysis, pyruvate oxidation, the citric acid cycle, "
            "and oxidative phosphorylation (electron transport + chemiosmosis). The ETC across the inner "
            "mitochondrial membrane generates a proton gradient used by ATP synthase. Regulation is influenced "
            "by allosteric effectors, energy charge (ATP/ADP/AMP), and membrane integrity.")

def build_context_with_fallback(query: str, min_chars: int = 1200) -> Tuple[str, List[Tuple[str,int]]]:
    ctx, src = build_context(query, max_chars=6000)
    if len(ctx) < min_chars:
        fb = load_openstax_fallback()
        ctx = (ctx + "\n\n[OpenStax fallback]\n" + fb) if ctx else fb
    return ctx, src

# --- LLM generation (slides + fallback) ---
def gen_fib_lines_from_context(topic: str, difficulty: str, n_items: int) -> List[str]:
    if client() is None:
        raise RuntimeError("OpenAI client not initialized.")
    context, _ = build_context_with_fallback(topic)
    sys = ("You are a Cell Biology tutor bound to the provided context. "
           "Only use information present in the context/fallback; do NOT invent facts. "
           "Prefer causal, predictive reasoning about perturbations.")
    user = f"""
Topic: {topic}
Difficulty: {difficulty}

Context:
{context}

Task:
Produce {n_items} concise, critical-thinking Fill-in-the-Blanks items using H5P syntax.
Rules:
- Exactly ONE sentence and ONE blank per item.
- The blank should be *increase/increased* or *decrease/decreased* (predictive outcomes).
- Use asterisks for acceptable variants, e.g., *increase/increased*.
- Keep load low but conceptual; no definition recall.
- Vary perturbed components and downstream targets.
- If context is insufficient, write items that say 'Not in slides *increase/decrease*.'

Return as a numbered list of plain lines.
"""
    resp = client().chat.completions.create(
        model="gpt-4o", temperature=0.2,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    raw = resp.choices[0].message.content or ""
    out = []
    for line in raw.splitlines():
        s = line.strip().lstrip("0123456789). ").strip()
        if not s:
            continue
        predictive = ("*increase" in s.lower()) or ("*decrease" in s.lower())
        if "*" in s and predictive and not re.search(r"\bdefined as\b", s.lower()):
            out.append(s)
    if not out:
        out = ["If proton leak increases, ATP synthase output will *decrease/decreased*."]
    return out[:n_items]

def gen_mcqs_from_context(topic: str, difficulty: str, n_items: int = 2) -> List[Dict]:
    if client() is None:
        return []
    context, _ = build_context_with_fallback(topic)
    sys = "Create multiple-choice questions ONLY from the context. Do not invent facts."
    user = f"""
Context:
{context}

Task:
Write {n_items} MCQs as JSON array: 
[{{"question": "...", "options": ["A","B","C","D"], "answer_index": 0}}]
Keep options concise; use only slide/fallback facts. If insufficient, return [].
"""
    resp = client().chat.completions.create(
        model="gpt-4o", temperature=0.2,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    raw = resp.choices[0].message.content or "[]"
    try:
        j = json.loads(raw.strip(" \n`"))
        return j if isinstance(j, list) else []
    except Exception:
        m = re.search(r"\[.*\]", raw, flags=re.S)
        return json.loads(m.group(0)) if m else []

def gen_dnd_pairs_from_context(topic: str, n_pairs: int = 5) -> List[Tuple[str,str]]:
    if client() is None:
        return []
    context, _ = build_context_with_fallback(topic)
    sys = "Create Drag-and-Drop pairs ONLY from context; map concise term to matching description."
    user = f"""
Context:
{context}

Task:
Provide {n_pairs} pairs as JSON array of objects:
- drag: short term (<= 8 words)
- drop: short label (<= 10 words)
Distinct pairs; concise; use only context/fallback.
"""
    pairs: List[Tuple[str,str]] = []
    try:
        resp = client().chat.completions.create(
            model="gpt-4o", temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        arr = json.loads((resp.choices[0].message.content or "[]").strip(" \n`"))
        for obj in arr:
            if isinstance(obj, dict) and "drag" in obj and "drop" in obj:
                pairs.append((obj["drag"], obj["drop"]))
    except Exception:
        pairs = []
    if pairs:
        return pairs[:n_pairs]
    # fallback
    t = topic.lower()
    if "electron transport" in t or "etc" in t or "oxidative" in t:
        return [
            ("Complex I", "NADH → e⁻, pumps H⁺"),
            ("Complex II", "FADH₂ → e⁻ (no pumping)"),
            ("Complex III", "Q → Cyt c, pumps H⁺"),
            ("Complex IV", "O₂ → H₂O, pumps H⁺"),
            ("ATP synthase", "H⁺ gradient → ATP"),
        ][:n_pairs]
    return [
        ("Nucleus", "DNA storage"),
        ("ER", "Protein folding"),
        ("Golgi", "Modification/Sorting"),
        ("Lysosome", "Acid hydrolases"),
        ("Mitochondria", "ATP production"),
    ][:n_pairs]

# --- H5P builders (from master .h5p) ---
def build_fib_from_master(instructions: str, lines: List[str]) -> Optional[bytes]:
    if not FIB_MASTER.exists():
        st.error("FIB master missing in templates/.")
        return None
    try:
        with zipfile.ZipFile(FIB_MASTER, "r") as zin:
            files = {n: zin.read(n) for n in zin.namelist()}
        content = json.loads(files["content/content.json"].decode("utf-8"))
        content["text"] = instructions
        content["questions"] = [f"<p>{ln}</p>" for ln in lines]
        beh = content.get("behaviour", {})
        beh["caseSensitive"] = False
        beh["ignorePunctuation"] = True
        beh["acceptSpellingErrors"] = True
        content["behaviour"] = beh
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for n, d in files.items():
                if n == "content/content.json":
                    z.writestr(n, json.dumps(content, ensure_ascii=False, indent=2))
                else:
                    z.writestr(n, d)
        return out.getvalue()
    except Exception as e:
        st.error(f"Couldn’t build FIB H5P: {e}")
        return None

def build_dnd_from_master(pairs: List[Tuple[str,str]]) -> Optional[bytes]:
    if not DND_MASTER.exists():
        st.error("DnD master missing in templates/.")
        return None
    # compact alignment
    elements, zones = [], []
    y_dr, y_zn = 6.5, 13.9
    for i, (drag, drop) in enumerate(pairs[:5]):
        elements.append({"text": drag, "x": 1.0 if i==0 else 0.8 + 0.2*(i%2),
                         "y": y_dr + 18.0*i, "w": 9.0, "h": 3.2})
        zones.append({"label": drop, "x": 43.0 - 2.0*(i%3),
                      "y": y_zn + 16.7*i, "w": 18.0, "h": 1.9, "correct_idx": i})
    try:
        with zipfile.ZipFile(DND_MASTER, "r") as zin:
            files = {n: zin.read(n) for n in zin.namelist()}
        content = json.loads(files["content/content.json"].decode("utf-8"))
        content["question"]["task"]["elements"] = [{
            "x": e["x"], "y": e["y"], "width": e["w"], "height": e["h"],
            "dropZones": [str(i) for i in range(len(zones))],
            "backgroundOpacity": 0,
            "type": {"library": "H5P.AdvancedText 1.1",
                     "params": {"text": f"<span>{e['text']}</span>"}}
        } for e in elements]
        content["question"]["task"]["dropZones"] = [{
            "x": z["x"], "y": z["y"], "width": z["w"], "height": z["h"],
            "label": f"<div style='text-align:center'>{z['label']}</div>",
            "showLabel": True, "backgroundOpacity": 0,
            "autoAlign": True, "single": False,
            "correctElements": [str(z['correct_idx'])]
        } for z in zones]
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for n, d in files.items():
                if n == "content/content.json":
                    zf.writestr(n, json.dumps(content, ensure_ascii=False, indent=2))
                else:
                    zf.writestr(n, d)
        return out.getvalue()
    except Exception as e:
        st.error(f"Couldn’t build DnD H5P: {e}")
        return None

# --- Rendering helpers ---
def render_inline_if_small(h5p_bytes: bytes, height: int = 620) -> bool:
    b64 = base64.b64encode(h5p_bytes).decode("utf-8")
    data_url = "data:application/zip;base64," + b64
    encoded = urllib.parse.quote(data_url, safe="")
    if len(encoded) > DATA_URL_LIMIT:
        return False
    st.components.v1.iframe(src=VIEWER_QS + encoded, height=height)
    return True

def render_via_github(h5p_bytes: bytes, fname: str, height: int = 620) -> Optional[str]:
    url = upload_h5p_to_github(h5p_bytes, fname)
    if not url:
        return None
    st.components.v1.iframe(src=VIEWER_QS + urllib.parse.quote(url, safe=""), height=height)
    return url

# --- Sidebar / Controls ---
with st.sidebar:
    st.header("Authoring")
    st.session_state.n_items = st.slider("Items per FIB", 2, 6, st.session_state.n_items, 1)
    st.session_state.gen_mcq = st.toggle("Also generate MCQs", value=st.session_state.gen_mcq)
    st.session_state.gen_dnd = st.toggle("Also generate Drag-and-Drop", value=st.session_state.gen_dnd)
    st.caption("Generations use ONLY your slides; if thin, adds a tiny OpenStax fallback snippet.")

topic = st.text_input("What do you want help with? (e.g., “electron transport chain”, “RTK”):", value=st.session_state.topic)
generate_clicked = st.button("Generate Activity")

# --- Generate and render ---
if generate_clicked:
    st.session_state.run_id = uuid.uuid4().hex[:8]
    run_id = st.session_state.run_id
    st.session_state.topic = topic
    st.session_state.cache[run_id] = {"fib_bytes": None, "dnd_bytes": None, "mcqs": [], "fib_url": None, "dnd_url": None}

    # FIB
    try:
        lines = gen_fib_lines_from_context(topic, difficulty="medium", n_items=st.session_state.n_items)
        instructions = f"Based ONLY on your course slides: predict the downstream effect for **{topic}** (answer with *increase/decrease*)."
        fib_bytes = build_fib_from_master(instructions, lines)
        st.session_state.cache[run_id]["fib_bytes"] = fib_bytes
    except Exception as e:
        st.error(f"FIB generation failed: {e}")

    # MCQs
    if st.session_state.gen_mcq:
        try:
            st.session_state.cache[run_id]["mcqs"] = gen_mcqs_from_context(topic, difficulty="medium", n_items=2)
        except Exception as e:
            st.warning(f"MCQ generation failed: {e}")

    # DnD
    if st.session_state.gen_dnd:
        try:
            pairs = gen_dnd_pairs_from_context(topic, n_pairs=5)
            if pairs:
                dnd_bytes = build_dnd_from_master(pairs)
                st.session_state.cache[run_id]["dnd_bytes"] = dnd_bytes
        except Exception as e:
            st.warning(f"DnD generation failed: {e}")

# --- Show results (latest run) ---
rid = st.session_state.run_id
if rid and rid in st.session_state.cache:
    data = st.session_state.cache[rid]

    if data.get("fib_bytes"):
        st.success("Fill-in-the-Blanks (slides-only)")
        if not render_inline_if_small(data["fib_bytes"]):
            url = render_via_github(data["fib_bytes"], f"fib_{rid}.h5p")
            if not url:
                st.info("Inline preview skipped (payload too large) and upload failed. Use the download button.")
        st.download_button(
            "⬇️ Download FIB (.h5p)",
            data=data["fib_bytes"],
            file_name=f"fib_{rid}.h5p",
            mime="application/zip",
            key=f"dl-fib-{rid}-{hashlib.md5(data['fib_bytes']).hexdigest()[:8]}",
        )

    mcqs = data.get("mcqs", [])
    if mcqs:
        st.subheader("Quick Check (slides-only)")
        for idx, q in enumerate(mcqs):
            key = f"{rid}-mcq-{idx}"
            sel = st.radio(q["question"], q["options"], index=None, key=key)
            if sel is not None:
                correct = (q["options"].index(sel) == q["answer_index"])
                st.markdown("✅ Correct!" if correct else "❌ Not quite — recheck the slides.")
                st.divider()

    if data.get("dnd_bytes"):
        st.success("Drag-and-Drop (slides-only)")
        if not render_inline_if_small(data["dnd_bytes"]):
            url = render_via_github(data["dnd_bytes"], f"dnd_{rid}.h5p")
            if not url:
                st.info("Inline preview skipped (payload too large) and upload failed. Use the download button.")
        st.download_button(
            "⬇️ Download DnD (.h5p)",
            data=data["dnd_bytes"],
            file_name=f"dnd_{rid}.h5p",
            mime="application/zip",
            key=f"dl-dnd-{rid}-{hashlib.md5(data['dnd_bytes']).hexdigest()[:8]}",
        )
