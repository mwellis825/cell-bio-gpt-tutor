import os, io, json, time, base64, zipfile, pathlib, re
from typing import List, Dict, Optional, Tuple
import streamlit as st
import numpy as np
from pypdf import PdfReader

# -------------------------------
# Page / Paths
# -------------------------------
st.set_page_config(page_title="Cell Bio Tutor (Slides-Only)", layout="centered")
st.title("🧬 Cell Bio Tutor — Slides Only")

ROOT = pathlib.Path(__file__).parent
SLIDES_DIR = ROOT / "slides"
FIB_MASTER = ROOT / "templates" / "rtk_fill_in_blanks_FIXED_blocks.h5p"
DND_MASTER = ROOT / "templates" / "cellular_respiration_aligned_course_style.h5p"

# -------------------------------
# OpenAI client (safe init)
# -------------------------------
def get_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        st.error(f"OpenAI SDK not available: {e}")
        return None
    key = os.getenv("OPENAI_API_KEY", None)
    if not key and hasattr(st, "secrets"):
        key = st.secrets.get("OPENAI_API_KEY")
    if not key:
        st.warning("Missing OPENAI_API_KEY (set as env var or Streamlit secret). Using no-AI mode.")
        return None
    try:
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"Could not initialize OpenAI client: {e}")
        return None

client = get_openai_client()

# -------------------------------
# Slide ingestion & embeddings (cache in session)
# -------------------------------
EMBED_MODEL = "text-embedding-3-small"  # cost-efficient, 1536-d

def extract_pages(pdf_path: pathlib.Path) -> List[Dict]:
    """Extract plain text by page with metadata."""
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

def ensure_index():
    """Load/compute embeddings for slides; store in session."""
    if "index_chunks" in st.session_state and "index_embeds" in st.session_state:
        return
    if not SLIDES_DIR.exists():
        st.error("slides/ folder not found. Add your course PDFs there.")
        st.stop()

    # 1) Extract
    chunks = []
    for p in sorted(SLIDES_DIR.glob("*.pdf")):
        chunks.extend(extract_pages(p))
    if not chunks:
        st.error("No extractable text found in slides/.")
        st.stop()

    # 2) Embed
    if not client:
        # Allow non-AI preview: we can still return raw text chunks later, but generation needs AI.
        st.warning("No OpenAI client: FIB/MCQ/DnD generation will be unavailable.")
    else:
        texts = [c["text"] for c in chunks]
        embeds = []
        # Batch to keep token usage manageable
        BATCH = 64
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            embeds.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
        st.session_state.index_embeds = np.vstack(embeds).astype(np.float32)
    st.session_state.index_chunks = chunks

def search_chunks(query: str, k: int = 8) -> List[Dict]:
    """Return top-k chunks most similar to the query."""
    ensure_index()
    chunks = st.session_state.index_chunks
    if not client or "index_embeds" not in st.session_state:
        # No embeddings → naive keyword filter
        scored = []
        q = query.lower()
        for ch in chunks:
            score = sum(q.count(tok) for tok in q.split() if tok in ch["text"].lower())
            scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s,c in scored[:k]]

    # Embed query and cosine-sim search
    qe = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    qe = np.array(qe, dtype=np.float32)
    M = st.session_state.index_embeds
    # cosine sim
    sims = (M @ qe) / (np.linalg.norm(M, axis=1) * (np.linalg.norm(qe) + 1e-8) + 1e-8)
    top_idx = np.argsort(-sims)[:k]
    return [st.session_state.index_chunks[i] for i in top_idx]

def build_context(query: str, max_chars: int = 6000) -> Tuple[str, List[Tuple[str,int]]]:
    """Join top chunks into a single context with source tags; return context and sources list."""
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

# -------------------------------
# Strict, slides-only LLM generation
# -------------------------------
def gen_fib_lines_from_context(topic: str, difficulty: str, n_items: int) -> List[str]:
    if not client:
        raise RuntimeError("OpenAI client not initialized.")
    context, sources = build_context(topic)
    sys = (
        "You are a Cell Biology tutor bound to the provided slide excerpts. "
        "You must ONLY use information present in the context. "
        "If the context is insufficient, reply with items that say 'Not in slides'."
    )
    user = f"""
Topic: {topic}
Difficulty: {difficulty}

Context (from the course slides, cite mentally but do NOT add citations in text):
{context}

Task:
Produce {n_items} concise, critical-thinking Fill-in-the-Blanks items using H5P syntax.
Rules:
- ONE sentence per item.
- Exactly ONE blank per sentence.
- Use asterisks for acceptable answers and variants, e.g., *increase/increased*.
- Keep cognitive load low. Focus on "predict increase/decrease" consequences.
- Do NOT invent facts beyond the context.

Return as a numbered list of plain lines."""
    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0.2,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    text = resp.choices[0].message.content or ""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        s = s.lstrip("0123456789). ").strip()
        # Ensure it contains asterisks
        if "*" in s:
            lines.append(s)
    if not lines:
        lines = ["Not in slides *increase/decrease*."]
    return lines[:n_items]

def gen_mcqs_from_context(topic: str, difficulty: str, n_items: int = 2):
    if not client:
        return []
    context, _ = build_context(topic)
    sys = ("Create multiple-choice questions ONLY from the context. "
           "Do not invent facts beyond the context.")
    user = f"""
Context:
{context}

Task:
Write {n_items} MCQs as JSON array with objects:
- question: string
- options: array of 4 concise options
- answer_index: 0-based index of the correct option

Keep wording short. Only use slide facts. If context is insufficient, return an empty JSON array.
"""
    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0.2,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    raw = resp.choices[0].message.content
    try:
        # Extract JSON block
        j = json.loads(raw.strip(" \n`"))
        return j if isinstance(j, list) else []
    except Exception:
        # Try to find JSON within markdown fences
        m = re.search(r"\[.*\]", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return []
        return []

def gen_dnd_pairs_from_context(topic: str, n_pairs: int = 5):
    """Return list of (draggable_text, zone_label)."""
    if not client:
        return []
    context, _ = build_context(topic)
    sys = ("Create DnD pairs strictly from the slides context; each pair maps a term to its matching description/location/output.")
    user = f"""
Context:
{context}

Task:
Provide {n_pairs} pairs for a Drag-and-Drop activity as JSON array of objects:
- drag: short text (term/step)
- drop: short label (matching description/location/output)
Constraints:
- Keep both texts short (<= 8 words).
- Use plain ASCII where possible.
- Only use information explicitly present in context.
"""
    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0.2,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    raw = resp.choices[0].message.content
    try:
        arr = json.loads(raw.strip(" \n`"))
        pairs = []
        for obj in arr:
            if isinstance(obj, dict) and "drag" in obj and "drop" in obj:
                pairs.append((obj["drag"], obj["drop"]))
        return pairs[:n_pairs]
    except Exception:
        return []

# -------------------------------
# H5P builders from your MASTER files
# -------------------------------
def build_fib_from_master(instructions: str, lines: List[str]) -> bytes:
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

def build_dnd_from_master(pairs: List[Tuple[str,str]]) -> bytes:
    # Geometry: your compact layout (left drags, right zones). Clamp to len(pairs)
    elements = []
    zones = []
    y_dr = 6.5
    y_zn = 13.9
    for i, (drag, drop) in enumerate(pairs[:5]):
        elements.append({"text": drag, "x": 1.0 if i==0 else 0.8 + 0.2*(i%2),
                         "y": y_dr + 18.0*i, "w": 9.0, "h": 3.2})
        zones.append({"label": drop, "x": 43.0 - 2.0*(i%3),
                      "y": y_zn + 16.7*i, "w": 18.0, "h": 1.9, "correct_idx": i})
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
        "correctElements": [str(z["correct_idx"])]
    } for z in zones]
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, d in files.items():
            if n == "content/content.json":
                zf.writestr(n, json.dumps(content, ensure_ascii=False, indent=2))
            else:
                zf.writestr(n, d)
    return out.getvalue()

# -------------------------------
# Inline H5P renderer (multi-CDN fallback) + download
# -------------------------------
def render_h5p_inline(h5p_bytes: bytes, height: int = 560):
    if not h5p_bytes:
        return
    b64 = base64.b64encode(h5p_bytes).decode("utf-8")
    html = f"""
    <div id="h5p-container" style="border:0; margin:0; padding:0;"></div>
    <script>
      (function() {{
        function loadScript(src, cb) {{
          var s = document.createElement('script'); s.src = src; s.onload = cb; s.onerror = cb;
          document.head.appendChild(s);
        }}
        function loadCss(href) {{
          var l = document.createElement('link'); l.rel = 'stylesheet'; l.href = href;
          document.head.appendChild(l);
        }}
        function boot(base) {{
          loadCss(base + '/styles/h5p.css');
          loadScript(base + '/frame.bundle.js', function(){{
            loadScript(base + '/main.bundle.js', function(){{
              try {{
                H5PStandalone.display('#h5p-container', {{
                  h5pContent: "data:application/zip;base64,{b64}",
                  frameJs: base + '/frame.bundle.js',
                  frameCss: base + '/styles/h5p.css'
                }});
              }} catch(e) {{
                document.getElementById('h5p-container').innerHTML = 
                  "<p style='color:#b00'>Couldn’t initialize H5P (CDN blocked?). Use download below.</p>";
              }}
            }});
          }});
        }}
        var jsDelivr = "https://cdn.jsdelivr.net/npm/h5p-standalone@1.3.0/dist";
        var unpkg    = "https://unpkg.com/h5p-standalone@1.3.0/dist";
        var testCss  = document.createElement('link'); testCss.rel='preload'; testCss.as='style'; testCss.href=jsDelivr+'/styles/h5p.css';
        testCss.onload = function(){{ boot(jsDelivr); }};
        testCss.onerror = function(){{ boot(unpkg); }};
        document.head.appendChild(testCss);
        setTimeout(function(){{
          if (!document.querySelector('link[href*="h5p.css"]')) {{
            document.getElementById('h5p-container').innerHTML =
              "<p style='color:#b00'>H5P scripts blocked by your network. Use the download button below.</p>";
          }}
        }}, 4000);
      }})();
    </script>
    """
    st.components.v1.html(html, height=height, scrolling=True)
    st.download_button("⬇️ Download activity (.h5p)", data=h5p_bytes,
                       file_name="activity_generated.h5p", mime="application/zip")

# -------------------------------
# UI
# -------------------------------
with st.sidebar:
    st.header("Authoring")
    n_items = st.slider("Items per FIB", 2, 6, 3, 1)
    gen_mcq = st.toggle("Also generate MCQs", value=True)
    gen_dnd = st.toggle("Also generate Drag-and-Drop", value=True)
    st.caption("All generations use ONLY your slides via retrieval.")

topic = st.text_input("What do you want help with? (e.g., “electron transport chain”, “RTK”):")
if st.button("Generate Activity"):
    if not client:
        st.error("OpenAI client not available. Set OPENAI_API_KEY.")
        st.stop()
    if not (FIB_MASTER.exists() and DND_MASTER.exists()):
        st.error("Master H5P templates missing in templates/.")
        st.stop()

    # 1) FIB from slides
    try:
        fib_lines = gen_fib_lines_from_context(topic, difficulty="medium", n_items=n_items)
        fib_bytes = build_fib_from_master("Predict whether the following would increase or decrease activity:", fib_lines)
        st.success("Fill-in-the-Blanks (slides-only)")
        render_h5p_inline(fib_bytes, height=560)
        st.subheader("Preview of items")
        for i, ln in enumerate(fib_lines, 1):
            st.write(f"{i}) {ln}")
    except Exception as e:
        st.error(f"Could not build FIB: {e}")

    # 2) MCQs from slides (adaptive coming from your prior loop if you want)
    if gen_mcq:
        try:
            mcqs = gen_mcqs_from_context(topic, difficulty="medium", n_items=2)
            if mcqs:
                st.subheader("Quick Check (slides-only)")
                for idx, q in enumerate(mcqs):
                    key = f"mcq_{time.time()}_{idx}"
                    sel = st.radio(q["question"], q["options"], index=None, key=key)
                    if sel is not None:
                        correct = (q["options"].index(sel) == q["answer_index"])
                        st.markdown("✅ Correct!" if correct else "❌ Not quite — check the slides.")
            else:
                st.info("Not enough slide context to make MCQs.")
        except Exception as e:
            st.warning(f"MCQ generation failed: {e}")

    # 3) Drag-and-Drop from slides
    if gen_dnd:
        try:
            pairs = gen_dnd_pairs_from_context(topic, n_pairs=5)
            if pairs:
                dnd_bytes = build_dnd_from_master(pairs)
                st.success("Drag-and-Drop (slides-only)")
                render_h5p_inline(dnd_bytes, height=520)
                with st.expander("DnD pairs used"):
                    for a,b in pairs:
                        st.write(f"- **{a}** → {b}")
            else:
                st.info("Not enough slide context to build Drag-and-Drop.")
        except Exception as e:
            st.warning(f"DnD generation failed: {e}")
