# app.py — Cell Bio Tutor (Slides-Only, LLM-authored, H5P local per-iframe loader, no CDN)

import os, io, json, base64, zipfile, pathlib, re, uuid, hashlib
from typing import List, Dict, Optional, Tuple
import streamlit as st
import numpy as np
from pypdf import PdfReader

# --------------------------------
# Page / Paths
# --------------------------------
st.set_page_config(page_title="Cell Bio Tutor — Slides Only", layout="centered")
st.title("🧬 Cell Bio Tutor — Slides-Only (LLM + H5P)")

ROOT = pathlib.Path(__file__).parent
SLIDES_DIR = ROOT / "slides"
TEMPLATES_DIR = ROOT / "templates"
FIB_MASTER = TEMPLATES_DIR / "rtk_fill_in_blanks_FIXED_blocks.h5p"
DND_MASTER = TEMPLATES_DIR / "cellular_respiration_aligned_course_style.h5p"
VENDOR_H5P_DIR = ROOT / "vendor" / "h5p"  # main.bundle.js, frame.bundle.js, styles/h5p.css

# --------------------------------
# OpenAI (lazy init; no proxies)
# --------------------------------
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

# --------------------------------
# Session state
# --------------------------------
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
        cache={},  # generated artifacts by run_id
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# --------------------------------
# Slides ingestion & retrieval
# --------------------------------
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
        st.info("No OpenAI client: retrieval will be keyword-only.")

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

# --------------------------------
# LLM generation (slides + fallback)
# --------------------------------
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

    # Heuristic fallback
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

# --------------------------------
# H5P builders (from master .h5p)
# --------------------------------
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
    # Compact L/R layout (rows of small text boxes)
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

# --------------------------------
# Inline H5P renderer — Local Blob per iframe
# --------------------------------
def _read_text(path: pathlib.Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None

def _local_assets_as_b64() -> Optional[Dict[str, str]]:
    main_js = _read_text(VENDOR_H5P_DIR / "main.bundle.js")
    frame_js = _read_text(VENDOR_H5P_DIR / "frame.bundle.js")
    css_txt = _read_text(VENDOR_H5P_DIR / "styles" / "h5p.css")
    if not (main_js and frame_js and css_txt):
        st.error("Local H5P assets missing. Add vendor/h5p/{main.bundle.js, frame.bundle.js, styles/h5p.css}.")
        return None
    return {
        "main_b64": base64.b64encode(main_js.encode("utf-8")).decode("utf-8"),
        "frame_b64": base64.b64encode(frame_js.encode("utf-8")).decode("utf-8"),
        "css": css_txt
    }

def render_h5p_inline(h5p_bytes: bytes, comp_id: str, height: int = 560):
    if not h5p_bytes:
        return

    assets = _local_assets_as_b64()
    if not assets:
        st.download_button("⬇️ Download activity (.h5p)",
                           data=h5p_bytes,
                           file_name=f"activity_{comp_id}.h5p",
                           mime="application/zip",
                           key=f"dl-missing-{comp_id}")
        return

    b64 = base64.b64encode(h5p_bytes).decode("utf-8")
    container_id = f"h5p-container-{comp_id}"
    css_txt = assets["css"]
    main_b64 = assets["main_b64"]
    frame_b64 = assets["frame_b64"]

    html = f"""
    <style>{css_txt}</style>
    <div id="{container_id}"></div>
    <script>
      (function() {{
        function b64ToUrl(b64, mime) {{
          var binStr = atob(b64);
          var len = binStr.length;
          var arr = new Uint8Array(len);
          for (var i=0; i<len; i++) arr[i] = binStr.charCodeAt(i);
          var blob = new Blob([arr], {{type: mime}});
          return URL.createObjectURL(blob);
        }}
        function loadScript(url) {{
          return new Promise(function(resolve, reject) {{
            var s = document.createElement('script');
            s.src = url;
            s.onload = function() {{ resolve(); }};
            s.onerror = function(e) {{ reject(e); }};
            document.head.appendChild(s);
          }});
        }}
        var mainUrl = b64ToUrl("{main_b64}", "text/javascript");
        var frameUrl = b64ToUrl("{frame_b64}", "text/javascript");

        // Load MAIN, then FRAME, then boot in THIS iframe
        loadScript(mainUrl)
          .then(function() {{ return loadScript(frameUrl); }})
          .then(function() {{
            var tries = 0;
            function boot() {{
              tries++;
              if (window.H5PStandalone && typeof window.H5PStandalone.display === 'function') {{
                try {{
                  window.H5PStandalone.display("#{container_id}", {{
                    h5pContent: "data:application/zip;base64,{b64}"
                  }});
                }} catch (e) {{
                  document.getElementById('{container_id}').innerHTML =
                    "<p style='color:#b00'>Couldn’t initialize H5P (local). Use the download button below.</p>";
                }}
              }} else if (tries < 240) {{
                setTimeout(boot, 50);
              }} else {{
                document.getElementById('{container_id}').innerHTML =
                  "<p style='color:#b00'>H5P API not ready (local). Use the download button below.</p>";
              }}
            }}
            boot();
          }})
          .catch(function(err) {{
            document.getElementById('{container_id}').innerHTML =
              "<p style='color:#b00'>Failed to load H5P libraries (local). Use the download button below.</p>";
          }});
      }})();
    </script>
    """
    st.components.v1.html(html, height=height, scrolling=True)

    run_id = st.session_state.run_id or "norun"
    digest = hashlib.md5(h5p_bytes).hexdigest()[:8]
    st.download_button("⬇️ Download activity (.h5p)",
                       data=h5p_bytes,
                       file_name=f"activity_{comp_id}.h5p",
                       mime="application/zip",
                       key=f"dl-{comp_id}-{run_id}-{digest}")

# --------------------------------
# Sidebar / Controls
# --------------------------------
with st.sidebar:
    st.header("Authoring")
    st.session_state.n_items = st.slider("Items per FIB", 2, 6, st.session_state.n_items, 1)
    st.session_state.gen_mcq = st.toggle("Also generate MCQs", value=st.session_state.gen_mcq)
    st.session_state.gen_dnd = st.toggle("Also generate Drag-and-Drop", value=st.session_state.gen_dnd)
    st.caption("All generations use ONLY your slides; if thin, adds a small OpenStax fallback snippet.")

topic = st.text_input("What do you want help with? (e.g., “electron transport chain”, “RTK”):", value=st.session_state.topic)
generate_clicked = st.button("Generate Activity")

# --------------------------------
# Generate (once) and cache by run_id
# --------------------------------
if generate_clicked:
    st.session_state.run_id = uuid.uuid4().hex[:8]
    st.session_state.topic = topic
    st.session_state.cache[st.session_state.run_id] = {"fib_bytes": None, "dnd_bytes": None, "mcqs": []}

    # FIB
    try:
        lines = gen_fib_lines_from_context(topic, difficulty="medium", n_items=st.session_state.n_items)
        instructions = f"Based ONLY on your course slides: predict the downstream effect for **{st.session_state.topic}** (answer with *increase/decrease*)."
        st.session_state.cache[st.session_state.run_id]["fib_bytes"] = build_fib_from_master(instructions, lines)
    except Exception as e:
        st.error(f"FIB generation failed: {e}")

    # MCQs
    if st.session_state.gen_mcq:
        try:
            st.session_state.cache[st.session_state.run_id]["mcqs"] = gen_mcqs_from_context(topic, difficulty="medium", n_items=2)
        except Exception as e:
            st.warning(f"MCQ generation failed: {e}")

    # DnD
    if st.session_state.gen_dnd:
        try:
            pairs = gen_dnd_pairs_from_context(topic, n_pairs=5)
            if pairs:
                st.session_state.cache[st.session_state.run_id]["dnd_bytes"] = build_dnd_from_master(pairs)
        except Exception as e:
            st.warning(f"DnD generation failed: {e}")

# --------------------------------
# Render last run
# --------------------------------
current_run = st.session_state.run_id
if current_run and current_run in st.session_state.cache:
    data = st.session_state.cache[current_run]

    if data.get("fib_bytes"):
        st.success("Fill-in-the-Blanks (slides-only)")
        try:
            render_h5p_inline(data["fib_bytes"], comp_id=f"fib-{current_run}", height=560)
        except Exception as e:
            st.error(f"Render error (FIB): {e}")

    mcqs = data.get("mcqs", [])
    if mcqs:
        st.subheader("Quick Check (slides-only)")
        for idx, q in enumerate(mcqs):
            key = f"{current_run}-mcq-{idx}"
            sel = st.radio(q["question"], q["options"], index=None, key=key)
            if sel is not None:
                correct = (q["options"].index(sel) == q["answer_index"])
                st.markdown("✅ Correct!" if correct else "❌ Not quite — recheck the slides.")
                st.divider()

    if data.get("dnd_bytes"):
        st.success("Drag-and-Drop (slides-only)")
        try:
            render_h5p_inline(data["dnd_bytes"], comp_id=f"dnd-{current_run}", height=520)
        except Exception as e:
            st.error(f"Render error (DnD): {e}")
