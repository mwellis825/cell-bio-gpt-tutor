# app.py — Cell Bio Tutor (Slides-Only, LLM-authored, Offline H5P)

import os, io, json, time, base64, zipfile, pathlib, re
from typing import List, Dict, Optional, Tuple
import streamlit as st
import numpy as np
from pypdf import PdfReader

# -------------------------------
# Page / Paths
# -------------------------------
st.set_page_config(page_title="Cell Bio Tutor — Slides Only", layout="centered")
st.title("🧬 Cell Bio Tutor — Slides-Only (LLM + H5P)")

ROOT = pathlib.Path(__file__).parent
SLIDES_DIR = ROOT / "slides"
FIB_MASTER = ROOT / "templates" / "rtk_fill_in_blanks_FIXED_blocks.h5p"
DND_MASTER = ROOT / "templates" / "cellular_respiration_aligned_course_style.h5p"
VENDOR_H5P_DIR = ROOT / "vendor" / "h5p"  # contains frame.bundle.js, main.bundle.js, styles/h5p.css

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
        st.warning("Missing OPENAI_API_KEY (set in Streamlit Secrets or env). Using no-AI mode.")
        return None
    try:
        return OpenAI(api_key=key)
    except Exception as e:
        st.error(f"Could not initialize OpenAI client: {e}")
        return None

client = get_openai_client()
EMBED_MODEL = "text-embedding-3-small"  # economical 1536-d

# -------------------------------
# Session state
# -------------------------------
def init_state():
    defaults = dict(
        topic="",
        n_items=3,
        gen_mcq=True,
        gen_dnd=True,
        index_ready=False,
        index_chunks=[],
        index_embeds=None,
        H5P_FRAME_JS=None,
        H5P_MAIN_JS=None,
        H5P_CSS=None,
    )
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# -------------------------------
# Slides ingestion & embeddings
# -------------------------------
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
    """Build index (chunks + embeddings) once per session."""
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

    if client:
        try:
            texts = [c["text"] for c in chunks]
            embeds = []
            BATCH = 64
            for i in range(0, len(texts), BATCH):
                batch = texts[i:i+BATCH]
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
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
    """Return top-k chunks most similar to query."""
    if not ensure_index():
        return []
    chunks = st.session_state.index_chunks

    # No embeddings → naive keyword scoring
    if client is None or st.session_state.index_embeds is None:
        scored = []
        q = query.lower()
        q_toks = [t for t in re.findall(r"\w+", q) if t]
        for ch in chunks:
            text = ch["text"].lower()
            score = sum(text.count(tok) for tok in q_toks)
            if score > 0:
                scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for s,c in scored[:k]] or chunks[:k]

    # Embedding cosine similarity
    qe = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    qe = np.array(qe, dtype=np.float32)
    M = st.session_state.index_embeds
    sims = (M @ qe) / (np.linalg.norm(M, axis=1) * (np.linalg.norm(qe) + 1e-8) + 1e-8)
    top_idx = np.argsort(-sims)[:k]
    return [st.session_state.index_chunks[i] for i in top_idx]

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
    """Concise fallback text (can replace with local files under repo/openstax/)."""
    return (
        "Cellular respiration spans glycolysis, pyruvate oxidation, the citric acid cycle, "
        "and oxidative phosphorylation (electron transport + chemiosmosis). The ETC across the inner "
        "mitochondrial membrane generates a proton gradient used by ATP synthase. Regulation is influenced "
        "by allosteric effectors, energy charge (ATP/ADP/AMP), and membrane integrity."
    )

def build_context_with_fallback(query: str, min_chars: int = 1200) -> Tuple[str, List[Tuple[str,int]]]:
    ctx, src = build_context(query, max_chars=6000)
    if len(ctx) < min_chars:
        fb = load_openstax_fallback()
        ctx = (ctx + "\n\n[OpenStax fallback]\n" + fb) if ctx else fb
    return ctx, src

# -------------------------------
# LLM generation (slides-only + fallback)
# -------------------------------
def gen_fib_lines_from_context(topic: str, difficulty: str, n_items: int) -> List[str]:
    if client is None:
        raise RuntimeError("OpenAI client not initialized.")
    context, sources = build_context_with_fallback(topic)
    sys = (
        "You are a Cell Biology tutor bound to the provided context. "
        "Only use information present in the context/fallback; do NOT invent facts. "
        "Avoid definition- or recall-style prompts. Prefer causal, predictive reasoning about perturbations."
    )
    user = f"""
Topic: {topic}
Difficulty: {difficulty}

Context:
{context}

Task:
Produce {n_items} concise, critical-thinking Fill-in-the-Blanks items using H5P syntax.
Rules:
- Exactly ONE sentence and ONE blank per item.
- The blank should be *increase/increased* or *decrease/decreased* (or similarly predictive outcomes).
- Use asterisks for acceptable answers/variants, e.g., *increase/increased*.
- Keep cognitive load low but conceptual (predictive). No definition recall (e.g., "X is Y").
- Do NOT repeat identical structures; vary components perturbed and downstream targets.
- If context is insufficient, write items that say 'Not in slides *increase/decrease*.' (rare).

Return as a numbered list of plain lines.
"""
    def call_llm(temp=0.2):
        resp = client.chat.completions.create(
            model="gpt-4o", temperature=temp,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        return resp.choices[0].message.content or ""

    text = call_llm(0.2)

    def parse_and_filter(text: str) -> List[str]:
        out = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            s = s.lstrip("0123456789). ").strip()
            # must contain *answer* and be predictive (heuristic)
            predictive = ("*increase" in s.lower()) or ("*decrease" in s.lower())
            if "*" in s and predictive and not re.search(r"\bis\b|\bare\b|\bdefined as\b", s.lower()):
                out.append(s)
        return out

    lines = parse_and_filter(text)
    if len(lines) < n_items:
        # regenerate once with slightly higher temperature to diversify
        text2 = call_llm(0.4)
        lines = (lines + parse_and_filter(text2))[:n_items]

    if not lines:
        lines = ["If proton leak increases across the inner membrane, ATP synthase output will *decrease/decreased*."]
    return lines[:n_items]


Context:
{context}

Task:
Produce {n_items} concise, critical-thinking Fill-in-the-Blanks items using H5P syntax.
Rules:
- ONE sentence per item.
- Exactly ONE blank per sentence.
- Use asterisks for acceptable answers and variants, e.g., *increase/increased*.
- Keep cognitive load low. Focus on 'predict increase/decrease' consequences.
- Do NOT invent facts beyond the context.

Return as a numbered list of plain lines."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o", temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s: 
            continue
        s = s.lstrip("0123456789). ").strip()
        if "*" in s:
            lines.append(s)
    if not lines:
        lines = ["If proton leak increases across the inner membrane, ATP synthase output will *decrease/decreased*."]
    return lines[:n_items]

def gen_mcqs_from_context(topic: str, difficulty: str, n_items: int = 2) -> List[Dict]:
    if client is None:
        return []
    context, _ = build_context_with_fallback(topic)
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

Keep wording short. Use only slide/fallback facts. If insufficient, return [].
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o", temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        raw = resp.choices[0].message.content or "[]"
    except Exception:
        return []
    # Try to parse JSON from response
    try:
        j = json.loads(raw.strip(" \n`"))
        return j if isinstance(j, list) else []
    except Exception:
        m = re.search(r"\[.*\]", raw, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return []
        return []

def gen_dnd_pairs_from_context(topic: str, n_pairs: int = 5) -> List[Tuple[str,str]]:
    if client is None:
        return []

    # Try slides + fallback first
    context, _ = build_context_with_fallback(topic)
    sys = ("Create Drag-and-Drop pairs ONLY from context; each pair maps a concise term to a matching description/location/output.")
    user = f"""
Context:
{context}

Task:
Provide {n_pairs} pairs for a Drag-and-Drop activity as a JSON array of objects:
- drag: short text (term/step) <= 8 words
- drop: short label (matching description/location/output) <= 10 words
Constraints:
- Use distinct pairs; avoid duplicates.
- Keep both texts concise and readable for undergrads.
- Only use information explicitly present in context/fallback.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o", temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        raw = resp.choices[0].message.content or "[]"
        arr = json.loads(raw.strip(" \n`"))
        pairs = []
        for obj in arr:
            if isinstance(obj, dict) and "drag" in obj and "drop" in obj:
                pairs.append((obj["drag"], obj["drop"]))
        pairs = pairs[:n_pairs]
    except Exception:
        pairs = []

    if len(pairs) >= max(3, n_pairs//2):
        return pairs

    # Heuristic fallback by topic (OpenStax-style)
    t = topic.lower()
    if "electron transport" in t or "etc" in t or "oxidative" in t:
        fallback = [
            ("Complex I", "NADH → e⁻, pumps H⁺"),
            ("Complex II", "FADH₂ → e⁻ (no pumping)"),
            ("Complex III", "Q → Cyt c, pumps H⁺"),
            ("Complex IV", "O₂ → H₂O, pumps H⁺"),
            ("ATP synthase", "H⁺ gradient → ATP"),
        ]
    elif "glycolysis" in t:
        fallback = [
            ("Hexokinase", "Glucose → G6P"),
            ("PFK-1", "F6P → F1,6BP"),
            ("Pyruvate kinase", "PEP → Pyruvate"),
            ("NAD⁺ reduction", "Generates NADH"),
            ("ATP yield", "Net 2 ATP"),
        ]
    elif "rtk" in t or "receptor tyrosine" in t:
        fallback = [
            ("Ligand binding", "Dimerization"),
            ("Autophosphorylation", "Tyr residues"),
            ("Grb2/SOS", "Ras activation"),
            ("MAPK cascade", "Phosphorylation"),
            ("PI3K → AKT", "Pro-survival"),
        ]
    else:
        fallback = [
            ("Nucleus", "DNA storage"),
            ("ER", "Protein folding"),
            ("Golgi", "Modification/Sorting"),
            ("Lysosome", "Acid hydrolases"),
            ("Mitochondria", "ATP production"),
        ]
    return fallback[:n_pairs]


Task:
Provide {n_pairs} pairs for a Drag-and-Drop activity as a JSON array of objects:
- drag: short text (term/step) <= 8 words
- drop: short label (matching description/location/output) <= 8 words

Only use information explicitly present in context/fallback.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o", temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        raw = resp.choices[0].message.content or "[]"
    except Exception:
        return []
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
# H5P builders from MASTER files
# -------------------------------
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
    # Compact left/right geometry (5 rows)
    elements, zones = [], []
    y_dr = 6.5
    y_zn = 13.9
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
    except Exception as e:
        st.error(f"Couldn’t build DnD H5P: {e}")
        return None

# -------------------------------
# Inline H5P renderer (LOCAL assets, no CDN)
# -------------------------------
def render_h5p_inline(h5p_bytes: bytes, height: int = 560):
    """Render .h5p using locally vendored h5p-standalone assets (no CDN)."""
    if not h5p_bytes:
        return

    # Load local JS/CSS once
    if st.session_state.H5P_FRAME_JS is None or st.session_state.H5P_MAIN_JS is None or st.session_state.H5P_CSS is None:
        try:
            st.session_state.H5P_MAIN_JS  = (VENDOR_H5P_DIR / "main.bundle.js").read_text(encoding="utf-8")
            st.session_state.H5P_FRAME_JS = (VENDOR_H5P_DIR / "frame.bundle.js").read_text(encoding="utf-8")
            st.session_state.H5P_CSS      = (VENDOR_H5P_DIR / "styles" / "h5p.css").read_text(encoding="utf-8")
        except Exception as e:
            st.error("Local H5P assets missing. Add vendor/h5p/{main.bundle.js, frame.bundle.js, styles/h5p.css}.")
            st.caption(f"(Details: {e})")
            st.download_button("⬇️ Download activity (.h5p)",
                               data=h5p_bytes, file_name="activity_generated.h5p", mime="application/zip")
            return

    b64 = base64.b64encode(h5p_bytes).decode("utf-8")

    # Load MAIN first, then FRAME; wait for H5PStandalone to exist
    html = f"""
    <style>{st.session_state.H5P_CSS}</style>
    <div id="h5p-container"></div>
    <script>
    {st.session_state.H5P_MAIN_JS}
    </script>
    <script>
    {st.session_state.H5P_FRAME_JS}
    </script>
    <script>
      (function() {{
        var tries = 0;
        function boot() {{
          tries += 1;
          if (window.H5PStandalone && typeof H5PStandalone.display === 'function') {{
            try {{
              H5PStandalone.display('#h5p-container', {{
                h5pContent: "data:application/zip;base64,{b64}"
              }});
            }} catch (e) {{
              document.getElementById('h5p-container').innerHTML =
                "<p style='color:#b00'>Couldn’t initialize H5P locally. Use the download button below.</p>";
            }}
          }} else if (tries < 40) {{
            setTimeout(boot, 100); // wait up to ~4s total
          }} else {{
            document.getElementById('h5p-container').innerHTML =
              "<p style='color:#b00'>H5P scripts loaded but API not ready. Try reload or use the download button below.</p>";
          }}
        }}
        boot();
      }})();
    </script>
    """
    st.components.v1.html(html, height=height, scrolling=True)
    st.download_button("⬇️ Download activity (.h5p)",
                       data=h5p_bytes, file_name="activity_generated.h5p", mime="application/zip")


# -------------------------------
# UI
# -------------------------------
with st.sidebar:
    st.header("Authoring")
    st.session_state.n_items = st.slider("Items per FIB", 2, 6, st.session_state.n_items, 1)
    st.session_state.gen_mcq = st.toggle("Also generate MCQs", value=st.session_state.gen_mcq)
    st.session_state.gen_dnd = st.toggle("Also generate Drag-and-Drop", value=st.session_state.gen_dnd)
    st.caption("All generations use ONLY your slides; if thin, adds a small OpenStax fallback paragraph.")

topic = st.text_input("What do you want help with? (e.g., “electron transport chain”, “RTK”):", value=st.session_state.topic)
generate = st.button("Generate Activity")

if generate:
    st.session_state.topic = topic

    # FIB (slides-only + fallback)
    fib_bytes = None
    try:
        lines = gen_fib_lines_from_context(topic, difficulty="medium", n_items=st.session_state.n_items)
        # after lines = gen_fib_lines_from_context(...)
instructions = f"Based ONLY on your course slides: predict the downstream effect for **{st.session_state.topic}** (use *increase/decrease* in the blank)."
fib_bytes = build_fib_from_master(instructions, lines)

    except Exception as e:
        st.error(f"FIB generation failed: {e}")

    if fib_bytes:
        st.success("Fill-in-the-Blanks (slides-only)")
        try:
            render_h5p_inline(fib_bytes, height=560)
        except Exception as e:
            st.error(f"Render error (FIB): {e}")

    # MCQs (slides-only + fallback)
    if st.session_state.gen_mcq:
        try:
            mcqs = gen_mcqs_from_context(topic, difficulty="medium", n_items=2)
            if mcqs:
                st.subheader("Quick Check (slides-only)")
                for idx, q in enumerate(mcqs):
                    key = f"mcq_{time.time()}_{idx}"
                    sel = st.radio(q["question"], q["options"], index=None, key=key)
                    if sel is not None:
                        correct = (q["options"].index(sel) == q["answer_index"])
                        st.markdown("✅ Correct!" if correct else "❌ Not quite — recheck the slides.")
                        st.divider()
            else:
                st.info("Not enough slide context to generate MCQs (after fallback).")
        except Exception as e:
            st.warning(f"MCQ generation failed: {e}")

    # DnD (slides-only + fallback)
    if st.session_state.gen_dnd:
        try:
            pairs = gen_dnd_pairs_from_context(topic, n_pairs=5)
            if pairs:
                dnd_bytes = build_dnd_from_master(pairs)
                if dnd_bytes:
                    st.success("Drag-and-Drop (slides-only)")
                    render_h5p_inline(dnd_bytes, height=520)
            else:
                st.info("Not enough slide context to build Drag-and-Drop (after fallback).")
        except Exception as e:
            st.warning(f"DnD generation failed: {e}")
