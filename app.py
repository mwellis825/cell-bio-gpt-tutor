import io, json, re, zipfile, base64, os
from typing import List, Dict, Any
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
from pathlib import Path

# ---------- Config ----------
st.set_page_config(page_title="Cell Bio Tutor — Inline H5P (local runtime)", layout="centered")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets or environment.")
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM = (
    "You are a rigorous, supportive Cell Biology tutor. "
    "You ONLY use the provided slide excerpts to author concise, causal, critical-thinking practice."
)

# ---------- Helpers ----------
def read_pdfs(files) -> str:
    out = []
    for f in files or []:
        try:
            r = PdfReader(f)
            out.append("\n".join((p.extract_text() or "") for p in r.pages))
        except Exception as e:
            st.warning(f"Could not read {getattr(f,'name','file')}: {e}")
    txt = re.sub(r"\s+", " ", "\n\n".join(out)).strip()
    return txt[:12000]  # keep prompt modest

def ask_json(topic: str, slide_text: str, model="gpt-4o-mini") -> Dict[str, Any]:
    base_user = f"""
Using ONLY these slide excerpts, author a concise Drag-the-Words activity for college Cell Biology on: "{topic}".

Return JSON ONLY with this exact shape:
{{
  "title": "short title",
  "instructions": "one clear task line",
  "clozes": [
    "Short sentence with **answer**",
    "Another sentence with **term**",
    "3rd sentence with **concept**",
    "4th sentence with **word**"
  ]
}}

Rules:
- 4–7 clozes total.
- Each **answer** is 1–2 words present in the slides.
- Prefer causal/function statements over trivia. Keep sentences short.
- DO NOT add extra fields. DO NOT use markdown outside the **answer** markers.
SLIDES:
{slide_text}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=700,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":SYSTEM},
                {"role":"user","content":base_user},
            ],
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        st.warning(f"Retrying with a stricter format (pass 2). First error: {e}")

    fewshot_user = base_user + """

EXAMPLE (structure only; content must come from slides):
{
  "title": "electron transport outcomes",
  "instructions": "Fill each blank with the correct term from the slides.",
  "clozes": [
    "Complex IV reduces **oxygen** to water.",
    "A proton gradient across the **inner membrane** drives ATP synthase.",
    "An inhibitor of Complex I lowers **NADH** oxidation.",
    "Uncouplers increase **oxygen consumption** but reduce ATP yield."
  ]
}
""".rstrip()

    try:
        resp2 = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=700,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":SYSTEM},
                {"role":"user","content":fewshot_user},
            ],
        )
        return json.loads(resp2.choices[0].message.content or "{}")
    except Exception as e2:
        st.error(f"OpenAI error (pass 2): {e2}")
        return {}

def build_h5p_drag_words(title: str, instructions: str, clozes: List[str]) -> bytes:
    """Minimal VALID H5P (Drag-the-Words). Accepts **word** and converts to *word*."""
    def to_dragtext(line: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"*\1*", line)

    content_json = {
        "taskDescription": instructions,
        "textField": "\n".join(to_dragtext(s) for s in clozes),
        "overallFeedback": [{"from": 0, "to": 100, "feedback": "Great job!"}],
        "behaviour": {
            "enableRetry": True, "enableSolutionsButton": True,
            "instantFeedback": True, "caseSensitive": False
        }
    }
    h5p_json = {
        "title": title,
        "language": "en",
        "mainLibrary": "H5P.DragText",
        "embedTypes": ["div"],
        "preloadedDependencies": [
            {"machineName": "H5P.DragText", "majorVersion": 1, "minorVersion": 8},
            {"machineName": "H5P.JoubelUI", "majorVersion": 1, "minorVersion": 3},
            {"machineName": "H5P.Transition", "majorVersion": 1, "minorVersion": 0},
        ],
    }
    libs = [
        ("H5P.DragText-1.8/library.json",
         {"machineName":"H5P.DragText","majorVersion":1,"minorVersion":8,"patchVersion":0,"language":"en","semantics":[]}),
        ("H5P.JoubelUI-1.3/library.json",
         {"machineName":"H5P.JoubelUI","majorVersion":1,"minorVersion":3}),
        ("H5P.Transition-1.0/library.json",
         {"machineName":"H5P.Transition","majorVersion":1,"minorVersion":0}),
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("h5p.json", json.dumps(h5p_json, ensure_ascii=False))
        z.writestr("content/content.json", json.dumps(content_json, ensure_ascii=False))
        for path, payload in libs:
            z.writestr(f"libraries/{path}", json.dumps(payload, ensure_ascii=False))
    return buf.getvalue()

def load_runtime_b64_from_repo() -> dict:
    """Read ./runtime files and return base64 strings."""
    root = Path(__file__).parent
    files = {
        "main":  root / "runtime" / "main.bundle.js",
        "frame": root / "runtime" / "frame.bundle.js",
        "css":   root / "runtime" / "h5p.css",
    }
    out = {}
    missing = [k for k,p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing runtime assets: " + ", ".join(missing) +
            ". Add runtime/main.bundle.js, runtime/frame.bundle.js, runtime/h5p.css to the repo."
        )
    for key, path in files.items():
        data = path.read_bytes()
        if len(data) < 10_000:
            raise ValueError(f"Runtime file too small or corrupted: {path}")
        out[key] = base64.b64encode(data).decode("ascii")
    return out

def render_h5p_inline_from_b64(h5p_b64: str, runtime_b64: dict, height: int = 760):
    """Create Blob URLs; robustly resolve H5PStandalone export (with .default fallback)."""
    main_b64  = runtime_b64["main"]
    frame_b64 = runtime_b64["frame"]
    css_b64   = runtime_b64["css"]

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  html,body,#app{{margin:0;height:100%}}
  .badge{{position:fixed;right:8px;top:8px;padding:4px 8px;background:#eef;border:1px solid #99c;border-radius:6px;font:12px sans-serif;opacity:.9}}
  .msg{{padding:12px;font:14px system-ui}}
  pre{{white-space:pre-wrap;word-break:break-word;background:#f7f7f8;padding:8px;border-radius:6px}}
</style>
<script>
  function blobUrl(b64, type){{
    const bin = atob(b64); const len = bin.length; const bytes = new Uint8Array(len);
    for (let i=0;i<len;i++) bytes[i] = bin.charCodeAt(i);
    return URL.createObjectURL(new Blob([bytes], {{type}}));
  }}
  // Required by frame.bundle.js
  window.H5PIntegration = {{
    baseUrl: location.origin, url: location.href, siteUrl: location.origin,
    hubIsEnabled: false, disableHub: true, postUserStatistics: false, saveFreq: false,
    l10n: {{ H5P: {{ fullscreen: "Fullscreen", exitFullscreen: "Exit fullscreen" }} }},
    ajax: {{ setFinished: "", contentUserData: "" }},
    libraryUrl: "./", core: {{ scripts: [], styles: [] }}, loadedJs: [], loadedCss: [], contents: {{}}
  }};
</script>
</head>
<body>
<div id="app"></div>
<div class="badge">Runtime: local</div>
<link id="h5pcss" rel="stylesheet">
<script>
  // Inject CSS
  document.getElementById('h5pcss').href = blobUrl("{css_b64}", "text/css");

  // Load JS in order
  const mainUrl  = blobUrl("{main_b64}",  "application/javascript");
  const frameUrl = blobUrl("{frame_b64}", "application/javascript");

  function loadScript(src){{
    return new Promise((res, rej)=>{{
      const s=document.createElement('script'); s.src=src; s.onload=res; s.onerror=rej; document.body.appendChild(s);
    }});
  }}

  function resolveHS() {{
    // Try all common export patterns
    let HS = window.H5PStandalone || window['h5p-standalone'];
    if (HS && typeof HS.display === 'function') return HS;
    if (HS && HS.default && typeof HS.default.display === 'function') return HS.default;
    // Some builds attach to window as { default: fn }
    const maybe = (window.H5PStandalone && window.H5PStandalone.default) || (window['h5p-standalone'] && window['h5p-standalone'].default);
    if (maybe && typeof maybe.display === 'function') return maybe;
    return null;
  }}

  (async function boot(){{
    try {{
      await loadScript(mainUrl);
      await loadScript(frameUrl);
      const HS = resolveHS();
      if (HS && typeof HS.display === 'function') {{
        const src = 'data:application/zip;base64,{h5p_b64}';
        HS.display('#app', {{ h5pContent: src }});
      }} else {{
        const diag = {{
          typeof_H5PStandalone: typeof window.H5PStandalone,
          typeof_h5p_standalone: typeof window['h5p-standalone'],
          has_default_on_H5PStandalone: !!(window.H5PStandalone && window.H5PStandalone.default),
          has_default_on_hyphen: !!(window['h5p-standalone'] && window['h5p-standalone'].default),
          keys_H5PStandalone: (window.H5PStandalone && Object.keys(window.H5PStandalone)) || null,
          keys_hyphen: (window['h5p-standalone'] && Object.keys(window['h5p-standalone'])) || null
        }};
        document.getElementById('app').innerHTML =
          '<div class="msg"><b>H5P API not ready (no display()).</b><br>Diagnostics:<br><pre>'+JSON.stringify(diag,null,2)+'</pre>' +
          'Confirm that <code>runtime/main.bundle.js</code> and <code>runtime/frame.bundle.js</code> are the UMD dist files from <code>h5p-standalone@1.3.0/dist</code> (not HTML, not source builds).</div>';
      }}
    }} catch(e) {{
      document.getElementById('app').innerHTML =
        '<div class="msg">Failed loading runtime: '+(e && e.message || e)+'</div>';
    }}
  }})();
</script>
</body>
</html>
    """.strip()

    st.components.v1.html(html, height=height, scrolling=True)

# ---------- UI ----------
st.title("🧬 Cell Bio Tutor — Inline H5P (local runtime, no network)")

with st.expander("1) Upload your course slides (PDF)"):
    slides = st.file_uploader("Upload 1–10 PDFs", type=["pdf"], accept_multiple_files=True)

topic = st.text_input("2) Topic (e.g., Electron transport chain, RTK signaling)")
go = st.button("Generate & Render Inline H5P")

if go:
    # 1) Slides
    slide_text = read_pdfs(slides)
    if not slide_text:
        st.error("No slide text detected. Please upload slides (PDF).")
        st.stop()

    # 2) LLM → JSON
    data = ask_json(topic=topic, slide_text=slide_text)
    if not data:
        st.error("Could not parse LLM output after two attempts. Try a narrower topic or different slides.")
        st.stop()

    title = data.get("title") or f"{topic} — Drag the Words"
    instructions = data.get("instructions") or "Fill the missing terms."
    clozes_raw = data.get("clozes", [])
    clozes = [c for c in clozes_raw if isinstance(c, str) and "**" in c][:7]
    if not clozes:
        st.error("No valid clozes with **answer** markers returned. Try regenerating.")
        st.stop()

    # 3) Build H5P
    h5p_bytes = build_h5p_drag_words(title, instructions, clozes)
    h5p_b64   = base64.b64encode(h5p_bytes).decode("ascii")

    # 4) Load local runtime
    try:
        runtime_b64 = load_runtime_b64_from_repo()
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.success("Generated H5P. Rendering inline below (look for the 'Runtime: local' badge)…")
    render_h5p_inline_from_b64(h5p_b64, runtime_b64, height=760)

    # 5) Optional download
    safe_name = re.sub(r'[^a-z0-9]+','_',title.lower()).strip('_') + ".h5p"
    st.download_button("⬇️ Download this H5P", data=h5p_bytes, file_name=safe_name, mime="application/zip")
