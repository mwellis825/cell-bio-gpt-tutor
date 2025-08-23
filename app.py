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
    return txt[:12000]

def ask_json(topic: str, slide_text: str, model="gpt-4o-mini") -> Dict[str, Any]:
    user_prompt = f"""
Using ONLY these slide excerpts, author a concise Drag-the-Words activity for college Cell Biology on: "{topic}".

Return JSON ONLY with this exact shape:
{{
  "title": "short title",
  "instructions": "one clear task line",
  "clozes": [
    "Sentence with **answer**",
    "Sentence with **answer**"
  ]
}}

SLIDES:
{slide_text}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=700,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":SYSTEM},
                {"role":"user","content":user_prompt},
            ],
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return {}

def build_h5p_drag_words(title: str, instructions: str, clozes: List[str]) -> bytes:
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
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("h5p.json", json.dumps(h5p_json, ensure_ascii=False))
        z.writestr("content/content.json", json.dumps(content_json, ensure_ascii=False))
    return buf.getvalue()

def load_runtime_b64_from_repo() -> dict:
    root = Path(__file__).parent
    files = {
        "main":  root / "runtime" / "main.bundle.js",
        "frame": root / "runtime" / "frame.bundle.js",
        "css":   root / "runtime" / "h5p.css",
    }
    out = {}
    for key, path in files.items():
        data = path.read_bytes()
        out[key] = base64.b64encode(data).decode("ascii")
    return out

def render_h5p_inline_from_b64(h5p_b64: str, runtime_b64: dict, height: int = 760):
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
</style>
<script>
  function blobUrl(b64, type){{
    const bin = atob(b64); const len = bin.length; const bytes = new Uint8Array(len);
    for (let i=0;i<len;i++) bytes[i] = bin.charCodeAt(i);
    return URL.createObjectURL(new Blob([bytes], {{type}}));
  }}
  window.H5PIntegration = {{ baseUrl: ".", url: ".", l10n:{{}}, contents:{{}} }};
</script>
</head>
<body>
<div id="app"></div>
<div class="badge">Runtime: local</div>
<link id="h5pcss" rel="stylesheet">
<script>
  document.getElementById('h5pcss').href = blobUrl("{css_b64}", "text/css");
  const mainUrl  = blobUrl("{main_b64}",  "application/javascript");
  const frameUrl = blobUrl("{frame_b64}", "application/javascript");

  function loadScript(src) {{
    return new Promise((res,rej)=>{{ let s=document.createElement('script'); s.src=src; s.onload=res; s.onerror=rej; document.body.appendChild(s); }});
  }}

  function resolveHS() {{
    let HS = window.H5PStandalone || window['h5p-standalone'];
    if (HS && typeof HS.display==='function') return HS;
    if (HS && HS.default && typeof HS.default.display==='function') return HS.default;
    return null;
  }}

  (async function boot(){{
    try {{
      await loadScript(mainUrl);
      await loadScript(frameUrl);
      const HS = resolveHS();
      if (HS) {{
        const src = 'data:application/zip;base64,{h5p_b64}';
        HS.display('#app', {{ h5pContent: src }});
      }} else {{
        document.getElementById('app').innerHTML = '<p>H5P API not ready.</p>';
      }}
    }} catch(e) {{
      document.getElementById('app').innerHTML = '<p>Load failed: '+e+'</p>';
    }}
  }})();
</script>
</body>
</html>
    """
    st.components.v1.html(html, height=height, scrolling=True)

# ---------- UI ----------
st.title("🧬 Cell Bio Tutor — Inline H5P (local runtime)")

with st.expander("1) Upload your course slides (PDF)"):
    slides = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

topic = st.text_input("2) Topic (e.g., Electron transport chain)")
if st.button("Generate H5P"):
    slide_text = read_pdfs(slides)
    data = ask_json(topic, slide_text)
    clozes = [c for c in data.get("clozes", []) if "**" in c][:6]
    if not clozes:
        st.error("No valid clozes found.")
        st.stop()
    h5p_bytes = build_h5p_drag_words(data.get("title","Cell Bio Activity"), data.get("instructions","Fill the blanks"), clozes)
    h5p_b64 = base64.b64encode(h5p_bytes).decode("ascii")
    runtime_b64 = load_runtime_b64_from_repo()
    render_h5p_inline_from_b64(h5p_b64, runtime_b64)
