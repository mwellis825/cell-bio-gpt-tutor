import io, json, re, zipfile, base64, os, hashlib
from typing import List, Dict, Any
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
from pathlib import Path

# ---------- Config ----------
st.set_page_config(page_title="Cell Bio Tutor — Inline H5P (eval loader)", layout="centered")
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
    "Sentence with **answer**",
    "Sentence with **answer**",
    "Sentence with **answer**"
  ]
}}

Rules:
- 4–7 clozes total.
- Each **answer** is 1–2 words present in the slides.
- Prefer causal/function statements over trivia; keep sentences short.
- DO NOT add extra fields; DO NOT use markdown other than **answer** markers.

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
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("h5p.json", json.dumps(h5p_json, ensure_ascii=False))
        z.writestr("content/content.json", json.dumps(content_json, ensure_ascii=False))
    return buf.getvalue()

def load_runtime_b64_from_repo():
    root = Path(__file__).parent
    files = {
        "main":  root / "runtime" / "main.bundle.js",
        "frame": root / "runtime" / "frame.bundle.js",
        "css":   root / "runtime" / "h5p.css",
    }
    out, dbg = {}, {}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing runtime assets: " + ", ".join(missing))
    for key, path in files.items():
        data = path.read_bytes()
        out[key] = base64.b64encode(data).decode("ascii")
        dbg[key] = {"path": str(path), "size_bytes": len(data), "md5": hashlib.md5(data).hexdigest()}
    return out, dbg

def render_h5p_inline_from_b64(h5p_b64: str, runtime_b64: dict, height: int = 760):
    """Eval the decoded JS directly in the iframe, with AMD/CJS neutralized."""
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
  function toText(b64) {{
    // decode base64 into a JS string (latin-1 safe)
    const bin = atob(b64); let s = '';
    for (let i=0;i<bin.length;i++) s += String.fromCharCode(bin.charCodeAt(i));
    return s;
  }}
  function injectCSS(b64) {{
    const css = toText(b64);
    const blob = new Blob([css], {{type:'text/css'}});
    const url  = URL.createObjectURL(blob);
    const link = document.createElement('link');
    link.rel='stylesheet'; link.href=url; document.head.appendChild(link);
  }}
  // Minimal H5PIntegration required by frame bundle
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
<script>
  (function boot(){{
    try {{
      injectCSS("{css_b64}");

      // ---- AMD/CJS NEUTRALIZER ----
      const oldDefine  = window.define;
      const oldModule  = window.module;
      const oldExports = window.exports;
      try {{
        Object.defineProperty(window,'define',{{value: undefined, configurable: true}});
        Object.defineProperty(window,'module',{{value: undefined, configurable: true}});
        Object.defineProperty(window,'exports',{{value: undefined, configurable: true}});
      }} catch(_){{
        window.define = undefined; window.module = undefined; window.exports = undefined;
      }}

      // EVAL the decoded bundles directly (no <script src>, no blob src)
      const mainCode  = toText("{main_b64}");
      const frameCode = toText("{frame_b64}");
      (new Function(mainCode))();   // executes in global scope
      (new Function(frameCode))();

      // restore AMD/CJS if they existed
      if (oldDefine !== undefined) window.define = oldDefine; else delete window.define;
      if (oldModule !== undefined) window.module = oldModule; else delete window.module;
      if (oldExports !== undefined) window.exports = oldExports; else delete window.exports;

      // Resolve export
      let HS = window.H5PStandalone || window['h5p-standalone'];
      if (!HS && window.H5PStandalone && window.H5PStandalone.default) HS = window.H5PStandalone.default;
      if (!HS && window['h5p-standalone'] && window['h5p-standalone'].default) HS = window['h5p-standalone'].default;

      if (HS && typeof HS.display === 'function') {{
        const src = 'data:application/zip;base64,{h5p_b64}';
        HS.display('#app', {{ h5pContent: src }});
      }} else {{
        const diag = {{
          typeof_H5PStandalone: typeof window.H5PStandalone,
          typeof_hyphen: typeof window['h5p-standalone'],
          has_default_on_H5PStandalone: !!(window.H5PStandalone && window.H5PStandalone.default),
          has_default_on_hyphen: !!(window['h5p-standalone'] && window['h5p-standalone'].default)
        }};
        document.getElementById('app').innerHTML =
          '<div class="msg"><b>H5P API not ready (no display()).</b><pre>'+JSON.stringify(diag,null,2)+'</pre>' +
          '<p>Bundles were executed via eval(). If this persists, the files may be mismatched versions.</p></div>';
      }}
    }} catch(e) {{
      document.getElementById('app').innerHTML =
        '<div class="msg">Runtime error: '+(e && e.message || e)+'</div>';
    }}
  }})();
</script>
</body>
</html>
    """.strip()

    st.components.v1.html(html, height=height, scrolling=True)

# ---------- UI ----------
st.title("🧬 Cell Bio Tutor — Inline H5P (eval loader)")

with st.expander("Runtime files on disk"):
    try:
        runtime_b64, rt_debug = load_runtime_b64_from_repo()
        st.json(rt_debug)
    except Exception as e:
        st.error(str(e))
        runtime_b64, rt_debug = None, None

with st.expander("1) Upload your course slides (PDF)"):
    slides = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

topic = st.text_input("2) Topic (e.g., Electron transport chain)")
if st.button("Generate H5P"):
    if not runtime_b64:
        st.error("Runtime not loaded.")
        st.stop()
    slide_text = read_pdfs(slides)
    if not slide_text:
        st.error("No slide text detected. Please upload slides (PDF).")
        st.stop()
    data = ask_json(topic, slide_text)
    clozes = [c for c in data.get("clozes", []) if isinstance(c,str) and "**" in c][:7]
    if not clozes:
        st.error("No valid clozes found in model output.")
        st.stop()
    h5p_bytes = build_h5p_drag_words(data.get("title","Cell Bio Activity"), data.get("instructions","Fill the blanks"), clozes)
    h5p_b64 = base64.b64encode(h5p_bytes).decode("ascii")
    st.success("Generated H5P. Rendering inline below (look for the 'Runtime: local' badge)…")
    render_h5p_inline_from_b64(h5p_b64, runtime_b64)
    st.download_button("⬇️ Download this H5P", data=h5p_bytes, file_name="activity.h5p", mime="application/zip")
