import io, json, re, zipfile, base64, requests, os
from typing import List, Dict, Any
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# ---------- Config ----------
st.set_page_config(page_title="Cell Bio Tutor — Inline H5P (blob runtime)", layout="centered")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets or environment.")
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM = (
    "You are a rigorous, supportive Cell Biology tutor. "
    "You ONLY use the provided slide excerpts to author concise, causal, critical-thinking practice."
)

H5P_VER = "1.3.0"
JSDELIVR = f"https://cdn.jsdelivr.net/npm/h5p-standalone@{H5P_VER}/dist"
UNPKG    = f"https://unpkg.com/h5p-standalone@{H5P_VER}/dist"

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
    """
    2-pass JSON enforcement.
    Pass 1: response_format=json_object (strict).
    Pass 2: low-temp with a tiny few-shot example if needed.
    Returns dict or {}.
    """
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

    # Pass 1: strict JSON
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
        txt = resp.choices[0].message.content or ""
        return json.loads(txt)
    except Exception as e:
        st.warning(f"Retrying with a stricter format (pass 2). First error: {e}")

    # Pass 2: add a tiny example and lower temp
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
        txt2 = resp2.choices[0].message.content or ""
        return json.loads(txt2)
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

@st.cache_resource(show_spinner=False)
def fetch_runtime_b64() -> dict:
    """Download h5p-standalone runtime server-side and return base64 blobs."""
    def fetch(url) -> bytes:
        r = requests.get(url, timeout=30)
        if not r.ok or len(r.content) < 10_000:
            raise RuntimeError(f"Bad fetch: {url}")
        return r.content

    prim = [
        (f"{JSDELIVR}/main.bundle.js", "main"),
        (f"{JSDELIVR}/frame.bundle.js","frame"),
        (f"{JSDELIVR}/styles/h5p.css", "css"),
    ]
    fb = [
        (f"{UNPKG}/main.bundle.js","main"),
        (f"{UNPKG}/frame.bundle.js","frame"),
        (f"{UNPKG}/styles/h5p.css","css"),
    ]

    out = {}
    for primary, key in prim:
        try:
            out[key] = base64.b64encode(fetch(primary)).decode("ascii")
        except Exception:
            alt = next(u for u,k in fb if k==key)
            out[key] = base64.b64encode(fetch(alt)).decode("ascii")
    return out

def render_h5p_inline_from_b64(h5p_b64: str, runtime_b64: dict, height: int = 760):
    """Create Blob URLs in iframe for runtime; feed base64 .h5p to H5PStandalone.display()."""
    main_b64  = runtime_b64["main"]
    frame_b64 = runtime_b64["frame"]
    css_b64   = runtime_b64["css"]

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body,#app{{margin:0;height:100%}} .badge{{position:fixed;right:8px;top:8px;padding:4px 8px;background:#eef;border:1px solid #99c;border-radius:6px;font:12px sans-serif;opacity:.9}}</style>
<script>
  function blobUrl(b64, type){{
    const bin = atob(b64); const len = bin.length; const bytes = new Uint8Array(len);
    for (let i=0;i<len;i++) bytes[i] = bin.charCodeAt(i);
    return URL.createObjectURL(new Blob([bytes], {{type}}));
  }}
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
<div class="badge">Runtime: blob</div>
<link id="h5pcss" rel="stylesheet">
<script>
  document.getElementById('h5pcss').href = blobUrl("{css_b64}", "text/css");
  const mainUrl  = blobUrl("{main_b64}",  "application/javascript");
  const frameUrl = blobUrl("{frame_b64}", "application/javascript");

  function loadScript(src){{
    return new Promise((res, rej)=>{{ const s=document.createElement('script'); s.src=src; s.onload=res; s.onerror=rej; document.body.appendChild(s); }});
  }}

  (async function boot(){{
    try {{
      await loadScript(mainUrl);
      await loadScript(frameUrl);
      const HS = window.H5PStandalone || window['h5p-standalone'];
      if (HS && typeof HS.display === 'function') {{
        const src = 'data:application/zip;base64,{h5p_b64}';
        HS.display('#app', {{ h5pContent: src }});
      }} else {{
        document.getElementById('app').innerHTML = '<div style="padding:12px;font:14px system-ui">H5P API not ready (no display()).</div>';
      }}
    }} catch(e) {{
      document.getElementById('app').innerHTML = '<div style="padding:12px;font:14px system-ui">Failed loading runtime: '+(e && e.message || e)+'</div>';
    }}
  }})();
</script>
</body>
</html>
    """.strip()

    st.components.v1.html(html, height=height, scrolling=True)

# ---------- UI ----------
st.title("🧬 Cell Bio Tutor — Inline H5P (no external loads in browser)")

with st.expander("1) Upload your course slides (PDF)"):
    slides = st.file_uploader("Upload 1–10 PDFs", type=["pdf"], accept_multiple_files=True)

topic = st.text_input("2) Topic (e.g., Electron transport chain, RTK signaling)")
go = st.button("Generate & Render Inline H5P")

if go:
    slide_text = read_pdfs(slides)
    if not slide_text:
        st.error("No slide text detected. Please upload slides (PDF).")
    else:
        data = ask_json(topic=topic, slide_text=slide_text)
        if not data:
            st.error("Could not parse LLM output after two attempts. Try a narrower topic or different slides.")
        else:
            title = data.get("title") or f"{topic} — Drag the Words"
            instructions = data.get("instructions") or "Fill the missing terms."
            clozes_raw = data.get("clozes", [])
            clozes = [c for c in clozes_raw if isinstance(c, str) and "**" in c][:7]

            if not clozes:
                st.error("No valid clozes with **answer** markers returned. Try regenerating.")
            else:
                # Build H5P and runtime
                h5p_bytes = build_h5p_drag_words(title, instructions, clozes)
                h5p_b64   = base64.b64encode(h5p_bytes).decode("ascii")
                try:
                    runtime_b64 = fetch_runtime_b64()
                except Exception as e:
                    st.error(f"Could not fetch H5P runtime server-side: {e}")
                    runtime_b64 = None

                if runtime_b64:
                    st.success("Generated H5P. Rendering inline below (look for the 'Runtime: blob' badge)…")
                    render_h5p_inline_from_b64(h5p_b64, runtime_b64, height=760)

                # Optional: also offer download (LMS backup)
                safe_name = re.sub(r'[^a-z0-9]+','_',title.lower()).strip('_') + ".h5p"
                st.download_button("⬇️ Download this H5P", data=h5p_bytes, file_name=safe_name, mime="application/zip")
