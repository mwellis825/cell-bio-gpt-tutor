import io, json, re, zipfile, base64, requests
from typing import List
import streamlit as st
from pypdf import PdfReader
import openai

# ---------------- Config ----------------
st.set_page_config(page_title="Cell Bio Tutor — Inline H5P (no-CDN)", layout="centered")
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

SYSTEM = (
    "You are a cell biology tutor. Only use the provided slide excerpts "
    "to author concise, causal, critical-thinking practice items."
)

H5P_STANDALONE_VERSION = "1.3.0"
JSDELIVR = f"https://cdn.jsdelivr.net/npm/h5p-standalone@{H5P_STANDALONE_VERSION}/dist"
UNPKG    = f"https://unpkg.com/h5p-standalone@{H5P_STANDALONE_VERSION}/dist"

# ---------------- Helpers ----------------
def read_pdfs(files) -> str:
    out = []
    for f in files or []:
        try:
            reader = PdfReader(f)
            pages = [p.extract_text() or "" for p in reader.pages]
            out.append("\n".join(pages))
        except Exception as e:
            st.warning(f"Could not read {getattr(f,'name','file')}: {e}")
    text = "\n\n".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:12000]

def ask_llm(messages, model="gpt-3.5-turbo", temperature=0.25, max_tokens=800):
    if not openai.api_key:
        st.error("Missing OPENAI_API_KEY in secrets.")
        return ""
    try:
        resp = openai.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return ""

def build_h5p_drag_words(title: str, instructions: str, clozes: List[str]) -> bytes:
    """Build a minimal, valid Drag-the-Words H5P (H5P.DragText 1.8)."""
    def to_dragtext(line: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"*\1*", line)

    content_json = {
        "taskDescription": instructions,
        "textField": "\n".join(to_dragtext(s) for s in clozes),
        "overallFeedback": [{"from": 0, "to": 100, "feedback": "Great job!"}],
        "behaviour": {
            "enableRetry": True,
            "enableSolutionsButton": True,
            "instantFeedback": True,
            "caseSensitive": False
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

def fetch_runtime_bytes() -> dict:
    """
    Fetch h5p-standalone dist files server-side (Python), then cache.
    Avoids client-side CDNs entirely. Returns dict with base64 strings.
    """
    if "runtime_b64" in st.session_state:
        return st.session_state["runtime_b64"]

    urls = [
        (f"{JSDELIVR}/main.bundle.js", "main"),
        (f"{JSDELIVR}/frame.bundle.js","frame"),
        (f"{JSDELIVR}/styles/h5p.css", "css"),
    ]
    fallbacks = [
        (f"{UNPKG}/main.bundle.js","main"),
        (f"{UNPKG}/frame.bundle.js","frame"),
        (f"{UNPKG}/styles/h5p.css","css"),
    ]

    out = {}
    for url, key in urls:
        r = requests.get(url, timeout=30)
        if r.ok and len(r.content) > 10_000:
            out[key] = base64.b64encode(r.content).decode("ascii")
        else:
            # fallback
            fu = next(u for u,k in fallbacks if k==key)
            r2 = requests.get(fu, timeout=30)
            if not (r2.ok and len(r2.content) > 10_000):
                raise RuntimeError(f"Failed to fetch H5P runtime: {key}")
            out[key] = base64.b64encode(r2.content).decode("ascii")

    st.session_state["runtime_b64"] = out
    return out

def render_h5p_inline_from_b64(h5p_b64: str, runtime_b64: dict, height: int = 760):
    """
    Render real H5P inline. We create Blob URLs in the iframe for the runtime JS/CSS
    from the base64 we downloaded server-side. No external network from the browser.
    """
    main_b64  = runtime_b64["main"]
    frame_b64 = runtime_b64["frame"]
    css_b64   = runtime_b64["css"]

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>html,body,#app{{margin:0;height:100%}} .msg{{font:14px system-ui,Segoe UI,Roboto,Helvetica,Arial}}</style>
<script>
  // create Blob URL from base64
  function blobUrl(b64, type) {{
    const bin = atob(b64);
    const len = bin.length;
    const bytes = new Uint8Array(len);
    for (let i=0;i<len;i++) bytes[i] = bin.charCodeAt(i);
    const blob = new Blob([bytes], {{type}});
    return URL.createObjectURL(blob);
  }}
  // Minimal H5PIntegration
  window.H5PIntegration = {{
    baseUrl: location.origin,
    url: location.href,
    siteUrl: location.origin,
    hubIsEnabled: false,
    disableHub: true,
    postUserStatistics: false,
    saveFreq: false,
    l10n: {{ H5P: {{ fullscreen: "Fullscreen", exitFullscreen: "Exit fullscreen" }} }},
    ajax: {{ setFinished: "", contentUserData: "" }},
    libraryUrl: "./",
    core: {{ scripts: [], styles: [] }},
    loadedJs: [], loadedCss: [],
    contents: {{}}
  }};
</script>
</head>
<body>
<div id="app"></div>
<link id="h5pcss" rel="stylesheet">
<script>
  // Inject CSS
  document.getElementById('h5pcss').href = blobUrl("{css_b64}", "text/css");

  // Inject JS in order
  const mainUrl  = blobUrl("{main_b64}",  "application/javascript");
  const frameUrl = blobUrl("{frame_b64}", "application/javascript");

  function loadScript(src) {{
    return new Promise((res, rej) => {{
      const s = document.createElement('script');
      s.src = src;
      s.onload = () => res();
      s.onerror = (e) => rej(e);
      document.body.appendChild(s);
    }});
  }}

  (async function boot(){{
    try {{
      await loadScript(mainUrl);
      await loadScript(frameUrl);
      let HS = window.H5PStandalone || window['h5p-standalone'];
      if (HS && HS.display) {{
        const src = 'data:application/zip;base64,{h5p_b64}';
        HS.display('#app', {{ h5pContent: src }});
      }} else {{
        document.getElementById('app').innerHTML =
          '<div class="msg">H5P API not ready. (Runtime loaded but no display function.)</div>';
      }}
    }} catch(e) {{
      document.getElementById('app').innerHTML =
        '<div class="msg">Failed to load inline H5P runtime: '+(e && e.message || e)+'</div>';
    }}
  }})();
</script>
</body>
</html>
    """.strip()

    st.components.v1.html(html, height=height, scrolling=True)

# ---------------- UI ----------------
st.title("🧬 Cell Bio Tutor — Inline H5P (bundles embedded)")

with st.expander("1) Upload your course slides (PDF)"):
    slides = st.file_uploader("Upload 1–10 PDFs", type=["pdf"], accept_multiple_files=True)

topic = st.text_input("2) Topic to practice (e.g., Electron transport chain, RTK signaling)")
go = st.button("Generate & Render Inline H5P")

if go:
    slide_text = read_pdfs(slides)
    if not slide_text:
        st.error("No slide text detected. Please upload slides (PDF).")
    else:
        prompt = f"""
Using ONLY these slide excerpts, author a concise Drag-the-Words activity for college Cell Biology on: "{topic}".

Return JSON ONLY:
{{
  "title": "short title",
  "instructions": "one clear task line",
  "clozes": [
    "Sentence with **missing** word",
    "Another with **blank**",
    "3rd with **something**",
    "4th with **term**"
  ]
}}

Rules:
- 4–7 clozes max.
- Each **answer** should be 1–2 words that appear in the slides.
- Prefer causal/function statements over trivia.
- Keep sentences short and readable.

SLIDES:
{slide_text}
"""
        out = ask_llm(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": prompt}],
            temperature=0.25, max_tokens=800
        )
        try:
            data = json.loads(out)
        except Exception:
            st.error("Could not parse LLM output. Try a narrower topic.")
            data = None

        if data:
            title = data.get("title") or f"{topic} — Drag the Words"
            instructions = data.get("instructions") or "Fill the missing terms."
            clozes = [c for c in data.get("clozes", []) if "**" in c][:7]

            if not clozes:
                st.error("No valid clozes returned. Try regenerating.")
            else:
                # Build H5P in memory
                h5p_bytes = build_h5p_drag_words(title, instructions, clozes)
                h5p_b64 = base64.b64encode(h5p_bytes).decode("ascii")

                # Fetch runtime files server-side, embed as blob URLs
                try:
                    runtime_b64 = fetch_runtime_bytes()
                except Exception as e:
                    st.error(f"Could not fetch H5P runtime: {e}")
                    runtime_b64 = None

                if runtime_b64:
                    st.success("Generated H5P. Rendering inline below…")
                    render_h5p_inline_from_b64(h5p_b64, runtime_b64, height=760)

                # Optional: also offer download
                st.download_button(
                    "⬇️ Download this H5P",
                    data=h5p_bytes,
                    file_name=re.sub(r'[^a-z0-9]+','_',title.lower()).strip('_') + ".h5p",
                    mime="application/zip",
                )
