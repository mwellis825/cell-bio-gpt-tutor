import os, re, io, json, zipfile, random, glob, string
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# ================== CONFIG ==================
st.set_page_config(page_title="Cell Bio Tutor — Inline DnD + Critical FITB", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in secrets/environment.")
client = OpenAI(api_key=OPENAI_API_KEY)

SLIDES_DIR = Path(__file__).parent / "slides"   # put your PDFs here

SYSTEM = (
    "You are a rigorous, supportive Cell Biology tutor. "
    "Use ONLY the provided slide excerpts (and OpenStax Biology if slides lack coverage) "
    "to create concise, causal, critical-thinking practice. "
    "Never invent placeholders like 'protein X' or 'protein Z'. "
    "Use concrete names (e.g., ATP synthase, Complex I). "
    "Write in plain language; if jargon appears, add a brief parenthetical clarifier."
)

# ================== UTILITIES ==================
@st.cache_data(show_spinner=False)
def extract_text_from_repo_slides() -> str:
    pdf_paths = sorted(glob.glob(str(SLIDES_DIR / "*.pdf")))
    chunks = []
    for path in pdf_paths:
        try:
            with open(path, "rb") as f:
                r = PdfReader(f)
                text = "\n".join((p.extract_text() or "") for p in r.pages)
                chunks.append(text)
        except Exception as e:
            st.warning(f"Could not read {Path(path).name}: {e}")
    full = re.sub(r"\s+", " ", "\n\n".join(chunks)).strip()
    return full[:24000]

def levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = prev[j] + 1
            dele = curr[j-1] + 1
            sub = prev[j-1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]

_PUNCT = str.maketrans("", "", string.punctuation + "–—-")
STOP = {"the","a","an","to","of","in","on","for","and","or"}
SYN = {
    "oxygen consumption":"oxygen use",
    "oxygen use":"oxygen consumption",
    "o2 consumption":"oxygen consumption",
    "o2 use":"oxygen consumption",
    "atp production":"atp synthesis",
    "atp synthesis":"atp production",
    "proton gradient":"pmf",
    "pmf":"proton gradient",
    "electron transport chain":"etc",
    "etc":"electron transport chain",
    "activation":"increased activity",
    "increased activity":"activation",
    "inhibition":"decreased activity",
    "decreased activity":"inhibition",
}

def normalize(s: str) -> str:
    return (s or "").strip().lower().translate(_PUNCT)

def token_set(s: str) -> set:
    return {w for w in normalize(s).split() if w and w not in STOP}

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))

def ok_match(student: str, truth: str) -> bool:
    """Lenient match: case/punct-insensitive, minor typos, synonyms, and token overlap."""
    if not isinstance(student, str) or not isinstance(truth, str):
        return False
    s, t = normalize(student), normalize(truth)
    if not s or not t: return False
    if s == t: return True

    # synonyms mapping (both ways)
    if SYN.get(s) == t or SYN.get(t) == s: return True

    # substring acceptance for short targets
    if len(t) <= 10 and (s in t or t in s): return True

    # edit distance threshold scales with length
    dist = levenshtein(s, t)
    thr = 1 if max(len(s), len(t)) <= 6 else 2
    if dist <= thr: return True

    # token overlap
    js = jaccard(token_set(s), token_set(t))
    return js >= 0.6

# ================== LLM PROMPTS ==================
CRITICAL_CLOZE_INSTRUCTION = """Return JSON ONLY with:
{
  "title": "short title",
  "instructions": "explicit task directions for students",
  "clozes": [
    "Short causal sentence with **answer**",
    "… 4–6 total items"
  ],
  "difficulty": "easy|medium|hard"
}

Rules:
- Use ONLY slide excerpts; if insufficient, you MAY draw on OpenStax Biology to fill small gaps.
- Prefer approachable causal predictions students can reason about from class:
  e.g., “If Complex IV is inhibited, **oxygen consumption** decreases.”
- Avoid obscure subunits, rare cofactors, or exact numbers.
- Each **answer** is 1–2 common course words (e.g., “ATP production”, “proton gradient”).
- Keep each sentence short (≤ 16 words).
- Use plain language with brief clarifiers (e.g., “cathode (negative end)”).
- Do NOT include extra fields or markdown outside **answer** markers.
- Never invent placeholders like “protein X/Z”. Use concrete names from slides or OpenStax only.
"""

MATCHING_INSTRUCTION = """Return JSON ONLY with:
{
  "title": "short title",
  "instructions": "explicit task directions for students",
  "pairs": [
    {"left": "short causal prompt", "right": "target/category"},
    {"left": "...", "right": "..."}
  ],
  "right_choices": ["all", "unique", "targets"],
  "difficulty": "easy|medium|hard"
}

Rules:
- Use ONLY the slide excerpts; if insufficient, you MAY draw on OpenStax Biology to fill small gaps.
- 6–8 pairs total. Conceptual, not trivia. Example: “RTK always-active → MAPK output increases”.
- Keep phrases short (≤ 6 words) and precise.
- Use plain language with brief clarifiers when needed.
- right_choices must contain every unique target in the pairs.
- Never invent placeholders like “protein X/Z”. Use concrete names from slides or OpenStax only.
"""

@st.cache_data(show_spinner=False)
def ask_llm_for_cloze_cached(topic: str, slides: str, grade_bucket: str) -> Dict[str, Any]:
    diff_hint = {
        "low":  "Focus on approachable, single- or two-step causal effects.",
        "mid":  "Use medium complexity causal effects with two steps.",
        "high": "Use harder multi-step causal effects, but stay in plain language.",
    }[grade_bucket]
    user = f"""
Topic: "{topic}"
Slides (excerpt; compacted): {slides[:18000]}

Task: Author a concise, causal fill-in-the-blank activity (Drag-the-Words style). {diff_hint}
{CRITICAL_CLOZE_INSTRUCTION}
""".strip()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.15,
        max_tokens=800,
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
    )
    return json.loads(resp.choices[0].message.content or "{}")

@st.cache_data(show_spinner=False)
def ask_llm_for_matching_cached(topic: str, slides: str, grade_bucket: str) -> Dict[str, Any]:
    diff_hint = {
        "low":  "Foundational matches with clear differences.",
        "mid":  "Medium complexity matches with subtle distractors.",
        "high": "Harder conceptual matches (multi-step), in plain language.",
    }[grade_bucket]
    user = f"""
Topic: "{topic}"
Slides (excerpt; compacted): {slides[:18000]}

Task: Author a concise drag-and-drop matching activity (concept → correct target). {diff_hint}
{MATCHING_INSTRUCTION}
""".strip()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.15,
        max_tokens=800,
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
    )
    return json.loads(resp.choices[0].message.content or "{}")

# ================== H5P (optional downloads) ==================
def build_h5p_dragtext_zip(title: str, instructions: str, clozes_markdown: List[str]) -> bytes:
    text_lines = [re.sub(r"\*\*(.+?)\*\*", r"*\1*", s) for s in clozes_markdown]
    content_json = {
        "taskDescription": instructions,
        "textField": "\n".join(text_lines),
        "overallFeedback": [{"from": 0, "to": 100, "feedback": "Nice work!"}],
        "behaviour": {"enableRetry": True, "enableSolutionsButton": True, "instantFeedback": True, "caseSensitive": False}
    }
    h5p_json = {
        "title": title,
        "language": "en",
        "mainLibrary": "H5P.DragText",
        "embedTypes": ["div"],
        "preloadedDependencies": [
            {"machineName":"H5P.DragText","majorVersion":1,"minorVersion":8},
            {"machineName":"H5P.JoubelUI","majorVersion":1,"minorVersion":3},
            {"machineName":"H5P.Transition","majorVersion":1,"minorVersion":0},
        ],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("h5p.json", json.dumps(h5p_json, ensure_ascii=False))
        z.writestr("content/content.json", json.dumps(content_json, ensure_ascii=False))
    return buf.getvalue()

def build_h5p_question_set_zip(title: str, instructions: str, pairs: List[Dict[str,str]]) -> bytes:
    questions = []
    all_rights = sorted(list({p["right"] for p in pairs}))
    for p in pairs:
        left, right = p["left"], p["right"]
        distractors = [r for r in all_rights if r != right]
        random.shuffle(distractors)
        opts = [right] + distractors[:3]
        random.shuffle(opts)
        questions.append({
            "params": {
                "question": f"{instructions}<br><b>{left}</b>",
                "answers": [{"text": o, "correct": o==right, "feedback": ""} for o in opts],
                "behaviour": {"enableRetry": True, "showSolutionsButton": True}
            },
            "library": "H5P.MultiChoice 1.14"
        })
    qs_content = {
        "introPage": {"showIntroPage": False, "startButtonText": "Start"},
        "title": title,
        "passPercentage": 0,
        "behaviour": {"enableRetry": True, "enableSolutionsButton": True},
        "texts": {"finishButton": "Finish"},
        "questionSets": [],
        "questions": questions
    }
    h5p_json = {
        "title": title,
        "language": "en",
        "mainLibrary": "H5P.QuestionSet",
        "embedTypes": ["div"],
        "preloadedDependencies": [
            {"machineName":"H5P.QuestionSet","majorVersion":1,"minorVersion":17},
            {"machineName":"H5P.MultiChoice","majorVersion":1,"minorVersion":14},
            {"machineName":"H5P.JoubelUI","majorVersion":1,"minorVersion":3},
        ],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("h5p.json", json.dumps(h5p_json, ensure_ascii=False))
        z.writestr("content/content.json", json.dumps(qs_content, ensure_ascii=False))
    return buf.getvalue()

# ================== INLINE RENDERERS ==================
def render_inline_cloze(title: str, instructions: str, clozes: List[str], ns: str = "fitb") -> Tuple[int,int]:
    state = st.session_state.setdefault(ns, {"answers": {}, "checked": {}, "score": (0, len(clozes))})
    st.subheader(f"🧠 {title}")
    st.write(instructions)

    for i, s in enumerate(clozes, start=1):
        parts = re.split(r"\*\*(.+?)\*\*", s)
        if len(parts) >= 3:
            before, truth, after = parts[0], parts[1], "".join(parts[2:])
        else:
            before, truth, after = s, "", ""
        st.markdown(f"{i}. {before} **____** {after}")

        k_ans = f"{ns}_ans_{i}"
        k_chk = f"{ns}_chk_{i}"

        default_val = state["answers"].get(i, "")
        state["answers"][i] = st.text_input("Your answer:", value=default_val, key=k_ans, label_visibility="collapsed")

        cols = st.columns([1,1,6])
        with cols[0]:
            if st.button("Check", key=k_chk):
                student = state["answers"].get(i, "")
                ok = ok_match(student, truth)
                state["checked"][i] = (ok, truth)
        with cols[1]:
            if i in state["checked"]:
                ok, _ = state["checked"][i]
                st.markdown("✅" if ok else "❌")

        if i in state["checked"]:
            ok, tshow = state["checked"][i]
            st.caption(f"Correct: **{tshow or '(no key)'}** — Yours: _{state['answers'].get(i) or '(blank)'}_")

        st.divider()

    if st.button("Check all", key=f"{ns}_check_all"):
        n_ok = 0
        for i, s in enumerate(clozes, start=1):
            truths = re.findall(r"\*\*(.+?)\*\*", s)
            truth = truths[0] if truths else ""
            student = state["answers"].get(i, "")
            ok = ok_match(student, truth)
            state["checked"][i] = (ok, truth)
            n_ok += int(ok)
        state["score"] = (n_ok, len(clozes))

    n_ok, total = state["score"]
    if state["checked"]:
        st.info(f"Score: {n_ok}/{total}")
    return n_ok, total

def render_inline_dragdrop(title: str, instructions: str, pairs: List[Dict[str,str]], right_choices: List[str], height: int = 560):
    rights = sorted(list({p["right"] for p in pairs})) or right_choices
    data = {"title": title, "instructions": instructions, "pairs": pairs, "rights": rights}
    payload = json.dumps(data)

    # IMPORTANT: escape all `${...}` inside f-string -> use ${{...}}
    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, sans-serif; margin:0; padding:12px; }}
  .wrap {{ display:flex; gap:16px; align-items:flex-start; }}
  .col {{ flex:1; min-width: 260px; }}
  h2 {{ margin: 0 0 6px 0; font-size: 18px; }}
  .instructions {{ margin: 4px 0 12px 0; color:#333; font-size: 14px; }}
  .bank, .zone {{ border:1px solid #d0d7de; border-radius:10px; padding:8px; background:#fff; }}
  .bank {{ min-height: 64px; display:flex; flex-wrap:wrap; gap:8px; align-content:flex-start; }}
  .pill {{
    display:inline-block; padding:4px 8px; border-radius:999px;
    border:1px solid #c7c7c7; background:#f7f7f8; cursor:grab;
    font-size:14px; line-height:1.2; white-space:nowrap; user-select:none;
  }}
  .pill:active {{ cursor:grabbing; }}
  .zones {{ display:grid; grid-template-columns: repeat(auto-fill,minmax(220px,1fr)); gap:10px; }}
  .zone {{ min-height: 60px; display:flex; flex-direction:column; gap:6px; }}
  .zlabel {{ font-size:13px; color:#222; text-align:center; }}
  .dropbox {{
    min-height: 38px; display:flex; flex-wrap:wrap; justify-content:center; align-items:center;
    gap:6px; border:1px dashed #c7c7c7; border-radius:8px; padding:6px; background:#fafafa;
  }}
  .over {{ background:#eef7ff; border-color:#8bb6ff; }}
  .ok {{ border-color:#3fb950; background:#eaffea; }}
  .bad {{ border-color:#f85149; background:#fff0f0; }}
  .buttons {{ margin-top: 12px; display:flex; gap:8px; }}
  .btn {{
    appearance:none; border:1px solid #d0d7de; background:#fff; padding:6px 10px; border-radius:8px; cursor:pointer;
    font-size:14px;
  }}
  .btn:hover {{ background:#f3f4f6; }}
  .score {{ margin-top:8px; font-size:14px; }}
</style>
</head>
<body>
  <h2>🧩 {title}</h2>
  <div class="instructions">{instructions}</div>
  <div class="wrap">
    <div class="col">
      <div class="zlabel"><b>Draggables</b></div>
      <div id="bank" class="bank" aria-label="Draggables bank"></div>
      <div class="buttons">
        <button id="reset" class="btn">Reset</button>
        <button id="check" class="btn">Check answers</button>
      </div>
      <div id="score" class="score"></div>
    </div>
    <div class="col">
      <div class="zlabel"><b>Drop zones</b></div>
      <div id="zones" class="zones"></div>
    </div>
  </div>

  <script id="data" type="application/json">{payload}</script>
  <script>
    const data = JSON.parse(document.getElementById('data').textContent);
    const pairs = Array.isArray(data.pairs) ? data.pairs : [];
    const rights = Array.isArray(data.rights) ? data.rights : [];

    const bank = document.getElementById('bank');
    const zones = document.getElementById('zones');
    const scoreBox = document.getElementById('score');

    pairs.forEach((p, i) => {{
      if (!p || !p.left || !p.right) return;
      const pill = document.createElement('div');
      pill.className = 'pill';
      pill.textContent = p.left;
      pill.setAttribute('draggable', 'true');
      pill.id = 'drag_' + i;
      pill.dataset.answer = p.right;
      pill.addEventListener('dragstart', ev => {{
        ev.dataTransfer.setData('text/plain', pill.id);
      }});
      bank.appendChild(pill);
    }});

    rights.forEach((r) => {{
      if (!r) return;
      const z = document.createElement('div');
      z.className = 'zone';
      z.dataset.target = r;

      const lab = document.createElement('div');
      lab.className = 'zlabel';
      lab.textContent = r;

      const box = document.createElement('div');
      box.className = 'dropbox';
      box.addEventListener('dragover', ev => {{ ev.preventDefault(); box.classList.add('over'); }});
      box.addEventListener('dragleave', () => box.classList.remove('over'));
      box.addEventListener('drop', ev => {{
        ev.preventDefault(); box.classList.remove('over');
        const id = ev.dataTransfer.getData('text/plain');
        const el = document.getElementById(id);
        if (el) {{
          el.classList.remove('ok','bad');
          box.appendChild(el);
        }}
      }});

      z.appendChild(lab);
      z.appendChild(box);
      zones.appendChild(z);
    }});

    bank.addEventListener('dragover', ev => {{ ev.preventDefault(); bank.classList.add('over'); }});
    bank.addEventListener('dragleave', () => bank.classList.remove('over'));
    bank.addEventListener('drop', ev => {{
      ev.preventDefault(); bank.classList.remove('over');
      const id = ev.dataTransfer.getData('text/plain');
      const el = document.getElementById(id);
      if (el) {{ el.classList.remove('ok','bad'); bank.appendChild(el); }}
    }});

    document.getElementById('reset').addEventListener('click', () => {{
      scoreBox.textContent = '';
      document.querySelectorAll('.pill').forEach(el => bank.appendChild(el));
      document.querySelectorAll('.pill').forEach(el => el.classList.remove('ok','bad'));
    }});

    document.getElementById('check').addEventListener('click', () => {{
      let score = 0, total = pairs.length;
      document.querySelectorAll('.pill').forEach(el => el.classList.remove('ok','bad'));
      document.querySelectorAll('.zone').forEach(zone => {{
        const target = zone.dataset.target;
        zone.querySelectorAll('.pill').forEach(pill => {{
          const ok = (pill.dataset.answer === target);
          pill.classList.add(ok ? 'ok' : 'bad');
          if (ok) score += 1;
        }});
      }});
      // Escape ${...} in f-string: use ${{...}}
      scoreBox.textContent = `Score: ${{score}}/${{total}}`;
    }});
  </script>
</body>
</html>
    """.strip()

    st.components.v1.html(html, height=height, scrolling=True)

# ================== APP STATE & UI ==================
if "running_grade" not in st.session_state:
    st.session_state.running_grade = 0.7
if "fitb_state" not in st.session_state:
    st.session_state.fitb_state = None
if "dnd_state" not in st.session_state:
    st.session_state.dnd_state = None

st.title("🧬 Cell Bio Tutor — Inline DnD + Critical FITB (fast, lenient)")

slides_ok = SLIDES_DIR.exists() and any(Path(p).suffix.lower()==".pdf" for p in glob.glob(str(SLIDES_DIR / "*.pdf")))
if not slides_ok:
    st.warning("No PDFs found in ./slides. Add your lecture PDFs to a 'slides' folder in the repo.")
slide_text = extract_text_from_repo_slides() if slides_ok else ""

topic = st.text_input("Topic (e.g., 'RTK signaling', 'Electron transport chain')", value="")
colA, colB, colC = st.columns(3)
with colA:
    want_cloze = st.checkbox("Generate Critical FITB", True)
with colB:
    want_dnd = st.checkbox("Generate True Drag-and-Drop", True)
with colC:
    if st.button("Generate / Refresh Activities"):
        if not slide_text:
            st.error("No slide text detected in ./slides.")
        else:
            g = st.session_state.running_grade
            bucket = "low" if g <= 0.6 else "mid" if g <= 0.85 else "high"

            if want_cloze:
                cdata = ask_llm_for_cloze_cached(topic, slide_text, bucket)
                cl_raw = cdata.get("clozes", []) or []
                clozes = []
                for s in cl_raw:
                    if isinstance(s, str) and "**" in s and not re.search(r"\bprotein [xz]\b", s, re.I):
                        clozes.append(s if len(s) <= 140 else s[:140].rstrip(". ") + "…")
                clozes = clozes[:6]
                st.session_state.fitb_state = {
                    "title": cdata.get("title","Causal fill-in-the-blank"),
                    "instructions": cdata.get("instructions","Predict the missing term in plain language (with brief clarifiers)."),
                    "clozes": clozes
                }
            else:
                st.session_state.fitb_state = None

            if want_dnd:
                mdata = ask_llm_for_matching_cached(topic, slide_text, bucket)
                pairs_raw = mdata.get("pairs", []) or []
                cleaned = []
                for p in pairs_raw:
                    if isinstance(p, dict) and p.get("left") and p.get("right"):
                        if re.search(r"\bprotein [xz]\b", p["left"], re.I) or re.search(r"\bprotein [xz]\b", p["right"], re.I):
                            continue
                        left = p["left"][:40] + ("…" if len(p["left"])>40 else "")
                        right = p["right"][:40] + ("…" if len(p["right"])>40 else "")
                        cleaned.append({"left": left, "right": right})
                pairs = cleaned[:8]
                rights = sorted(list({p["right"] for p in pairs})) or mdata.get("right_choices", [])
                st.session_state.dnd_state = {
                    "title": mdata.get("title","Concept → Target (Drag-and-Drop)"),
                    "instructions": mdata.get("instructions","Drag each prompt to the correct target category (plain language with clarifiers)."),
                    "pairs": pairs,
                    "rights": rights
                }
            else:
                st.session_state.dnd_state = None

# --------- RENDER FITB (from state) ----------
if st.session_state.fitb_state:
    fs = st.session_state.fitb_state
    if fs["clozes"]:
        n_ok, total = render_inline_cloze(fs["title"], fs["instructions"], fs["clozes"], ns="fitb_ns")
        if total > 0:
            st.session_state.running_grade = max(0.0, min(1.0, n_ok/total))
        try:
            h5p_bytes = build_h5p_dragtext_zip(fs["title"], fs["instructions"], fs["clozes"])
            st.download_button("⬇️ Download FITB as H5P (DragText)", data=h5p_bytes,
                               file_name=re.sub(r'[^a-z0-9]+','_',fs["title"].lower())+".h5p",
                               mime="application/zip", key="dl_dragtext_fitb_ok")
        except Exception as e:
            st.warning(f"Could not build H5P for FITB: {e}")
    else:
        st.warning("No valid FITB items were generated for that topic.")

# --------- RENDER DnD (from state) ----------
if st.session_state.dnd_state:
    ds = st.session_state.dnd_state
    if ds["pairs"] and (ds["rights"] or {p["right"] for p in ds["pairs"]}):
        render_inline_dragdrop(ds["title"], ds["instructions"], ds["pairs"], ds["rights"], height=560)
        try:
            qs_bytes = build_h5p_question_set_zip(ds["title"], ds["instructions"], ds["pairs"])
            st.download_button("⬇️ Download DnD as H5P (QuestionSet)", data=qs_bytes,
                               file_name=re.sub(r'[^a-z0-9]+','_',ds["title"].lower())+"_qs.h5p",
                               mime="application/zip", key="dl_qs_dnd_ok")
        except Exception as e:
            st.warning(f"Could not build H5P for DnD: {e}")
    else:
        st.warning("No valid DnD items were generated for that topic.")
