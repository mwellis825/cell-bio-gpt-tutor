import os, re, io, json, base64, zipfile, random, textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Cell Bio Tutor — Inline Activities", layout="wide")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit secrets or environment.")
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM = (
    "You are a rigorous, supportive Cell Biology tutor. "
    "Use ONLY the provided slide excerpts (and OpenStax Biology if slides lack coverage) to create concise, causal, critical-thinking practice."
)

# ------------------ HELPERS ------------------
def extract_text_from_pdfs(files) -> str:
    chunks = []
    for f in files or []:
        try:
            reader = PdfReader(f)
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            chunks.append(text)
        except Exception as e:
            st.warning(f"Could not read {getattr(f,'name','(file)')}: {e}")
    full = "\n\n".join(chunks)
    # compact whitespace and trim to a reasonable prompt size
    full = re.sub(r"\s+", " ", full).strip()
    return full[:20000]

def leve_dist(a: str, b: str) -> int:
    """Levenshtein distance (small, for typos)."""
    a, b = a.lower(), b.lower()
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b)+1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insertions = previous[j] + 1
            deletions  = current[j-1] + 1
            subs       = previous[j-1] + (ca != cb)
            current.append(min(insertions, deletions, subs))
        previous = current
    return previous[-1]

def ok_match(student: str, truth: str) -> bool:
    if not student.strip():
        return False
    s, t = student.strip().lower(), truth.strip().lower()
    if s == t:
        return True
    # minor misspellings tolerated (distance ≤ 1), and ignore plural ‘s’
    if s.endswith("s") and s[:-1] == t:
        return True
    if t.endswith("s") and t[:-1] == s:
        return True
    return leve_dist(s, t) <= 1

# ------------------ LLM TEMPLATES ------------------
CRITICAL_CLOZE_INSTRUCTION = """Return JSON ONLY with:
{
  "title": "short title",
  "instructions": "explicit task directions for students",
  "clozes": [
    "Short causal sentence with **answer**",
    "… 4–7 total items"
  ],
  "difficulty": "easy|medium|hard"
}

Rules:
- Use ONLY slide excerpts; if insufficient, you MAY draw on OpenStax Biology to fill small gaps.
- Prefer causal predictions and ‘what happens if…’ reasoning over recall.
- Each **answer** is 1–2 words present in natural course language.
- Keep each sentence short (<= 18 words).
- Do NOT include extra fields or markdown outside **answer** markers.
"""

MATCHING_INSTRUCTION = """Return JSON ONLY with:
{
  "title": "short title",
  "instructions": "explicit task directions for students",
  "pairs": [
    {"left": "short concept phrase", "right": "correct target category"},
    {"left": "...", "right": "..."}
  ],
  "right_choices": ["list", "of", "all", "targets", "unique"],
  "difficulty": "easy|medium|hard"
}

Rules:
- Use ONLY the slide excerpts; if insufficient, you MAY draw on OpenStax Biology to fill small gaps.
- 5–8 pairs total. Make them conceptual (e.g., ‘always-active RTK → increased MAPK output’), not trivia.
- Keep phrases short (<= 6 words).
- right_choices must contain every unique target in the pairs.
"""

def ask_llm_for_cloze(topic: str, slides: str, prior_grade: float) -> Dict[str, Any]:
    # adaptive nudge
    if prior_grade >= 0.8:
        diff_hint = "Focus on harder, multi-step causal effects."
    elif prior_grade <= 0.5:
        diff_hint = "Prefer foundational, single-step causal effects."
    else:
        diff_hint = "Use medium complexity causal effects."

    user = f"""
Topic: "{topic}"
Slides (excerpt; compacted): {slides[:18000]}

Task: Author a concise, causal *fill-in-the-blank* activity (Drag-the-Words style). {diff_hint}
{CRITICAL_CLOZE_INSTRUCTION}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=900,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":user},
        ],
    )
    return json.loads(resp.choices[0].message.content or "{}")

def ask_llm_for_matching(topic: str, slides: str, prior_grade: float) -> Dict[str, Any]:
    if prior_grade >= 0.8:
        diff_hint = "Harder conceptual matches with subtle distractors."
    elif prior_grade <= 0.5:
        diff_hint = "Foundational matches with clear differences."
    else:
        diff_hint = "Medium complexity matches."

    user = f"""
Topic: "{topic}"
Slides (excerpt; compacted): {slides[:18000]}

Task: Author a concise *matching* activity (concept → target/category). {diff_hint}
{MATCHING_INSTRUCTION}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=900,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":user},
        ],
    )
    return json.loads(resp.choices[0].message.content or "{}")

# ------------------ BUILD H5P ZIP (download only) ------------------
def build_h5p_dragtext_zip(title: str, instructions: str, clozes_markdown: List[str]) -> bytes:
    # Convert **answer** → *answer* for H5P DragText
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
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("h5p.json", json.dumps(h5p_json, ensure_ascii=False))
        z.writestr("content/content.json", json.dumps(content_json, ensure_ascii=False))
    return buf.getvalue()

def build_h5p_question_set_zip(title: str, instructions: str, pairs: List[Dict[str,str]]) -> bytes:
    # Represent matching as Multiple Choice within Question Set for portability
    questions = []
    for p in pairs:
        left, right = p["left"], p["right"]
        # build a small MCQ where only right is correct; add 3 distractors sampled from other rights
        all_rights = list({q["right"] for q in pairs})
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
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("h5p.json", json.dumps(h5p_json, ensure_ascii=False))
        z.writestr("content/content.json", json.dumps(qs_content, ensure_ascii=False))
    return buf.getvalue()

# ------------------ INLINE RENDERERS (no H5P) ------------------
def render_inline_cloze(title: str, instructions: str, clozes: List[str], key_prefix: str) -> Tuple[int,int]:
    st.subheader(f"🧠 {title}")
    st.write(instructions)
    total = len(clozes)
    correct = 0
    with st.form(key=f"{key_prefix}_form"):
        answers: Dict[int,str] = {}
        for i, s in enumerate(clozes, start=1):
            # show sentence with ___ and input
            parts = re.split(r"\*\*(.+?)\*\*", s)
            if len(parts) >= 3:
                before, truth, after = parts[0], parts[1], "".join(parts[2:])
            else:
                before, truth, after = s, "", ""
            st.markdown(f"{i}. {before} **____** {after}")
            answers[i] = st.text_input(f"Your answer for {i}", key=f"{key_prefix}_a{i}")
        submitted = st.form_submit_button("Submit answers")
    if submitted:
        for i, s in enumerate(clozes, start=1):
            truth = re.findall(r"\*\*(.+?)\*\*", s)
            truth = truth[0] if truth else ""
            student = st.session_state.get(f"{key_prefix}_a{i}","")
            ok = ok_match(student, truth)
            correct += int(ok)
            st.write(f"{'✅' if ok else '❌'} {i}. Correct: **{truth}** — Yours: _{student or '(blank)'}_")
        st.info(f"Score: {correct}/{total}")
    return correct, total

def render_inline_matching(title: str, instructions: str, pairs: List[Dict[str,str]], right_choices: List[str], key_prefix: str) -> Tuple[int,int]:
    st.subheader(f"🧩 {title}")
    st.write(instructions)
    total = len(pairs)
    correct = 0
    # left column words; each row gets a selectbox of right choices
    with st.form(key=f"{key_prefix}_form"):
        cols = st.columns([2,1.5])
        with cols[0]:
            st.markdown("**Prompt/Concept**")
        with cols[1]:
            st.markdown("**Match**")
        shuffled_rights = list(right_choices)
        for i, p in enumerate(pairs, start=1):
            left, right = p["left"], p["right"]
            with cols[0]:
                st.markdown(f"{i}. {left}")
            with cols[1]:
                st.selectbox("",
                             options=["(choose)"] + shuffled_rights,
                             key=f"{key_prefix}_sel{i}",
                             index=0)
        submitted = st.form_submit_button("Submit matches")
    if submitted:
        for i, p in enumerate(pairs, start=1):
            truth = p["right"]
            student = st.session_state.get(f"{key_prefix}_sel{i}", "(choose)")
            ok = (student == truth)
            correct += int(ok)
            st.write(f"{'✅' if ok else '❌'} {i}. {p['left']} → **{truth}** (yours: _{student}_)")
        st.info(f"Score: {correct}/{total}")
    return correct, total

# ------------------ APP UI ------------------
if "history" not in st.session_state:
    st.session_state.history = []  # [(kind, score, total)]
if "running_grade" not in st.session_state:
    st.session_state.running_grade = 0.7  # start medium

st.title("🧬 Cell Bio Tutor — Inline, Adaptive Activities (H5P download optional)")
with st.expander("1) Upload your course slides (PDF)"):
    slides = st.file_uploader("Upload 1–10 PDFs", type=["pdf"], accept_multiple_files=True)

topic = st.text_input("2) Topic (e.g., 'RTK signaling', 'Electron transport chain')")
colA, colB, colC = st.columns(3)
with colA:
    want_cloze = st.checkbox("Generate Fill-in-the-Blank (critical thinking)", True)
with colB:
    want_match = st.checkbox("Generate Matching (concept → target)", True)
with colC:
    gen = st.button("Generate Activities")

if gen:
    slide_text = extract_text_from_pdfs(slides)
    if not slide_text:
        st.error("No slide text detected. Please upload slides.")
        st.stop()

    grade_hint = st.session_state.running_grade

    # CLOZE
    cloze_data = {}
    if want_cloze:
        cloze_data = ask_llm_for_cloze(topic, slide_text, grade_hint)
        c_title = cloze_data.get("title","Causal fill-in-the-blank")
        c_instr = cloze_data.get("instructions","Fill each blank with the best term from the slides.")
        clozes_raw = cloze_data.get("clozes", [])
        clozes = [s for s in clozes_raw if isinstance(s,str) and "**" in s][:7]
        if not clozes:
            st.warning("Cloze generation returned no valid items.")
        else:
            c_ok, c_total = render_inline_cloze(c_title, c_instr, clozes, key_prefix="cloze")
            if c_total > 0:
                st.session_state.history.append(("cloze", c_ok, c_total))

            # H5P download for the same content
            try:
                h5p_bytes = build_h5p_dragtext_zip(c_title, c_instr, clozes)
                st.download_button("⬇️ Download as H5P (DragText)", data=h5p_bytes,
                                   file_name=re.sub(r'[^a-z0-9]+','_',c_title.lower())+".h5p",
                                   mime="application/zip", key="dl_dragtext")
            except Exception as e:
                st.warning(f"Could not build H5P for cloze: {e}")

    # MATCHING
    match_data = {}
    if want_match:
        match_data = ask_llm_for_matching(topic, slide_text, grade_hint)
        m_title = match_data.get("title","Concept matching")
        m_instr = match_data.get("instructions","Match each prompt to the correct target.")
        pairs = match_data.get("pairs", [])
        right_choices = match_data.get("right_choices", [])
        # sanity filters
        pairs = [p for p in pairs if isinstance(p, dict) and p.get("left") and p.get("right")]
        rights = sorted(list({p["right"] for p in pairs})) or right_choices
        if not pairs or not rights:
            st.warning("Matching generation returned no valid items.")
        else:
            m_ok, m_total = render_inline_matching(m_title, m_instr, pairs, rights, key_prefix="match")
            if m_total > 0:
                st.session_state.history.append(("match", m_ok, m_total))

            # H5P QuestionSet download (MCQ representation of matches)
            try:
                qs_bytes = build_h5p_question_set_zip(m_title, m_instr, pairs)
                st.download_button("⬇️ Download as H5P (QuestionSet/MCQ)", data=qs_bytes,
                                   file_name=re.sub(r'[^a-z0-9]+','_',m_title.lower())+"_qs.h5p",
                                   mime="application/zip", key="dl_qs")
            except Exception as e:
                st.warning(f"Could not build H5P for matching: {e}")

    # Update running grade for adaptivity
    if st.session_state.history:
        total_correct = sum(s for _, s, _ in st.session_state.history)
        total_items = sum(t for _, _, t in st.session_state.history)
        st.session_state.running_grade = total_correct / max(1, total_items)
        st.info(f"Adaptive level updated. Running accuracy: {st.session_state.running_grade:.0%}")
