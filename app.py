
import streamlit as st
import re
from difflib import SequenceMatcher

# ==============
# Minimal veneer
# ==============
st.set_page_config(page_title="Practice App", page_icon="🎓", layout="centered")

# Hide Streamlit's keyboard hint like "Press Enter to apply"
st.markdown(
    """
    <style>
    div[data-testid="stWidgetLabelHelp"] {display: none !important;}
    /* Also hide form-level submit hints if any */
    div[role="alert"] p:has(span:contains("Press")) {display:none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

TITLE = "Let's Practice Biology!"
PROMPT = "Enter a topic for review and press generate:"
PLACEHOLDER = "e.g., membranes, protein sorting, glycolysis"

st.title(TITLE)

# ==============
# Helpers
# ==============
def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s\-\+]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

SYNONYMS = {
    "increase": {"accumulate", "build up", "rise", "elevate", "grow", "upregulate"},
    "decrease": {"reduce", "drop", "fall", "decline", "downregulate", "lower"},
    "no transcription": {"no initiation", "transcription absent", "no rna synthesis"},
    "initiation": {"start", "beginning", "onset"},
}

def fuzzy_equal(a: str, b: str, threshold: float = 0.86) -> bool:
    a_n, b_n = normalize(a), normalize(b)
    if a_n == b_n:
        return True
    # synonym cross-check
    for key, vals in SYNONYMS.items():
        if (a_n == key and b_n in vals) or (b_n == key and a_n in vals):
            return True
        if a_n in vals and b_n in vals:
            return True
    return SequenceMatcher(None, a_n, b_n).ratio() >= threshold

# =============================
# Content (placeholder examples)
# Replace with slide-driven items
# =============================
def get_items_for_topic(topic: str):
    t = normalize(topic)
    # Each item is either FITB or DRAG
    # We'll return up to one of each for the demo
    if "membrane" in t or "membranes" in t:
        fitb = {
            "mode": "fitb",
            "prompt": "At lower temperatures, longer fatty acid tails tend to ______ membrane fluidity.",
            "answers": ["decrease"],
            "slide_ref": "Membranes: fluidity vs tail length",
        }
        drag = {
            "mode": "drag",
            "stem": "Select the best term for each blank: Longer tails (1) ______ fluidity; more unsaturated fatty acids (2) ______ fluidity.",
            "blanks": 2,
            "bank": ["increase", "decrease", "saturate", "shorten"],
            "answers": ["decrease", "increase"],
            "slide_ref": "Membranes: composition & fluidity",
        }
        return fitb, drag
    if "protein sorting" in t or "target" in t or "er signal" in t:
        fitb = {
            "mode": "fitb",
            "prompt": "Loss of the ER signal sequence typically results in a protein remaining in the ______.",
            "answers": ["cytosol"],
            "slide_ref": "Protein targeting: ER signal",
        }
        drag = {
            "mode": "drag",
            "stem": "Match outcome: No N-terminal signal (1) → ______; Presence of ER signal (2) → ______.",
            "blanks": 2,
            "bank": ["cytosol", "ER lumen", "lysosome", "nucleus"],
            "answers": ["cytosol", "ER lumen"],
            "slide_ref": "Protein targeting flow",
        }
        return fitb, drag
    if "glycolysis" in t:
        fitb = {
            "mode": "fitb",
            "prompt": "PFK-1 catalyzes the committed step of glycolysis by converting F6P to ______.",
            "answers": ["fructose 1,6-bisphosphate", "f1,6bp", "fructose-1,6-bisphosphate"],
            "slide_ref": "Glycolysis regulation",
        }
        drag = {
            "mode": "drag",
            "stem": "Under anaerobic conditions in muscle, pyruvate (1) → ______; purpose is to (2) regenerate ______.",
            "blanks": 2,
            "bank": ["NAD+", "lactate", "acetyl-CoA", "ATP"],
            "answers": ["lactate", "NAD+"],
            "slide_ref": "Fermentation",
        }
        return fitb, drag
    # default generic (still produce both)
    fitb = {
        "mode": "fitb",
        "prompt": f"State one key concept about {topic} as presented in the slides.",
        "answers": ["answer varies"],
        "slide_ref": f"{topic} core slide",
    }
    drag = {
        "mode": "drag",
        "stem": f"Choose terms that best complete these statements about {topic}.",
        "blanks": 2,
        "bank": ["increase", "decrease", "activate", "inhibit"],
        "answers": ["increase", "decrease"],
        "slide_ref": f"{topic} relationships",
    }
    return fitb, drag

# ==============
# State
# ==============
if "topic" not in st.session_state:
    st.session_state["topic"] = ""
if "fitb_item" not in st.session_state:
    st.session_state["fitb_item"] = None
if "drag_item" not in st.session_state:
    st.session_state["drag_item"] = None
if "reveal_fitb" not in st.session_state:
    st.session_state["reveal_fitb"] = False
if "reveal_drag" not in st.session_state:
    st.session_state["reveal_drag"] = False

# ==============
# Input + Generate (single flow)
# ==============
topic = st.text_input(PROMPT, value=st.session_state["topic"], placeholder=PLACEHOLDER, label_visibility="visible")
st.session_state["topic"] = topic.strip()
generate = st.button("Generate")

if generate:
    if not st.session_state["topic"]:
        st.warning("Please enter a topic first.")
    else:
        fitb, drag = get_items_for_topic(st.session_state["topic"])
        st.session_state["fitb_item"] = fitb
        st.session_state["drag_item"] = drag
        st.session_state["reveal_fitb"] = False
        st.session_state["reveal_drag"] = False

# ==============
# Render Activities (both after one generate)
# ==============
fitb = st.session_state["fitb_item"]
drag = st.session_state["drag_item"]

if fitb or drag:
    st.divider()

# --- Fill in the Blank ---
if fitb:
    st.subheader("Fill in the Blank")
    st.write(fitb["prompt"])
    user = st.text_input("Your answer:", value="", key="fitb_ans", label_visibility="visible")
    c1, c2 = st.columns(2)
    submitted = c1.button("Submit FITB")
    show = c2.button("Show answer (FITB)")

    if submitted:
        # If "answer varies", we do not grade
        if any(normalize(ans) == "answer varies" for ans in fitb["answers"]):
            st.info("Thanks! Ask to reveal the answer if you want to check, or try another question.")
        else:
            if any(fuzzy_equal(user, ans) for ans in fitb["answers"]):
                st.success("Correct!")
            else:
                st.info("Good attempt. You can revise or click 'Show answer'.")
    if show:
        st.session_state["reveal_fitb"] = True

    if st.session_state["reveal_fitb"]:
        st.caption(f"Slide reference: {fitb.get('slide_ref','')}")
        st.write("**Answer:** " + (fitb["answers"][0] if fitb["answers"] else ""))
        st.session_state["reveal_fitb"] = False

# --- Drag & Drop (implemented with selects for portability) ---
if drag:
    st.subheader("Drag and Drop")
    st.write(drag["stem"])
    bank = drag["bank"][:4]  # ≤4 terms
    blanks = drag.get("blanks", 2)
    selections = []
    for i in range(blanks):
        selections.append(st.selectbox(f"Blank {i+1}", ["--"] + bank, index=0, key=f"blank_{i}"))
    c3, c4 = st.columns(2)
    submitted_d = c3.button("Submit Drag & Drop")
    show_d = c4.button("Show answer (Drag & Drop)")

    if submitted_d:
        correct = True
        for sel, ans in zip(selections, drag["answers"]):
            if normalize(sel) == "--" or not fuzzy_equal(sel, ans):
                correct = False
                break
        if correct:
            st.success("Nice matching!")
        else:
            st.info("Some blanks need adjustment. Revise or click 'Show answer'.")
    if show_d:
        st.session_state["reveal_drag"] = True

    if st.session_state["reveal_drag"]:
        st.caption(f"Slide reference: {drag.get('slide_ref','')}")
        st.write("**Answers:** " + ", ".join(drag["answers"]))
        st.session_state["reveal_drag"] = False
