
import streamlit as st
import re
from difflib import SequenceMatcher

# ==============
# UI TEXT
# ==============
APP_TITLE = "Let's Practice Biology!"
PROMPT_LABEL = "Enter a topic for review and press generate:"
PLACEHOLDER_TOPIC = "e.g., membranes, protein sorting, glycolysis"

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
    # Each item: {"mode": "fitb" | "drag", ...}
    if "membrane" in t or "membranes" in t:
        return [
            {
                "mode": "fitb",
                "prompt": "At lower temperatures, longer fatty acid tails tend to ______ membrane fluidity.",
                "answers": ["decrease"],
                "slide_ref": "Membranes: fluidity vs tail length",
            },
            {
                "mode": "drag",
                "stem": "Select the best term for each blank: Longer tails (1) ______ fluidity; more unsaturated fatty acids (2) ______ fluidity.",
                "blanks": 2,
                "bank": ["increase", "decrease", "saturate", "shorten"],
                "answers": ["decrease", "increase"],
                "slide_ref": "Membranes: composition & fluidity",
            },
        ]
    if "protein sorting" in t or "target" in t or "er signal" in t:
        return [
            {
                "mode": "fitb",
                "prompt": "Loss of the ER signal sequence typically results in a protein remaining in the ______.",
                "answers": ["cytosol"],
                "slide_ref": "Protein targeting: ER signal",
            },
            {
                "mode": "drag",
                "stem": "Match outcome: No N‑terminal signal (1) → ______; Presence of ER signal (2) → ______.",
                "blanks": 2,
                "bank": ["cytosol", "ER lumen", "lysosome", "nucleus"],
                "answers": ["cytosol", "ER lumen"],
                "slide_ref": "Protein targeting flow",
            },
        ]
    if "glycolysis" in t:
        return [
            {
                "mode": "fitb",
                "prompt": "PFK‑1 catalyzes the committed step of glycolysis by converting F6P to ______.",
                "answers": ["fructose 1,6-bisphosphate", "f1,6bp", "fructose-1,6-bisphosphate"],
                "slide_ref": "Glycolysis regulation",
            },
            {
                "mode": "drag",
                "stem": "Under anaerobic conditions in muscle, pyruvate (1) → ______; purpose is to (2) regenerate ______.",
                "blanks": 2,
                "bank": ["NAD+", "lactate", "acetyl-CoA", "ATP"],
                "answers": ["lactate", "NAD+"],
                "slide_ref": "Fermentation",
            },
        ]
    # default generic
    return [
        {
            "mode": "fitb",
            "prompt": f"State one key concept about {topic} as presented in the slides.",
            "answers": ["answer varies"],
            "slide_ref": f"{topic} core slide",
        }
    ]

# ==============
# App
# ==============
def main():
    st.set_page_config(page_title="Practice App", page_icon="🎓", layout="centered")
    st.title(APP_TITLE)

    topic = st.text_input(PROMPT_LABEL, value=st.session_state.get("topic",""), placeholder=PLACEHOLDER_TOPIC, label_visibility="visible")
    st.session_state["topic"] = topic.strip()

    # Controls: generate new items
    col_a, col_b = st.columns(2)
    gen_fitb = col_a.button("Generate Fill‑in‑the‑Blank")
    gen_drag = col_b.button("Generate Drag & Drop")

    # State holders
    if "current_item" not in st.session_state:
        st.session_state["current_item"] = None
    if "reveal" not in st.session_state:
        st.session_state["reveal"] = False

    # On generate
    if (gen_fitb or gen_drag) and not st.session_state["topic"]:
        st.warning("Please enter a topic first.")
    elif gen_fitb:
        items = [it for it in get_items_for_topic(st.session_state["topic"]) if it["mode"] == "fitb"]
        st.session_state["current_item"] = items[0] if items else None
        st.session_state["reveal"] = False
    elif gen_drag:
        items = [it for it in get_items_for_topic(st.session_state["topic"]) if it["mode"] == "drag"]
        st.session_state["current_item"] = items[0] if items else None
        st.session_state["reveal"] = False

    item = st.session_state["current_item"]
    if item:
        st.divider()
        st.subheader("Question")

        if item["mode"] == "fitb":
            st.write(item["prompt"])
            user = st.text_input("Your answer:", value="", key="fitb_ans", label_visibility="visible")
            c1, c2 = st.columns(2)
            submitted = c1.button("Submit")
            show = c2.button("Show answer")

            if submitted:
                # If "answer varies", we do not grade — keep consistent with tutor rules
                if any(normalize(ans) == "answer varies" for ans in item["answers"]):
                    st.info("Thanks! Ask to reveal the answer if you want to check, or try another question.")
                else:
                    if any(fuzzy_equal(user, ans) for ans in item["answers"]):
                        st.success("Correct!")
                    else:
                        st.info("Good attempt. You can revise or click 'Show answer'.")
            if show:
                st.session_state["reveal"] = True

        elif item["mode"] == "drag":
            st.write(item["stem"])
            # Up to 4 simple options
            bank = item["bank"][:4]
            blanks = item.get("blanks", 2)
            selections = []
            for i in range(blanks):
                selections.append(st.selectbox(f"Blank {i+1}", ["--"] + bank, index=0, key=f"blank_{i}"))
            c1, c2 = st.columns(2)
            submitted = c1.button("Submit")
            show = c2.button("Show answer")

            if submitted:
                correct = True
                for sel, ans in zip(selections, item["answers"]):
                    if normalize(sel) == "--" or not fuzzy_equal(sel, ans):
                        correct = False
                        break
                if correct:
                    st.success("Nice matching!")
                else:
                    st.info("Some blanks need adjustment. Revise or click 'Show answer'.")
            if show:
                st.session_state["reveal"] = True

        # Reveal section only when explicitly requested
        if st.session_state["reveal"]:
            st.divider()
            st.caption(f"Slide reference: {item.get('slide_ref','')}")
            if item["mode"] == "drag":
                st.write("**Answers:** " + ", ".join(item["answers"]))
            else:
                st.write("**Answer:** " + (item["answers"][0] if item["answers"] else ""))
            st.session_state["reveal"] = False  # reset after one reveal

if __name__ == "__main__":
    main()
