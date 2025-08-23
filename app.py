# app.py
# Streamlined Streamlit app: Two activities, auto-LO extraction, critical-thinking focus
# -----------------------------------------------------------------------------
# Activities:
#   1) Lenient Fill-in-the-Blank (short-answer, critical thinking prompts)
#   2) Drag-the-Words (word bank -> blanks, click-to-fill simulation)
#
# Instructor uploads or pastes course text (e.g., slide notes). The app
# automatically extracts Learning Objectives and uses them to generate
# activities tailored to the topic. No internet calls; fully local.
# -----------------------------------------------------------------------------

import streamlit as st
import re, uuid, datetime, json

st.set_page_config(page_title="Study Mode: Two Activities", page_icon="📘", layout="wide")

# ------------------------------ Utilities ------------------------------------
def new_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def detect_topic(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["dna replication", "polymerase", "okazaki", "lagging strand", "leading strand", "helicase", "primase"]):
        return "dna_replication"
    if any(k in t for k in ["glycolysis", "tca", "electron transport", "oxidative phosphorylation", "mitochond"]):
        return "metabolism"
    if any(k in t for k in ["actin", "microtubule", "kinesin", "dynein", "focal adhesion"]):
        return "cytoskeleton"
    if any(k in t for k in ["cell cycle", "cyclin", "cdk", "p53", "meiosis", "mitosis"]):
        return "cell_cycle"
    if any(k in t for k in ["rtk", "pi3k", "akt", "mapk", "gpcr", "second messenger"]):
        return "signaling"
    return "generic"

# Extract candidate Learning Objectives from text (no student entry).
# Heuristics: lines beginning with verbs; lines containing 'LO', 'Learning Objective', or bullets.
VERBS = [
    "explain","predict","justify","compare","contrast","distinguish","design","interpret",
    "evaluate","critique","diagnose","infer","propose","calculate","derive","model",
    "identify","classify","analyze","synthesize","outline","formulate","apply","assess"
]
def extract_los(raw_text: str):
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw_text.splitlines()]
    los = []
    for ln in lines:
        low = ln.lower()
        if not ln or len(ln) < 6: 
            continue
        bullet = low.startswith(("-","•","—","*")) or re.match(r"^\d+[\).\]]\s", low)
        mentions_lo = any(k in low for k in ["learning objective", "learning objectives", "lo:", "objective:", "outcome:"])
        starts_verb = any(re.match(rf"^\s*({v})\b", low) for v in VERBS)
        if bullet or mentions_lo or starts_verb:
            # remove bullet markers and LO prefixes
            cleaned = re.sub(r"^(\d+[\).\]]\s*|[-•—*]\s*)", "", ln).strip()
            cleaned = re.sub(r"(?i)^(learning objective(s)?|lo|objective|outcome)[:\-]\s*", "", cleaned).strip()
            if len(cleaned) >= 10:
                los.append(cleaned)
    # Deduplicate while preserving order
    seen = set(); uniq = []
    for x in los:
        key = x.lower()
        if key not in seen:
            uniq.append(x); seen.add(key)
    return uniq[:12]  # cap to keep UI light

# -------------------------- Generators (DNA focus) ----------------------------
# For DNA replication, craft critical-thinking prompts and cloze items.
DNA_KEY_TERMS = [
    "origin of replication","helicase","single-strand binding proteins","topoisomerase",
    "primase","RNA primer","DNA polymerase III","leading strand","lagging strand",
    "Okazaki fragments","DNA polymerase I","RNase H","DNA ligase","proofreading",
    "3'→5' exonuclease","processivity","replication fork","bidirectional replication"
]

def make_fitb_from_lo(lo: str, topic: str):
    # Create a short, critical-thinking prompt requiring a precise concept as the answer.
    # For DNA, bias to a key term; otherwise fallback to a salient noun phrase (last resort).
    answer = None
    if topic == "dna_replication":
        for term in DNA_KEY_TERMS:
            if term in lo.lower():
                answer = term
                break
        if answer is None:
            # pick a plausible answer if not explicitly present
            answer = "DNA ligase"
        prompt = (f"{lo} — In the specific context of DNA replication, which mechanism or factor is the "
                  f"primary constraint here? Provide the **most specific term**.")
    else:
        # generic fallback
        answer = "mechanism"
        prompt = f"{lo} — What is the single most critical mechanism that determines the outcome?"

    return {
        "id": new_id("fitb"),
        "prompt": prompt,
        "answer": answer,
        "accepted": list(set([answer, answer.lower(), answer.title()])),
        "explanation": f"The key idea the LO targets is **{answer}**. Provide targeted reasoning referencing the LO.",
    }

def normalize(s: str):
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’","'")
    return s

def check_fitb(user_text: str, accepted):
    u = normalize(user_text or "")
    acc = [normalize(x) for x in accepted]
    # lenient: substring match or exact
    return any((u == a) or (a in u) for a in acc)

def make_dragwords_from_lo(lo: str, topic: str):
    # Build a cloze sentence with 4 blanks and 6-8 word-bank items.
    if topic != "dna_replication":
        base_terms = ["mechanism","regulator","inhibitor","activator","feedback","rate-limiting"]
        sentence = ("In this scenario, ______ coordinates with ______ to overcome ______; "
                    "failure here elevates error risk unless ______ restores function.")
        answers = ["regulator","activator","rate-limiting","feedback"]
        bank = list(set(answers + base_terms))
    else:
        # Try to map LO to a chain of reasoning for replication
        sentence = ("At the replication fork, ______ unwinds DNA while ______ lays down primers. "
                    "Continuous synthesis occurs on the ______, whereas discontinuous synthesis forms ______ "
                    "that are sealed by ______.")
        answers = ["helicase","primase","leading strand","Okazaki fragments","DNA ligase"]
        # Expand bank with distractors
        bank = list(set(answers + [
            "topoisomerase","DNA polymerase III","lagging strand","RNase H","proofreading"
        ]))

    # Build item
    blanks = [None]*len(answers)
    return {
        "id": new_id("drag"),
        "sentence": sentence,
        "answers": answers,
        "bank": bank
    }

# ------------------------------ UI -------------------------------------------
with st.sidebar:
    st.title("Instructor")
    st.caption("Upload/paste course text. The app auto-extracts **Learning Objectives** and builds two activities.\n"
               "Toggle **Instructor mode** to reveal answers & export.")
    instructor_mode = st.toggle("Instructor mode", True)
    st.markdown("---")
    st.write("**Export**")
    add_ts = st.checkbox("Timestamp filenames", True)

st.title("📘 Study Mode")
st.subheader("Two activities • Auto-LO extraction • Critical thinking")

# Input: paste or upload text (we avoid heavy PDF parsing; paste exported text content recommended)
tabs = st.tabs(["Paste Text", "Upload Text File (.txt)"])
raw_text = ""
with tabs[0]:
    raw_text = st.text_area("Paste course text (include LO bullets if possible):", height=220,
                            placeholder="e.g., Learning Objectives:\n- Explain how helicase and primase coordinate at the replication fork...\n- Predict the effect of a ligase inhibitor on lagging-strand synthesis...\n- Justify how proofreading affects mutation rates under nucleotide scarcity...")
with tabs[1]:
    up = st.file_uploader("Upload a plain text file (.txt) with slide notes/objectives", type=["txt"])
    if up is not None:
        try:
            raw_text = up.read().decode("utf-8", errors="ignore")
        except Exception:
            st.error("Could not read the uploaded file as UTF-8 text.")

gen = st.button("Generate Activities")

if gen:
    if not raw_text or len(raw_text.strip()) < 20:
        st.error("Please paste or upload sufficient course text that includes learning objectives.")
    else:
        topic = detect_topic(raw_text)
        los = extract_los(raw_text)
        if not los:
            st.warning("No clear Learning Objectives detected. Using top sentences as stand-ins.")
            # fallback: use first few long sentences
            cand = re.split(r"[.;]\s+", raw_text)
            los = [c.strip() for c in cand if len(c.strip())>40][:6]

        # Build activities from top 4 LOs
        picked = los[:4]
        fitb_items = [make_fitb_from_lo(lo, topic) for lo in picked[:2]]
        drag_items = [make_dragwords_from_lo(lo, topic) for lo in picked[2:4] or [picked[0]]]

        st.success(f"Detected topic: **{topic.replace('_',' ').title()}** • Extracted {len(los)} learning objectives.")
        st.write("You can proceed directly to student mode, or review below in instructor mode.")

        st.session_state["generated"] = {
            "timestamp": now_str(),
            "topic": topic,
            "los": picked,
            "fitb": fitb_items,
            "drag": drag_items,
        }

data = st.session_state.get("generated")

if data:
    st.markdown("---")
    st.header("Activity 1: Fill in the Blank (Lenient, critical-thinking)")

    for i, item in enumerate(data["fitb"], start=1):
        with st.expander(f"FITB {i}", expanded=True):
            st.write(item["prompt"])
            user = st.text_input("Your answer:", key=item["id"]+"_ans")
            if instructor_mode:
                st.info(f"Accepted answer(s): {', '.join(item['accepted'])}")
                st.caption(item["explanation"])
            else:
                if st.button("Check", key=item["id"]+"_check"):
                    correct = check_fitb(user, item["accepted"])
                    if correct:
                        st.success("Correct (lenient match).")
                    else:
                        st.error("Not quite. Try to be more precise.")

    st.markdown("---")
    st.header("Activity 2: Drag the Words (click-to-fill)")

    for j, d in enumerate(data["drag"], start=1):
        with st.expander(f"Drag-the-Words {j}", expanded=True):
            st.caption("Click words from the bank to fill blanks. Click a filled blank to clear it.")
            # initialize state
            bank_key = d["id"]+"_bank"
            blanks_key = d["id"]+"_blanks"
            if bank_key not in st.session_state:
                st.session_state[bank_key] = list(d["bank"])
            if blanks_key not in st.session_state:
                st.session_state[blanks_key] = [None]*len(d["answers"])

            bank = st.session_state[bank_key]
            blanks = st.session_state[blanks_key]

            # Sentence renderer
            parts = d["sentence"].split("______")
            row = st.container()
            with row:
                # render sentence with clickable blanks
                disp = []
                for idx in range(len(parts)+len(blanks)):
                    if idx % 2 == 0:
                        disp.append(parts[idx//2])
                    else:
                        b_i = (idx-1)//2
                        label = blanks[b_i] if blanks[b_i] else "______"
                        if st.button(label, key=f"{d['id']}_blank_{b_i}"):
                            # clear if already filled
                            if blanks[b_i]:
                                bank.append(blanks[b_i])
                                blanks[b_i] = None
                            st.session_state[blanks_key] = blanks
                st.write("".join(disp))

            st.write("**Word bank:**")
            cols = st.columns(4)
            for idx, w in enumerate(bank):
                col = cols[idx % 4]
                if col.button(w, key=f"{d['id']}_bank_{idx}"):
                    # fill first empty blank
                    for bi in range(len(blanks)):
                        if blanks[bi] is None:
                            blanks[bi] = w
                            break
                    # remove from bank
                    bank.pop(idx)
                    st.session_state[bank_key] = bank
                    st.session_state[blanks_key] = blanks

            # Check answers
            if st.button("Check Sentence", key=d["id"]+"_check"):
                correct = all((blanks[i] or "").lower() == d["answers"][i].lower() for i in range(len(d["answers"])))
                if correct:
                    st.success("All blanks correct!")
                else:
                    # partial feedback
                    good = sum(1 for i in range(len(d["answers"])) if (blanks[i] or "").lower() == d["answers"][i].lower())
                    st.warning(f"{good}/{len(d['answers'])} correct. Keep refining.")
                if instructor_mode:
                    st.info("Answer key: " + ", ".join(d["answers"]))

    # ---------------- Export (Instructor) ----------------
    st.markdown("---")
    if instructor_mode:
        st.subheader("Export Activity Set")
        payload = json.dumps(data, indent=2).encode("utf-8")
        fname = "activities"
        if add_ts:
            fname += "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("⬇️ Download JSON", data=payload, file_name=f"{fname}.json", mime="application/json")
else:
    st.info("To begin, paste or upload your course text, then click **Generate Activities**.")
