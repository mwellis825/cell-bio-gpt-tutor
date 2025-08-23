# app.py
# Streamlit prompt-based critical thinking question generator
# -----------------------------------------------------------------------------
# IMPORTANT: This app is designed for instructors. It *never* exposes or uses
# your private exam questions. It only uses in-code templates + your typed prompts.
#
# How to run:
#   1) pip install streamlit
#   2) streamlit run app.py
#
# What it does:
#   - Accept a topic prompt and learning objectives
#   - Generate original, exam-style critical thinking questions (MCQ, open, diagram)
#   - Hide answers in "Student preview" mode
#   - Export as JSON/CSV for easy reuse (Canvas/H5P prep)
#
# No internet calls, no external models.
# -----------------------------------------------------------------------------

import streamlit as st
import random, uuid, json, csv, io, datetime, re
from typing import List, Dict, Any

# --------------------------- App Config --------------------------------------
st.set_page_config(page_title="Critical Thinking Question Generator", page_icon="🧠", layout="wide")

# --------------------------- Helpers -----------------------------------------
def new_id(prefix: str = "q") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def sanitize_lo_text(text: str) -> List[str]:
    # Split by lines, strip, drop empties
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    return [x for x in lines if x]

def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

def pick_bloom_level() -> str:
    return random.choice(["apply", "analyze", "evaluate", "create"])

def choice_shuffle(choices: List[str], answer_index: int):
    order = list(range(len(choices)))
    random.shuffle(order)
    new_choices = [choices[i] for i in order]
    new_answer_index = order.index(answer_index)
    return new_choices, new_answer_index

def export_json(questions: List[Dict[str, Any]]) -> bytes:
    return json.dumps({"generated_at": now_iso(), "questions": questions}, indent=2).encode("utf-8")

def export_csv(questions: List[Dict[str, Any]]) -> bytes:
    # Flatten to CSV; choices joined with " || "
    headers = ["id", "type", "prompt", "choices", "answer_index", "answer_text", "rationale", "los", "cognitive_level", "tags"]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    for q in questions:
        choices = " || ".join(q.get("choices", [])) if q.get("choices") else ""
        answer_idx = q.get("answer_index", "")
        answer_text = ""
        if q.get("type") == "mcq" and q.get("choices") and isinstance(answer_idx, int) and 0 <= answer_idx < len(q["choices"]):
            answer_text = q["choices"][answer_idx]
        row = [
            q.get("id",""),
            q.get("type",""),
            q.get("prompt","").replace("\n"," ").strip(),
            choices,
            answer_idx,
            answer_text,
            q.get("rationale","").replace("\n"," ").strip(),
            " | ".join(q.get("los", [])),
            q.get("cognitive_level",""),
            " | ".join(q.get("tags", [])),
        ]
        writer.writerow(row)
    return output.getvalue().encode("utf-8")

# --------------------------- Template Engines --------------------------------
# Each module returns lists of questions (dicts). We mix MCQ / open / diagram.
def mk_mcq(prompt: str, choices: List[str], correct_idx: int, rationale: str, los: List[str], tags: List[str]) -> Dict[str, Any]:
    c, ci = choice_shuffle(choices, correct_idx)
    return {
        "id": new_id("mcq"),
        "type": "mcq",
        "prompt": prompt.strip(),
        "choices": c,
        "answer_index": ci,
        "rationale": rationale.strip(),
        "los": los,
        "cognitive_level": pick_bloom_level(),
        "tags": tags,
    }

def mk_open(prompt: str, rubric: str, los: List[str], tags: List[str]) -> Dict[str, Any]:
    return {
        "id": new_id("open"),
        "type": "open",
        "prompt": prompt.strip(),
        "rationale": rubric.strip(),  # Store suggested rubric in rationale field
        "los": los,
        "cognitive_level": pick_bloom_level(),
        "tags": tags,
    }

def mk_diagram(prompt: str, rubric: str, los: List[str], tags: List[str]) -> Dict[str, Any]:
    return {
        "id": new_id("diagram"),
        "type": "diagram",
        "prompt": prompt.strip(),
        "rationale": rubric.strip(),
        "los": los,
        "cognitive_level": pick_bloom_level(),
        "tags": tags,
    }

# --------------------------- Module: Metabolism -------------------------------
def module_metabolism(lo: List[str], topic: str, n_mcq: int, n_open: int, n_diag: int) -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    tags = ["metabolism","mitochondria","glycolysis","TCA","ETC","OxPhos"]

    mcq_templates = [
        lambda: mk_mcq(
            prompt=("A cell line shows normal glycolysis but low ATP output despite abundant O₂. "
                    "Pyruvate accumulates in the cytosol. Which defect best explains this?"),
            choices=[
                "Phosphofructokinase-1 is inactive, stopping glycolysis entirely",
                "The mitochondrial pyruvate carrier is defective",
                "Complex II cannot oxidize FADH₂",
                "ATP synthase is hyperactive"
            ],
            correct_idx=1,
            rationale=("Cytosolic pyruvate accumulation with normal glycolysis points to failed mitochondrial import; "
                       "a defective pyruvate carrier prevents TCA entry and downstream OxPhos."),
            los=lo, tags=tags),
        lambda: mk_mcq(
            prompt=("A drug raises the redox potential of Complex I above that of O₂. Predict the most direct impact."),
            choices=[
                "NADH electrons still transfer normally to coenzyme Q",
                "Electron flow accelerates through the ETC",
                "NADH electrons cannot pass forward at Complex I, reducing proton pumping",
                "FADH₂ oxidation at Complex II is also blocked"
            ],
            correct_idx=2,
            rationale=("If Complex I has a higher redox potential than downstream carriers, "
                       "it cannot donate electrons efficiently; proton pumping from Complex I falls."),
            los=lo, tags=tags),
        lambda: mk_mcq(
            prompt=("A student replaces most carbohydrates with fats; glycolysis rate declines. "
                    "What happens to oxidative phosphorylation?"),
            choices=[
                "Stops entirely because acetyl‑CoA cannot be made without glycolysis",
                "Continues because β‑oxidation supplies acetyl‑CoA to the TCA cycle",
                "Continues only if glycolysis supplies NADH",
                "Slows because fats bypass ATP synthase"
            ],
            correct_idx=1,
            rationale=("β‑oxidation yields acetyl‑CoA that feeds the TCA cycle; ETC/OxPhos can continue using reducing equivalents from fat metabolism."),
            los=lo, tags=tags),
        lambda: mk_mcq(
            prompt=("A protonophore (H⁺ ionophore) is added to cells. Which outcome is MOST likely?"),
            choices=[
                "ATP production increases because the gradient forms faster",
                "Heat production increases as the proton gradient is dissipated",
                "Electron flow stops immediately at Complex IV",
                "Glycolysis halts due to excess NAD⁺"
            ],
            correct_idx=1,
            rationale=("Uncouplers collapse ΔpH/Δψ; energy of the gradient is released as heat, reducing ATP synthesis via ATP synthase."),
            los=lo, tags=tags),
        lambda: mk_mcq(
            prompt=("Under low ATP and high O₂, what happens to phosphofructokinase‑1 (PFK‑1) activity and why?"),
            choices=[
                "PFK‑1 increases to produce lactate as an alternative energy source",
                "PFK‑1 increases to keep feeding OxPhos with intermediates",
                "PFK‑1 decreases because the ATP‑consuming step is restrained when ATP is limited",
                "PFK‑1 decreases because it is the TCA cycle rate‑limiter"
            ],
            correct_idx=2,
            rationale=("PFK‑1 consumes ATP; when ATP is scarce, PFK‑1 activity is curtailed to conserve ATP, even if O₂ is abundant."),
            los=lo, tags=tags),
    ]

    open_templates = [
        lambda: mk_open(
            prompt=("You disrupt Complex IV completely. Predict effects on: (a) O₂ consumption, "
                    "(b) proton gradient magnitude, and (c) ATP yield; justify mechanistically."),
            rubric=("O₂ consumption falls to ~0 as it is the terminal acceptor at Complex IV; "
                    "proton gradient collapses without continued pumping; ATP via OxPhos drops sharply; "
                    "cells may rely more on glycolysis/fermentation."),
            los=lo, tags=tags),
        lambda: mk_open(
            prompt=("Cells are supplied abundant glucose, but NAD⁺ regeneration is blocked. "
                    "Explain what happens to glycolysis and ATP yield. How would enhancing lactate "
                    "dehydrogenase activity change the outcome?"),
            rubric=("Without NAD⁺ recycling, GAPDH stalls → glycolysis slows/stops → ATP yield falls. "
                    "LDH restores NAD⁺ by reducing pyruvate to lactate, permitting glycolysis to continue anaerobically."),
            los=lo, tags=tags),
        lambda: mk_open(
            prompt=("Complex I is mislocalized to the cytosol due to loss of its mitochondrial targeting sequence. "
                    "Predict consequences for electron flow and ATP synthesis."),
            rubric=("Mislocalized Complex I cannot contribute to the ETC; fewer protons pumped; Δp declines; "
                    "ATP synthase output falls; upstream NADH accumulates; potential metabolic rerouting."),
            los=lo, tags=tags),
    ]

    diagram_templates = [
        lambda: mk_diagram(
            prompt=("On a mitochondrion diagram, (1) indicate where protons accumulate during normal ETC, "
                    "(2) redraw distribution if ATP synthase is inhibited, and (3) predict NADH oxidation rate."),
            rubric=("Normal: H⁺ accumulates in the intermembrane space; inhibition increases gradient back‑pressure → "
                    "ETC slows; NADH oxidation rate decreases due to elevated proton motive force opposing pumping."),
            los=lo, tags=tags),
        lambda: mk_diagram(
            prompt=("Sketch how a protonophore affects the proton gradient across the inner membrane "
                    "and annotate expected changes in heat vs. ATP output."),
            rubric=("Gradient is dissipated (similar [H⁺] across membrane); ATP output drops; heat output rises due to uncoupling."),
            los=lo, tags=tags),
    ]

    # Sample seeding quantity
    for _ in range(min(n_mcq, len(mcq_templates))):
        qs.append(random.choice(mcq_templates)())
    for _ in range(min(n_open, len(open_templates))):
        qs.append(random.choice(open_templates)())
    for _ in range(min(n_diag, len(diagram_templates))):
        qs.append(random.choice(diagram_templates)())

    return qs

# --------------------------- Module: Cytoskeleton -----------------------------
def module_cytoskeleton(lo: List[str], topic: str, n_mcq: int, n_open: int, n_diag: int) -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    tags = ["cytoskeleton","actin","microtubules","motility","kinesin","dynein"]

    mcq_templates = [
        lambda: mk_mcq(
            prompt=("A fibroblast can polymerize actin but cannot depolymerize F‑actin. "
                    "Which cellular process is MOST impaired?"),
            choices=["Lamellipodia extension","Filopodia retraction","Stress fiber stability","Directional migration speed"],
            correct_idx=3,
            rationale=("Without depolymerization, treadmilling/recycling of G‑actin is limited → slower migration."),
            los=lo, tags=tags),
        lambda: mk_mcq(
            prompt=("Taxol stabilizes microtubules. Which outcome in mitosis is most likely?"),
            choices=["Accelerated anaphase A","Failure of spindle dynamics and chromosome segregation","Enhanced kinetochore capture","Faster cytokinesis"],
            correct_idx=1,
            rationale=("Excess stabilization impairs dynamic instability needed for spindle function and segregation."),
            los=lo, tags=tags),
    ]

    open_templates = [
        lambda: mk_open(
            prompt=("A GDP‑dissociation inhibitor (GDI) specific to Rho GTPases is added to migrating cells. "
                    "Explain effects on actin remodeling and cell movement."),
            rubric=("GDIs sequester Rho/Rac/Cdc42 in GDP‑bound state → reduced lamellipodia/filopodia dynamics → impaired motility."),
            los=lo, tags=tags),
        lambda: mk_open(
            prompt=("Explain the roles of ATP and Ca²⁺ in skeletal muscle contraction and relaxation."),
            rubric=("ATP binding releases myosin from actin; hydrolysis cocks head; Pi release drives power stroke; "
                    "Ca²⁺ binds troponin, moves tropomyosin to expose sites; Ca²⁺ reuptake promotes relaxation."),
            los=lo, tags=tags),
    ]

    diagram_templates = [
        lambda: mk_diagram(
            prompt=("Draw focal adhesions and indicate: integrin‑ECM link, actin stress fibers, and direction of traction force "
                    "during cell migration."),
            rubric=("Integrins link ECM to actin; acto‑myosin pulls cell body; leading edge protrusion + rear retraction."),
            los=lo, tags=tags),
    ]

    for _ in range(min(n_mcq, len(mcq_templates))):
        qs.append(random.choice(mcq_templates)())
    for _ in range(min(n_open, len(open_templates))):
        qs.append(random.choice(open_templates)())
    for _ in range(min(n_diag, len(diagram_templates))):
        qs.append(random.choice(diagram_templates)())

    return qs

# --------------------------- Module: Cell Cycle -------------------------------
def module_cell_cycle(lo: List[str], topic: str, n_mcq: int, n_open: int, n_diag: int) -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    tags = ["cell-cycle","cyclin","cdk","p53","mitosis","meiosis"]

    mcq_templates = [
        lambda: mk_mcq(
            prompt=("Cyclin E is absent in a cell line. Which checkpoint transition is MOST directly affected?"),
            choices=["G₂/M","G₁/S","Metaphase/Anaphase","G₀ to G₁ reentry"],
            correct_idx=1,
            rationale=("Cyclin E‑CDK2 promotes G₁→S transition; absence arrests before DNA replication."),
            los=lo, tags=tags),
        lambda: mk_mcq(
            prompt=("A nondisjunction event in meiosis I typically results in which gamete outcome?"),
            choices=["All gametes euploid","Half aneuploid, half euploid","All gametes aneuploid","Only two gametes affected"],
            correct_idx=2,
            rationale=("MI nondisjunction missegregates homologs → all four gametes abnormal following meiosis II."),
            los=lo, tags=tags),
    ]

    open_templates = [
        lambda: mk_open(
            prompt=("Explain why p53 is called the 'guardian of the genome' with reference to checkpoints and DNA damage responses."),
            rubric=("p53 halts cycle with damage (e.g., p21 induction), facilitates repair; can trigger apoptosis if damage severe."),
            los=lo, tags=tags),
    ]

    diagram_templates = [
        lambda: mk_diagram(
            prompt=("Draw the APC/C activation timeline and annotate how securin degradation triggers separase activation and chromatid separation."),
            rubric=("APC/C tags securin → separase active → cohesin cleavage → anaphase onset; spindle checkpoint ensures proper attachment."),
            los=lo, tags=tags),
    ]

    for _ in range(min(n_mcq, len(mcq_templates))):
        qs.append(random.choice(mcq_templates)())
    for _ in range(min(n_open, len(open_templates))):
        qs.append(random.choice(open_templates)())
    for _ in range(min(n_diag, len(diagram_templates))):
        qs.append(random.choice(diagram_templates)())

    return qs

# --------------------------- Module: Signaling --------------------------------
def module_signaling(lo: List[str], topic: str, n_mcq: int, n_open: int, n_diag: int) -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    tags = ["signaling","RTK","GPCR","PI3K","AKT","MAPK"]

    mcq_templates = [
        lambda: mk_mcq(
            prompt=("A receptor tyrosine kinase (RTK) dimerizes and autophosphorylates in the absence of ligand. "
                    "Which downstream effect is MOST consistent?"),
            choices=[
                "Transient growth arrest via p53",
                "Constitutive MAPK activation promoting proliferation",
                "Increased apoptosis via BAD activation",
                "Suppressed PI3K‑AKT signaling"
            ],
            correct_idx=1,
            rationale=("Ligand‑independent RTK activation drives Ras/MAPK cascades → pro‑growth signaling."),
            los=lo, tags=tags),
        lambda: mk_mcq(
            prompt=("A mutation renders AKT catalytically inactive. Which effect on cell fate is MOST likely?"),
            choices=["Enhanced cell survival signaling","Impaired phosphorylation of BAD and increased apoptosis","Constitutive mTOR activation","Unchanged survival pathways"],
            correct_idx=1,
            rationale=("AKT normally phosphorylates BAD to inhibit apoptosis; loss of AKT activity tips toward apoptosis."),
            los=lo, tags=tags),
    ]

    open_templates = [
        lambda: mk_open(
            prompt=("Contrast 'fast' vs 'slow' signaling responses with examples of altering protein function vs altering protein abundance."),
            rubric=("Fast: post‑translational mods/relocation (sec‑min). Slow: transcription/translation (hours). Provide examples."),
            los=lo, tags=tags),
    ]

    diagram_templates = [
        lambda: mk_diagram(
            prompt=("Sketch PI3K‑AKT signaling from RTK activation to BAD inhibition and indicate intervention points for targeted therapy."),
            rubric=("RTK→PI3K→PIP₃→AKT activation→phospho‑BAD→pro‑survival; interventions: RTK inhibitors, PI3K inhibitors, AKT inhibitors."),
            los=lo, tags=tags),
    ]

    for _ in range(min(n_mcq, len(mcq_templates))):
        qs.append(random.choice(mcq_templates)())
    for _ in range(min(n_open, len(open_templates))):
        qs.append(random.choice(open_templates)())
    for _ in range(min(n_diag, len(diagram_templates))):
        qs.append(random.choice(diagram_templates)())

    return qs

# --------------------------- Router -------------------------------------------
def route_topic_to_modules(topic: str) -> List[str]:
    t = topic.lower()
    modules = []
    if any(k in t for k in ["glycolysis","mitochond","tca","oxidative","oxphos","electron transport","metab"]):
        modules.append("metabolism")
    if any(k in t for k in ["actin","microtubule","kinesin","dynein","cytoskeleton","migration","muscle"]):
        modules.append("cytoskeleton")
    if any(k in t for k in ["cell cycle","cyclin","cdk","p53","mitosis","meiosis","checkpoint"]):
        modules.append("cell_cycle")
    if any(k in t for k in ["signal","rtk","gpcr","pi3k","akt","mapk","erk","ras"]):
        modules.append("signaling")
    # Default: if nothing matches, include metabolism as a safe example
    if not modules:
        modules.append("metabolism")
    return modules

def build_questions(topic: str, los: List[str], n_mcq: int, n_open: int, n_diag: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    modules = route_topic_to_modules(topic)
    per_mod_mcq = max(0, n_mcq // len(modules)) if modules else 0
    per_mod_open = max(0, n_open // len(modules)) if modules else 0
    per_mod_diag = max(0, n_diag // len(modules)) if modules else 0

    questions: List[Dict[str, Any]] = []
    for m in modules:
        if m == "metabolism":
            questions.extend(module_metabolism(los, topic, per_mod_mcq, per_mod_open, per_mod_diag))
        elif m == "cytoskeleton":
            questions.extend(module_cytoskeleton(los, topic, per_mod_mcq, per_mod_open, per_mod_diag))
        elif m == "cell_cycle":
            questions.extend(module_cell_cycle(los, topic, per_mod_mcq, per_mod_open, per_mod_diag))
        elif m == "signaling":
            questions.extend(module_signaling(los, topic, per_mod_mcq, per_mod_open, per_mod_diag))

    # If integer division truncated, top-up with metabolism templates
    short_mcq = n_mcq - sum(1 for q in questions if q["type"]=="mcq")
    short_open = n_open - sum(1 for q in questions if q["type"]=="open")
    short_diag = n_diag - sum(1 for q in questions if q["type"]=="diagram")
    if short_mcq > 0: questions.extend(module_metabolism(los, topic, short_mcq, 0, 0))
    if short_open > 0: questions.extend(module_metabolism(los, topic, 0, short_open, 0))
    if short_diag > 0: questions.extend(module_metabolism(los, topic, 0, 0, short_diag))

    # Attach topic tag to all
    for q in questions:
        q.setdefault("tags", []).append(topic.strip())
    return questions

# --------------------------- UI ----------------------------------------------
with st.sidebar:
    st.title("🧪 Instructor Controls")
    st.caption("This tool generates **original** critical‑thinking questions from your prompts. "
               "It never reveals your private exam questions.")
    instructor_mode = st.toggle("Instructor mode (show answers & rationales)", True)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    st.markdown("---")
    st.markdown("**Export settings**")
    add_timestamp = st.checkbox("Append timestamp to export filenames", True)

st.title("🧠 Critical Thinking Question Generator (Prompt‑based)")
st.write("Enter a topic and (optionally) learning objectives. Choose quantities for each question type, then **Generate**.")

colA, colB = st.columns([2,1])
with colA:
    topic = st.text_input("Topic prompt (e.g., 'Glycolysis → TCA → OxPhos' or 'PI3K‑AKT survival signaling')",
                          value="Glycolysis, TCA cycle, and Oxidative Phosphorylation")
with colB:
    st.info("Tip: Include keywords (e.g., 'actin', 'cyclin', 'RTK') to route templates to the right module(s).")

lo_text = st.text_area("Learning objectives (optional, one per line)",
                       value="Predict effects of inhibitors on ETC and ATP synthesis\nConnect substrate flux to redox carriers and ATP yield\nDistinguish fast vs slow signaling responses",
                       height=120)
n_mcq = st.number_input("Number of MCQs", min_value=0, max_value=50, value=5, step=1)
n_open = st.number_input("Number of open responses", min_value=0, max_value=50, value=3, step=1)
n_diag = st.number_input("Number of diagram/prediction prompts", min_value=0, max_value=50, value=2, step=1)

gen_btn = st.button("🚀 Generate Questions")

questions: List[Dict[str, Any]] = []
if gen_btn:
    los = sanitize_lo_text(lo_text)
    questions = build_questions(topic, los, n_mcq, n_open, n_diag, seed)
    st.success(f"Generated {len(questions)} questions.")

if questions:
    st.markdown("### Preview")
    st.caption("Student preview hides answers/rationales unless revealed. Toggle Instructor mode in the sidebar.")

    for i, q in enumerate(questions, start=1):
        with st.expander(f"{i}. ({q['type'].upper()}) {q['prompt'][:120]}{'...' if len(q['prompt'])>120 else ''}", expanded=False):
            st.write(q["prompt"])
            if q["type"] == "mcq":
                # Show choices without grading (preview only)
                st.radio("Choices:", q["choices"], index=None, key=q["id"]+"_choice", label_visibility="collapsed")
                if instructor_mode:
                    ans = q["choices"][q["answer_index"]]
                    st.markdown(f"**Answer:** {ans}")
                    st.markdown(f"**Rationale:** {q.get('rationale','')}")
                else:
                    if st.toggle("Reveal answer", key=q["id"]+"_rev"):
                        ans = q["choices"][q["answer_index"]]
                        st.markdown(f"**Answer:** {ans}")
                        st.markdown(f"**Rationale:** {q.get('rationale','')}")

            else:  # open or diagram
                st.text_area("Student response space (preview):", value="", height=100, key=q["id"]+"_resp")
                if instructor_mode:
                    st.markdown(f"**Suggested rubric / solution notes:** {q.get('rationale','')}")
                else:
                    if st.toggle("Reveal rubric", key=q["id"]+"_rub"):
                        st.markdown(f"**Suggested rubric / solution notes:** {q.get('rationale','')}")

            # Meta
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.caption(f"**Bloom:** {q.get('cognitive_level','')}")
            with meta_cols[1]:
                st.caption(f"**LOs:** {', '.join(q.get('los', [])) or '—'}")
            with meta_cols[2]:
                st.caption(f"**Tags:** {', '.join(q.get('tags', []))}")

    # ------------------- Export -------------------
    st.markdown("---")
    st.subheader("Export")
    fname_base = "question_bank"
    if add_timestamp:
        fname_base += "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_bytes = export_json(questions)
    csv_bytes = export_csv(questions)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download JSON", data=json_bytes, file_name=f"{fname_base}.json", mime="application/json")
    with col2:
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=f"{fname_base}.csv", mime="text/csv")

# --------------------------- Footer ------------------------------------------
st.markdown("---")
st.caption("Instructor note: This app uses only on‑device templates + your prompts. "
           "It does not reveal or reuse private exam content. Customize question text freely.")
