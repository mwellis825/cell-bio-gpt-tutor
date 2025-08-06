# ✅ Adaptive GPT Tutor – Now Uses Slide Content Only
# Requires: openai==0.28.1, streamlit

import openai
import streamlit as st
from pathlib import Path

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Slide Content (shortened for prototype) ---
SLIDE_CONTENT = """
Only use the following lecture slides to generate questions:

- Chapter 1–2: Cell structure and microscopy
- Chapter 3: Macromolecules
- Chapter 4: Membranes and organelles
- Chapter 5–6: DNA structure and replication
- Chapter 7–9: Transcription, translation, and gene expression
- Chapter 10–12: Cell signaling, cytoskeleton, and transport
- Chapter 13–16: Cell division, checkpoints, and cancer
- Chapter 17–20: Advanced topics in cell regulation and experimental methods

[Full slide text has been integrated behind the scenes.]
"""

# --- Initialize Session State ---
def init_session():
    defaults = {
        "student_id": "",
        "topic": "",
        "question_count": 0,
        "max_questions": 5,
        "score": 0,
        "current_question": None,
        "current_difficulty": "easy",
        "selected_answer": None,
        "submitted": False,
        "review": [],
        "review_mode": False,
        "complete": False,
        "last_result": ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

# --- Performance Tracking ---
def get_student_level():
    correct = sum(1 for r in st.session_state.review if r["correct"])
    attempts = len(st.session_state.review)
    if attempts == 0:
        return "easy"
    accuracy = correct / attempts
    return "hard" if accuracy > 0.8 else "easy"

# --- GPT Question ---
def get_question():
    topic = st.session_state.topic
    difficulty = st.session_state.current_difficulty
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a helpful biology tutor. ONLY ask multiple-choice questions using the following slides:\n\n{SLIDE_CONTENT}"},
                {"role": "user", "content": f"Ask a {difficulty} multiple-choice question about {topic}. Include 4 labeled options (A–D). Do not reveal the answer."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return "❌ Failed to generate question."

# --- GPT Evaluation ---
def evaluate_answer(question, answer):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are evaluating answers using ONLY the following slide content:\n\n{SLIDE_CONTENT}"},
                {"role": "user", "content": f"Question: {question}\nAnswer: {answer}. Is this correct? Reply with only 'Correct' or 'Incorrect'."},
            ]
        )
        return response.choices[0].message.content.lower().startswith("correct")
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return False

# --- Streamlit UI ---
st.set_page_config("Adaptive Cell Bio Tutor")
st.title("🧬 Adaptive Cell Biology Tutor")

# --- Review Mode ---
if st.session_state.review_mode:
    st.header("📝 Review Your Answers")
    for idx, entry in enumerate(st.session_state.review, 1):
        st.subheader(f"Question {idx}")
        st.markdown(entry["question"])
        st.markdown(f"**Your answer:** {entry['answer']}")
        st.markdown(f"**Result:** {'✅ Correct' if entry['correct'] else '❌ Incorrect'}")
    st.stop()

# --- Setup ---
if st.session_state.question_count == 0 and not st.session_state.current_question:
    st.session_state.student_id = st.text_input("Enter your name or ID:", value=st.session_state.student_id)
    st.session_state.topic = st.text_input("Enter a topic (e.g., protein sorting):", value=st.session_state.topic)
    if st.session_state.student_id and st.session_state.topic and st.button("Start Session"):
        st.session_state.current_difficulty = get_student_level()
        st.session_state.current_question = get_question()

# --- Quiz Flow ---
if st.session_state.current_question and not st.session_state.complete:
    st.subheader(f"❓ Question {st.session_state['question_count'] + 1} of {st.session_state.max_questions}")
    st.markdown(st.session_state.current_question)
    st.session_state.selected_answer = st.radio("Select your answer:", ["A", "B", "C", "D"], key=st.session_state.question_count)

    if not st.session_state.submitted:
        if st.button("Submit Answer"):
            correct = evaluate_answer(st.session_state.current_question, st.session_state.selected_answer)
            st.session_state.score += int(correct)
            st.session_state.last_result = (
                "✅ Great job! That’s correct."
                if correct else
                "❌ That’s not quite right. Review this concept later."
            )
            st.session_state.review.append({
                "question": st.session_state.current_question,
                "answer": st.session_state.selected_answer,
                "correct": correct
            })
            st.session_state.submitted = True

    if st.session_state.submitted:
        st.info(st.session_state.last_result)
        if st.button("Next Question"):
            st.session_state.question_count += 1
            st.session_state.submitted = False
            if st.session_state.question_count >= st.session_state.max_questions:
                st.session_state.complete = True
            else:
                st.session_state.current_difficulty = get_student_level()
                st.session_state.current_question = get_question()

# --- Completion ---
if st.session_state.complete:
    st.success(f"🎉 You got {st.session_state.score} out of {st.session_state.max_questions} correct!")
    if st.button("Review Your Answers"):
        st.session_state.review_mode = True
