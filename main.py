# ✅ Adaptive GPT Tutor – Streamlit App with Reliable Single-Click Flow and Review Screen
# Requires: openai==0.28.1, streamlit

import openai
import streamlit as st

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Initialize Session State ---
def init_session():
    defaults = {
        "student_data": {},
        "question_count": 0,
        "max_questions": 5,
        "score": 0,
        "current_question": None,
        "current_difficulty": "easy",
        "student_id": "",
        "topic": "",
        "review": [],
        "review_mode": False,
        "answer_submitted": False,
        "last_result": "",
        "selected_answer": None,
        "phase": "setup"  # setup, quiz, feedback, complete
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

# --- Performance Tracking ---
def get_student_level(student_id):
    record = st.session_state["student_data"].get(student_id, {"attempts": 0, "correct": 0})
    if record["attempts"] == 0:
        return "easy"
    accuracy = record["correct"] / record["attempts"]
    return "hard" if accuracy > 0.8 else "easy"

def update_student_record(student_id, correct):
    record = st.session_state["student_data"].setdefault(student_id, {"attempts": 0, "correct": 0})
    record["attempts"] += 1
    record["correct"] += int(correct)

# --- GPT Prompts ---
def build_prompt(topic, difficulty):
    return (
        f"Ask one {difficulty} multiple-choice question about {topic}, using only content from the lecture slides. "
        f"Provide 4 labeled options (A-D), one of which is correct. Do NOT give the correct answer or any explanation."
    )

def get_question(topic, difficulty):
    prompt = build_prompt(topic, difficulty)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a cell biology tutor. Only ask questions based on the lecture slides."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return "❌ Failed to generate question."

def evaluate_answer(question, student_answer):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Evaluate a student's answer. Reply only with 'Correct' or 'Incorrect'."},
                {"role": "user", "content": f"Question: {question}\nAnswer: {student_answer}"}
            ]
        )
        verdict = response.choices[0].message.content.strip().lower()
        return verdict.startswith("correct")
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return False

# --- Streamlit UI ---
st.set_page_config(page_title="Adaptive Cell Bio Tutor", layout="centered")
st.title("🧬 Adaptive Cell Biology Tutor")

# --- Review Mode ---
if st.session_state["review_mode"]:
    st.header("📝 Review Your Answers")
    for idx, entry in enumerate(st.session_state["review"], 1):
        st.subheader(f"Question {idx}")
        st.markdown(entry["question"])
        st.markdown(f"**Your answer:** {entry['answer']}")
        st.markdown(f"**Result:** {'✅ Correct' if entry['correct'] else '❌ Incorrect'}")
    st.stop()

# --- Setup Phase ---
if st.session_state["phase"] == "setup":
    student_id = st.text_input("Enter your name or ID:", value=st.session_state["student_id"])
    topic = st.text_input("Enter a topic (e.g., protein sorting):", value=st.session_state["topic"])

    if student_id and topic and st.button("Start Session"):
        st.session_state["student_id"] = student_id
        st.session_state["topic"] = topic
        st.session_state["question_count"] = 0
        st.session_state["score"] = 0
        st.session_state["review"] = []
        st.session_state["current_difficulty"] = get_student_level(student_id)
        st.session_state["current_question"] = get_question(topic, st.session_state["current_difficulty"])
        st.session_state["phase"] = "quiz"
        st.session_state["answer_submitted"] = False
        st.session_state["last_result"] = ""
        st.session_state["selected_answer"] = None

# --- Quiz Phase ---
if st.session_state["phase"] == "quiz":
    st.subheader(f"❓ Question {st.session_state['question_count'] + 1} of {st.session_state['max_questions']}")
    st.markdown(st.session_state["current_question"])

    st.session_state["selected_answer"] = st.radio("Select your answer:", ["A", "B", "C", "D"], key=f"q{st.session_state['question_count']}")

    if st.button("Submit Answer") and not st.session_state["answer_submitted"]:
        correct = evaluate_answer(st.session_state["current_question"], st.session_state["selected_answer"])
        update_student_record(st.session_state["student_id"], correct)
        st.session_state["score"] += int(correct)
        st.session_state["last_result"] = (
            "✅ Great job! That’s correct. You’re doing well."
            if correct else
            "❌ That’s not quite right. Don’t worry—review the concept and try again!"
        )
        st.session_state["review"].append({
            "question": st.session_state["current_question"],
            "answer": st.session_state["selected_answer"],
            "correct": correct
        })
        st.session_state["answer_submitted"] = True

    if st.session_state["answer_submitted"]:
        st.info(st.session_state["last_result"])
        if st.button("Next Question"):
            st.session_state["question_count"] += 1
            if st.session_state["question_count"] >= st.session_state["max_questions"]:
                st.session_state["phase"] = "complete"
            else:
                st.session_state["current_difficulty"] = get_student_level(st.session_state["student_id"])
                st.session_state["current_question"] = get_question(st.session_state["topic"], st.session_state["current_difficulty"])
                st.session_state["answer_submitted"] = False
                st.session_state["selected_answer"] = None

# --- Completion Phase ---
if st.session_state["phase"] == "complete":
    st.success(f"🎉 Session complete! You got {st.session_state['score']} out of {st.session_state['max_questions']} correct.")
    if st.button("Review Your Answers"):
        st.session_state["review_mode"] = True
