# ✅ Adaptive GPT Tutor – Streamlit Version with multiple questions and supportive feedback
# Requires: openai==0.28.1, streamlit

import openai
import streamlit as st

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Secure key management

# --- In-memory session data ---
if "student_data" not in st.session_state:
    st.session_state["student_data"] = {}
if "question_count" not in st.session_state:
    st.session_state["question_count"] = 0
if "max_questions" not in st.session_state:
    st.session_state["max_questions"] = 5
if "score" not in st.session_state:
    st.session_state["score"] = 0
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# --- Student performance tracking ---
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

# --- Prompt builder ---
def build_prompt(topic, difficulty):
    return (
        f"Ask one {difficulty} multiple-choice question about {topic}, using only content from the lecture slides. "
        f"Provide 4 labeled options (A-D), one of which is correct. Do NOT give the correct answer or any explanation."
    )

# --- GPT question generator ---
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

# --- GPT evaluator ---
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

# --- Streamlit App UI ---
st.set_page_config(page_title="Adaptive Cell Bio Tutor", layout="centered")
st.title("🧬 Adaptive Cell Biology Tutor")

student_id = st.text_input("Enter your name or ID:")
topic = st.text_input("Enter a topic (e.g., protein sorting):")

if student_id and topic:
    difficulty = get_student_level(student_id)
    if st.button("Start Session"):
        st.session_state["question_count"] = 0
        st.session_state["score"] = 0
        question = get_question(topic, difficulty)
        st.session_state["question"] = question
        st.session_state["difficulty"] = difficulty
        st.session_state["last_result"] = None

if "question" in st.session_state and st.session_state["question_count"] < st.session_state["max_questions"]:
    st.subheader(f"❓ Question {st.session_state['question_count'] + 1} of {st.session_state['max_questions']}")
    st.markdown(st.session_state["question"])
    student_answer = st.radio("Select your answer:", ["A", "B", "C", "D"], key=st.session_state["question_count"])

    if st.button("Submit Answer"):
        correct = evaluate_answer(st.session_state["question"], student_answer)
        update_student_record(student_id, correct)
        st.session_state["question_count"] += 1

        if correct:
            st.session_state["score"] += 1
            st.session_state["last_result"] = "✅ Great job! That’s correct. You’re doing well."
        else:
            st.session_state["last_result"] = "❌ That’s not quite right. Don’t worry—review the concept and try again!"

        # Prepare next question
        next_difficulty = get_student_level(student_id)
        st.session_state["question"] = get_question(topic, next_difficulty)

if st.session_state.get("last_result"):
    st.info(st.session_state["last_result"])

if st.session_state["question_count"] >= st.session_state["max_questions"]:
    st.success(f"🎉 Session complete! You got {st.session_state['score']} out of {st.session_state['max_questions']} correct.")
