# ✅ Adaptive GPT Tutor – Streamlit App with Working Review + Single-Click Flow
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
        "last_result": None,
        "awaiting_answer": False,
        "current_question": None,
        "current_difficulty": "easy",
        "submitted": False,
        "selected_answer": None,
        "student_id": "",
        "topic": "",
        "review": [],
        "review_mode": False,
        "awaiting_next": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

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

if st.session_state["review_mode"]:
    st.header("📝 Review Your Answers")
    for idx, entry in enumerate(st.session_state["review"], 1):
        st.subheader(f"Question {idx}")
        st.markdown(entry["question"])
        st.markdown(f"**Your answer:** {entry['answer']}")
        st.markdown(f"**Result:** {'✅ Correct' if entry['correct'] else '❌ Incorrect'}")
    st.stop()

student_id = st.text_input("Enter your name or ID:", value=st.session_state["student_id"])
topic = st.text_input("Enter a topic (e.g., protein sorting):", value=st.session_state["topic"])

if student_id and topic and st.session_state["question_count"] == 0 and not st.session_state["awaiting_answer"]:
    if st.button("Start Session"):
        st.session_state["score"] = 0
        st.session_state["last_result"] = None
        st.session_state["student_id"] = student_id
        st.session_state["topic"] = topic
        st.session_state["question_count"] = 0
        st.session_state["review"] = []
        st.session_state["current_difficulty"] = get_student_level(student_id)
        st.session_state["current_question"] = get_question(topic, st.session_state["current_difficulty"])
        st.session_state["awaiting_answer"] = True
        st.session_state["submitted"] = False
        st.session_state["awaiting_next"] = False

if st.session_state["awaiting_answer"] and st.session_state["current_question"]:
    st.subheader(f"❓ Question {st.session_state['question_count'] + 1} of {st.session_state['max_questions']}")
    st.markdown(st.session_state["current_question"])
    answer_key = f"answer_{st.session_state['question_count']}"
    selected = st.radio("Select your answer:", ["A", "B", "C", "D"], key=answer_key)

    if not st.session_state["submitted"]:
        if st.button("Submit Answer"):
            correct = evaluate_answer(st.session_state["current_question"], selected)
            update_student_record(student_id, correct)
            st.session_state["score"] += int(correct)
            st.session_state["last_result"] = (
                "✅ Great job! That’s correct. You’re doing well."
                if correct else
                "❌ That’s not quite right. Don’t worry—review the concept and try again!"
            )
            st.session_state["review"].append({
                "question": st.session_state["current_question"],
                "answer": selected,
                "correct": correct
            })
            st.session_state["submitted"] = True
            st.session_state["awaiting_next"] = True

    if st.session_state["submitted"] and st.session_state["awaiting_next"]:
        st.info(st.session_state["last_result"])
        if st.button("Next Question"):
            st.session_state["question_count"] += 1
            st.session_state["awaiting_next"] = False
            if st.session_state["question_count"] >= st.session_state["max_questions"]:
                st.session_state["awaiting_answer"] = False
                st.success(f"🎉 Session complete! You got {st.session_state['score']} out of {st.session_state['max_questions']} correct.")
                if st.button("Review Your Answers"):
                    st.session_state["review_mode"] = True
            else:
                st.session_state["current_difficulty"] = get_student_level(student_id)
                st.session_state["current_question"] = get_question(topic, st.session_state["current_difficulty"])
                st.session_state["submitted"] = False
