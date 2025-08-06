# ✅ Adaptive GPT Tutor – Clean Single-Click Flow with Review Screen
# Requires: openai==0.28.1, streamlit

import openai
import streamlit as st

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

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

# --- GPT Prompts ---
def build_prompt(topic, difficulty):
    return (
        f"Ask one {difficulty} multiple-choice question about {topic}, using only content from the lecture slides. "
        f"Provide 4 labeled options (A-D), one of which is correct. Do NOT give the correct answer or any explanation."
    )

def get_question():
    prompt = build_prompt(st.session_state.topic, st.session_state.current_difficulty)
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

def evaluate_answer(question, answer):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Evaluate a student's answer. Reply only with 'Correct' or 'Incorrect'."},
                {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"},
            ]
        )
        return response.choices[0].message.content.lower().startswith("correct")
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return False

# --- UI ---
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
    st.subheader(f"❓ Question {st.session_state.question_count + 1} of {st.session_state.max_questions}")
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
