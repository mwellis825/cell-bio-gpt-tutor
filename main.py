# ✅ Adaptive GPT Tutor – Streamlit Version
# Requires: openai, streamlit (pip install openai streamlit)

import openai
import json
import datetime
import streamlit as st

# --- Configuration ---
openai.api_key = "OPEN_API_KEY"  # Replace with your actual key

# --- Load or initialize student history ---
def load_student_data():
    try:
        with open("student_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_student_data(data):
    with open("student_data.json", "w") as f:
        json.dump(data, f, indent=2)

student_data = load_student_data()

# --- Get student level ---
def get_student_level(student_id):
    record = student_data.get(student_id, {"attempts": 0, "correct": 0})
    if record["attempts"] == 0:
        return "easy"
    accuracy = record["correct"] / record["attempts"]
    return "hard" if accuracy > 0.8 else "easy"

# --- Update record ---
def update_student_record(student_id, correct):
    record = student_data.setdefault(student_id, {"attempts": 0, "correct": 0})
    record["attempts"] += 1
    record["correct"] += int(correct)
    save_student_data(student_data)

# --- Prompt builder ---
def build_prompt(topic, difficulty):
    return (
        f"Ask one {difficulty} multiple-choice question about {topic}, using only content from the lecture slides. "
        f"Provide 4 labeled options (A-D), one of which is correct. Do NOT give the correct answer or any explanation."
    )

# --- GPT question generator ---
def get_question(topic, difficulty):
    prompt = build_prompt(topic, difficulty)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a cell biology tutor. Only ask questions based on the lecture slides."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

# --- GPT evaluator ---
def evaluate_answer(question, student_answer):
    response = openai.ChatCompletion.create(
        try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a cell biology tutor. Only ask questions based on the lecture slides."},
            {"role": "user", "content": prompt},
        ]
    )
except Exception as e:
    st.error(f"OpenAI Error: {e}")
    return "Error generating question."
model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Evaluate a student's answer. Reply only with 'Correct' or 'Incorrect'."},
            {"role": "user", "content": f"Question: {question}\nAnswer: {student_answer}"}
        ]
    )
    verdict = response.choices[0].message.content.strip().lower()
    return verdict.startswith("correct")

# --- Streamlit App ---
st.set_page_config(page_title="Adaptive Cell Bio Tutor", layout="centered")
st.title("🧬 Adaptive Cell Biology Tutor")

student_id = st.text_input("Enter your name or ID:")
topic = st.text_input("Enter a topic (e.g., protein sorting):")

if student_id and topic:
    difficulty = get_student_level(student_id)
    if st.button("Get Question"):
        question = get_question(topic, difficulty)
        st.session_state["question"] = question
        st.session_state["difficulty"] = difficulty

if "question" in st.session_state:
    st.subheader("❓ Question")
    st.markdown(st.session_state["question"])
    student_answer = st.radio("Select your answer:", ["A", "B", "C", "D"])

    if st.button("Submit Answer"):
        correct = evaluate_answer(st.session_state["question"], student_answer)
        update_student_record(student_id, correct)
        if correct:
            st.success("✅ Correct!")
        else:
            st.error("❌ Incorrect. Review the concept and try again.")
