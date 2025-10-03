import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load and Train Model
# -----------------------------
@st.cache_resource
def load_model():
    file_path = "Original.xlsx" 
    data = pd.read_excel(file_path, sheet_name="original")

    features = ["Linguistic", "Musical", "Bodily", 
                "Logical - Mathematical", "Spatial-Visualization", "Interpersonal"]
    target = "Job profession"

    df_clean = data[features + [target]].dropna()
    X = df_clean[features]
    y = df_clean[target]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return clf, le, features, acc

clf, le, features, acc = load_model()

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🎓 Career Prediction Based on Multiple Intelligences")
st.write(f"✅ **Model Accuracy:** {acc*100:.2f}%")

st.write("""
This app predicts your most suitable career based on **six types of intelligence**.
Here’s what each means:
- **Linguistic** 📝 → How good you are with words, speaking, writing, storytelling.
- **Musical** 🎵 → Talent for rhythm, sounds, singing, instruments, music patterns.
- **Bodily (Kinesthetic)** 🤸 → Using your body, sports, dance, crafts, acting.
- **Logical - Mathematical** 🔢 → Problem-solving, numbers, reasoning, puzzles.
- **Spatial - Visualization** 🎨 → Imagine, draw, design, visualize in 3D.
- **Interpersonal** 🤝 → Understanding and working with people, teamwork, empathy.
""")

# -----------------------------
# Ask 5 Questions per Intelligence
# -----------------------------
def ask_questions(intel_name, questions):
    st.subheader(f"{intel_name} Questions (1–4 points each)")
    total_score = 0
    for i, q in enumerate(questions, 1):
        score = st.slider(f"Q{i}: {q}", 1, 4, 2)
        total_score += score
    # Normalize to 20 points
    total_score = int(total_score / 20 * 20)
    return total_score

# Example questions (simple and clear)
linguistic_q = [
    "I enjoy reading books or articles.",
    "I like writing stories or essays.",
    "I remember words easily.",
    "I enjoy explaining things to others.",
    "I like telling stories or jokes."
]

musical_q = [
    "I enjoy singing or playing instruments.",
    "I recognize music patterns easily.",
    "I enjoy listening to music often.",
    "I can keep rhythm easily.",
    "I can reproduce tunes or melodies."
]

bodily_q = [
    "I enjoy sports or physical activities.",
    "I can learn movements easily.",
    "I like dancing or acting.",
    "I am good at crafts or making things.",
    "I enjoy hands-on activities."
]

logical_q = [
    "I enjoy solving puzzles or brain games.",
    "I am good at math or numbers.",
    "I notice patterns easily.",
    "I enjoy experiments or problem-solving.",
    "I think logically to solve problems."
]

spatial_q = [
    "I can imagine objects in 3D easily.",
    "I like drawing, painting, or designing.",
    "I can read maps or diagrams easily.",
    "I notice colors and shapes well.",
    "I can visualize how things will look."
]

interpersonal_q = [
    "I understand how others feel.",
    "I work well in teams.",
    "I enjoy helping or teaching others.",
    "I communicate clearly with people.",
    "I notice moods and emotions of others."
]

# Get scores
linguistic_score = ask_questions("Linguistic", linguistic_q)
musical_score = ask_questions("Musical", musical_q)
bodily_score = ask_questions("Bodily (Kinesthetic)", bodily_q)
logical_score = ask_questions("Logical - Mathematical", logical_q)
spatial_score = ask_questions("Spatial - Visualization", spatial_q)
interpersonal_score = ask_questions("Interpersonal", interpersonal_q)

if st.button("Predict My Career"):
    input_scores = [[linguistic_score, musical_score, bodily_score,
                     logical_score, spatial_score, interpersonal_score]]
    probs = clf.predict_proba(input_scores)[0]
    sorted_idx = probs.argsort()[::-1]
    careers_sorted = le.inverse_transform(sorted_idx)
    probs_sorted = probs[sorted_idx]

    st.subheader("🔮 Career Possibilities")
    st.success(f"**High Possibility:** {careers_sorted[0]} ({probs_sorted[0]*100:.2f}%)")
    st.info(f"**Average Possibility:** {careers_sorted[1]} ({probs_sorted[1]*100:.2f}%)")
    st.warning(f"**Less Possibility:** {careers_sorted[2]} ({probs_sorted[2]*100:.2f}%)")

