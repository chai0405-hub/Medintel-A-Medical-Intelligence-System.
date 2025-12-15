# ----------------------------------------------------------
# MedIntel – A Medical Intelligence System (FINAL & FIXED)
# ----------------------------------------------------------

import sqlite3
import pandas as pd
from datetime import datetime, date
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from fpdf import FPDF
from specialty_keywords import specialties_keywords
import pydeck as pdk
import os

# -----------------------
# Constants
# -----------------------
DB_FILE = "medintel.db"

# -----------------------
# Safety DB Creator (only if DB missing)
# -----------------------
def create_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        city TEXT,
        language TEXT,
        history TEXT
    )
    """)

    conn.commit()
    conn.close()

if not os.path.exists(DB_FILE):
    create_database()

# -----------------------
# Streamlit Config
# -----------------------
st.set_page_config(
    page_title="MedIntel – A Medical Intelligence System",
    page_icon="🩺",
    layout="wide"
)

st.title("MedIntel – A Medical Intelligence System 🩺")

# -----------------------
# Load AI Model
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

MODEL = load_model()

# -----------------------
# Database Helper
# -----------------------
def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

# -----------------------
# Load Doctors
# -----------------------
@st.cache_data(ttl=3600)
def load_doctors():
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM doctors", conn)
    finally:
        conn.close()

    df["doc_text"] = (
        df["specialty"].fillna("") + " " +
        df["languages"].fillna("") + " " +
        df["bio"].fillna("") + " " +
        df["city"].fillna("")
    ).str.lower()

    return df

doctors_df = load_doctors()

@st.cache_data(ttl=3600)
def compute_embeddings(texts):
    return MODEL.encode(texts, convert_to_tensor=True)

doctor_embeddings = compute_embeddings(doctors_df["doc_text"].tolist())

# -----------------------
# Utility Functions
# -----------------------
def normalize_text(s):
    return str(s).lower().strip() if s else ""

def detect_specialty(symptoms):
    if not symptoms:
        return "General Physician"

    symptom_emb = MODEL.encode(symptoms.lower(), convert_to_tensor=True)
    best_score = -1
    best_spec = "General Physician"

    for spec, keywords in specialties_keywords.items():
        spec_emb = MODEL.encode(" ".join(keywords), convert_to_tensor=True)
        score = util.cos_sim(symptom_emb, spec_emb).item()
        if score > best_score:
            best_score = score
            best_spec = spec

    return best_spec

# -----------------------
# Insert Helpers
# -----------------------
def insert_patient(name, age, city, language):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO patients (name, age, city, language, history) VALUES (?, ?, ?, ?, ?)",
        (name, age, city, language, "[]")
    )
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid

def insert_visit(patient_id, doctor_id, symptoms, specialty, score):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO visits (patient_id, doctor_id, symptoms, predicted_specialty, matched_score, visit_date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (patient_id, doctor_id, symptoms, specialty, score, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    vid = cur.lastrowid
    conn.close()
    return vid

def insert_appointment(pid, did, date_str, time_str, notes):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO appointments (patient_id, doctor_id, date, time, notes, status)
        VALUES (?, ?, ?, ?, ?, 'Scheduled')
    """, (pid, did, date_str, time_str, notes))
    conn.commit()
    conn.close()

def insert_feedback(visit_id, doctor_id, patient_id, rating, comments):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO feedback (visit_id, doctor_id, patient_id, rating, comments)
        VALUES (?, ?, ?, ?, ?)
    """, (visit_id, doctor_id, patient_id, rating, comments))
    conn.commit()
    conn.close()

# -----------------------
# PDF Generator
# -----------------------
def generate_pdf(patient, doctor, symptoms, specialty):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "MedIntel Medical Report", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, f"Patient Name: {patient['name']}")
    pdf.multi_cell(0, 7, f"Age: {patient['age']}")
    pdf.multi_cell(0, 7, f"City: {patient['city']}")
    pdf.multi_cell(0, 7, f"Language: {patient['language']}")
    pdf.ln(4)

    pdf.multi_cell(0, 7, f"Symptoms: {symptoms}")
    pdf.multi_cell(0, 7, f"Predicted Specialty: {specialty}")
    pdf.ln(4)

    pdf.multi_cell(0, 7, f"Doctor: {doctor['name']} ({doctor['specialty']})")
    pdf.multi_cell(0, 7, f"Experience: {doctor['experience']} years")
    pdf.multi_cell(0, 7, f"Rating: {doctor['rating']}/5")

    return pdf.output(dest="S").encode("latin-1")

# -----------------------
# Session State
# -----------------------
if "last_pdf" not in st.session_state:
    st.session_state.last_pdf = None
    st.session_state.last_pdf_name = None

# -----------------------
# Patient Portal
# -----------------------
st.subheader("Patient Portal")

with st.form("patient_form"):
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 0, 120, 25)
    city = st.text_input("City")
    language = st.text_input("Language")
    symptoms = st.text_area("Symptoms")
    severity = st.slider("Severity (1–5)", 1, 5, 3)
    submit = st.form_submit_button("Find Doctors")

if submit:
    conn = get_conn()
    patient_df = pd.read_sql_query(
        "SELECT * FROM patients WHERE name = ?", conn, params=(name,)
    )
    conn.close()

    if patient_df.empty:
        patient_id = insert_patient(name, age, city, language)
    else:
        patient_id = int(patient_df.iloc[0]["id"])

    specialty = detect_specialty(symptoms)
    st.info(f"Detected Specialty: **{specialty}**")

    query_text = f"{specialty} {city} {language} {symptoms}".lower()
    q_emb = MODEL.encode(query_text, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, doctor_embeddings)[0].cpu().numpy()

    results = []
    for i, score in enumerate(scores):
        doc = doctors_df.iloc[i]
        final_score = min(float(score) + severity * 0.02, 1.0)

        results.append({
            "id": int(doc["id"]),
            "name": doc["name"],
            "specialty": doc["specialty"],
            "city": doc["city"],
            "experience": doc["experience"],
            "rating": doc["rating"],
            "score": final_score,
            "latitude": doc["latitude"],
            "longitude": doc["longitude"]
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

    st.subheader("Top Doctor Matches")
    for r in results:
        st.markdown(
            f"### {r['name']} ({r['specialty']}) — Score {r['score']:.3f}"
        )
        insert_visit(patient_id, r["id"], symptoms, specialty, r["score"])

    pdf = generate_pdf(
        {"name": name, "age": age, "city": city, "language": language},
        results[0],
        symptoms,
        specialty
    )

    st.session_state.last_pdf = pdf
    st.session_state.last_pdf_name = f"Medical_Report_{name}.pdf"

# -----------------------
# PDF Download
# -----------------------
if st.session_state.last_pdf:
    st.download_button(
        "📄 Download Medical Report",
        st.session_state.last_pdf,
        file_name=st.session_state.last_pdf_name,
        mime="application/pdf"
    )