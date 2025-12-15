# ----------------------------------------------------------
# MedIntel – A Medical Intelligence System (ALL-IN-ONE FINAL FIXED)
# ----------------------------------------------------------

import sqlite3
import pandas as pd
from datetime import datetime, date
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from fpdf import FPDF
from specialty_keywords import specialties_keywords
import os

# -----------------------
# Constants
# -----------------------
DB_FILE = "medintel.db"

# -----------------------
# Database Creator
# -----------------------
def create_database():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        city TEXT,
        language TEXT,
        history TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        specialty TEXT,
        city TEXT,
        languages TEXT,
        experience INTEGER,
        rating REAL,
        bio TEXT,
        latitude REAL,
        longitude REAL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS visits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        doctor_id INTEGER,
        symptoms TEXT,
        predicted_specialty TEXT,
        matched_score REAL,
        visit_date TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        doctor_id INTEGER,
        date TEXT,
        time TEXT,
        notes TEXT,
        status TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id INTEGER,
        doctor_id INTEGER,
        patient_id INTEGER,
        rating INTEGER,
        comments TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
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

st.title("🩺 MedIntel – A Medical Intelligence System")

# -----------------------
# Load AI Model
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

MODEL = load_model()

# -----------------------
# DB Helper
# -----------------------
def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

# -----------------------
# Load Doctors
# -----------------------
@st.cache_data
def load_doctors():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM doctors", conn)
    conn.close()

    if not df.empty:
        df["doc_text"] = (
            df["specialty"].fillna("") + " " +
            df["languages"].fillna("") + " " +
            df["bio"].fillna("") + " " +
            df["city"].fillna("")
        ).str.lower()
    return df

doctors_df = load_doctors()

if not doctors_df.empty:
    doctor_embeddings = MODEL.encode(
        doctors_df["doc_text"].tolist(),
        convert_to_tensor=True
    )

# -----------------------
# Utility Functions
# -----------------------
def detect_specialty(symptoms):
    symptom_emb = MODEL.encode(symptoms.lower(), convert_to_tensor=True)
    best_score, best_spec = -1, "General Physician"

    for spec, keywords in specialties_keywords.items():
        spec_emb = MODEL.encode(" ".join(keywords), convert_to_tensor=True)
        score = util.cos_sim(symptom_emb, spec_emb).item()
        if score > best_score:
            best_score, best_spec = score, spec

    return best_spec

def insert_patient(name, age, city, language):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO patients VALUES (NULL, ?, ?, ?, ?, ?)",
                (name, age, city, language, "[]"))
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid

def insert_visit(pid, did, symptoms, specialty, score):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO visits VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                (pid, did, symptoms, specialty, score,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    vid = cur.lastrowid
    conn.close()
    return vid

def insert_appointment(pid, did, d, t, notes):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO appointments VALUES (NULL, ?, ?, ?, ?, ?, 'Scheduled')",
                (pid, did, str(d), str(t), notes))
    conn.commit()
    conn.close()

def insert_feedback(vid, did, pid, rating, comments):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback VALUES (NULL, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (vid, did, pid, rating, comments))
    conn.commit()
    conn.close()

# -----------------------
# Session State
# -----------------------
st.session_state.setdefault("patient_id", None)
st.session_state.setdefault("pdf", None)
st.session_state.setdefault("pdf_name", None)

# -----------------------
# Patient Portal
# -----------------------
st.subheader("👤 Patient Portal")

with st.form("patient_form"):
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 0, 120, 25)
    city = st.text_input("City")
    language = st.text_input("Language")
    symptoms = st.text_area("Symptoms")
    severity = st.slider("Severity (1–5)", 1, 5, 3)
    submit = st.form_submit_button("Find Doctors")

# -----------------------
# Doctor Matching + Booking (FIXED)
# -----------------------
if submit and not doctors_df.empty:
    conn = get_conn()
    dfp = pd.read_sql_query("SELECT * FROM patients WHERE name = ?", conn, params=(name,))
    conn.close()

    pid = insert_patient(name, age, city, language) if dfp.empty else int(dfp.iloc[0]["id"])
    st.session_state.patient_id = pid

    specialty = detect_specialty(symptoms)
    st.success(f"Detected Specialty: {specialty}")

    query_emb = MODEL.encode(f"{specialty} {city} {language} {symptoms}".lower(), convert_to_tensor=True)
    scores = util.cos_sim(query_emb, doctor_embeddings)[0].cpu().numpy()

    results = []
    for i, s in enumerate(scores):
        doc = doctors_df.iloc[i]
        results.append({**doc, "score": min(float(s) + severity * 0.02, 1.0)})

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

    st.subheader("🏥 Top Doctor Matches")

    for r in results:
        st.markdown(f"### {r['name']} ({r['specialty']}) — {r['score']:.3f}")
        visit_id = insert_visit(pid, r["id"], symptoms, specialty, r["score"])

        with st.expander("📅 Book Appointment"):
            with st.form(f"appointment_form_{r['id']}"):
                d = st.date_input("Date", min_value=date.today())
                t = st.time_input("Time")
                notes = st.text_area(
                    "Notes",
                    "Experiencing headache, fever and dry cough for the past few days."
                )
                confirm = st.form_submit_button("✅ Confirm Appointment")

                if confirm:
                    insert_appointment(pid, r["id"], d, t, notes)
                    st.success("✅ Appointment Booked Successfully!")
                    st.toast("Appointment confirmed 🩺")

        with st.expander("⭐ Give Feedback"):
            with st.form(f"feedback_form_{r['id']}"):
                rating = st.slider("Rating", 1, 5, 4)
                comment = st.text_area("Comment")
                fb = st.form_submit_button("Submit Feedback")

                if fb:
                    insert_feedback(visit_id, r["id"], pid, rating, comment)
                    st.success("Feedback Submitted")

# -----------------------
# My Appointments
# -----------------------
if st.session_state.patient_id:
    st.subheader("📋 My Appointments")

    conn = get_conn()
    appts = pd.read_sql_query("""
        SELECT a.date, a.time, a.status, a.notes,
               d.name AS doctor, d.specialty
        FROM appointments a
        JOIN doctors d ON a.doctor_id = d.id
        WHERE a.patient_id = ?
        ORDER BY a.date
    """, conn, params=(st.session_state.patient_id,))
    conn.close()

    if appts.empty:
        st.info("No appointments yet.")
    else:
        st.dataframe(appts, use_container_width=True)
