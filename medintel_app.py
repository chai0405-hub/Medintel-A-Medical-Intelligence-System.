# medintel_app.py
# ----------------------------------------------------------
# MedIntel – A Medical Intelligence System (Complete)
# Works with medintel.db created by your medintel_setup_db.py
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
# App Config
# -----------------------
st.set_page_config(page_title="MedIntel – A Medical Intelligence System",
                   page_icon="🩺", layout="wide")
st.title("MedIntel – A Medical Intelligence System 🩺")

DB_FILE = "medintel.db"
if not os.path.exists(DB_FILE):
    st.error(f"Database file '{DB_FILE}' not found. Run medintel_setup_db.py first.")
    st.stop()

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

MODEL = load_model()

# -----------------------
# Load Doctors
# -----------------------
@st.cache_data(ttl=3600)
def load_doctors():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    try:
        df = pd.read_sql_query("SELECT * FROM doctors", conn)
    finally:
        conn.close()
    # Build a searchable text field
    df["doc_text"] = (df["specialty"].fillna("") + " " + df["languages"].fillna("") + " " +
                      df["bio"].fillna("") + " " + df["city"].fillna("")).str.lower()
    return df

doctors_df = load_doctors()

@st.cache_data(ttl=3600)
def compute_embeddings(texts):
    if not texts:
        return None
    return MODEL.encode(texts, convert_to_tensor=True)

doctor_embeddings = compute_embeddings(doctors_df["doc_text"].tolist())

# -----------------------
# DB Helper
# -----------------------
def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

# -----------------------
# Helper Functions
# -----------------------
def normalize_text(s):
    return str(s).lower().strip() if s else ""

def detect_specialty(symptoms_text):
    if not symptoms_text or str(symptoms_text).strip() == "":
        return "General Physician"
    symptom_emb = MODEL.encode(str(symptoms_text).lower(), convert_to_tensor=True)
    max_score = -1
    predicted_specialty = "General Physician"
    for spec, keywords in specialties_keywords.items():
        spec_emb = MODEL.encode(str(keywords).lower(), convert_to_tensor=True)
        score = util.cos_sim(symptom_emb, spec_emb).item()
        if score > max_score:
            max_score = score
            predicted_specialty = spec
    return predicted_specialty

def insert_patient(patient_name, age, city, language):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO patients (name, age, city, language, history) VALUES (?, ?, ?, ?, ?)',
                (patient_name, age, city, language, "[]"))
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return pid

def insert_visit(patient_id, doctor_id, symptoms, predicted_specialty, matched_score):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO visits (patient_id, doctor_id, symptoms, predicted_specialty, matched_score, visit_date) VALUES (?, ?, ?, ?, ?, ?)',
                (patient_id, doctor_id, symptoms, predicted_specialty, matched_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    # Return the visit_id of the inserted visit
    visit_id = cur.lastrowid
    conn.close()
    return visit_id

def generate_pdf_bytes(patient_info, doctor_info, symptoms, specialty):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "MedIntel Medical Report", 0, 1, 'C')
    pdf.ln(8)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 7, f"Patient Name: {patient_info.get('name','')}")
    pdf.multi_cell(0, 7, f"Age: {patient_info.get('age','')}")
    pdf.multi_cell(0, 7, f"City: {patient_info.get('city','')}")
    pdf.multi_cell(0, 7, f"Language: {patient_info.get('language','')}")
    pdf.ln(3)
    pdf.multi_cell(0, 7, f"Symptoms: {symptoms}")
    pdf.multi_cell(0, 7, f"Predicted Specialty: {specialty}")
    pdf.ln(3)
    pdf.multi_cell(0, 7, f"Recommended Doctor: {doctor_info.get('name','N/A')} ({doctor_info.get('specialty','N/A')})")
    pdf.multi_cell(0, 7, f"Experience: {doctor_info.get('experience','N/A')} years | Rating: {doctor_info.get('rating','N/A')}/5")
    return pdf.output(dest='S').encode('latin-1')

# -----------------------
# Feedback / Appointment Insert Helpers
# -----------------------
def insert_appointment(patient_id, doctor_id, date_str, time_str, notes):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO appointments (patient_id, doctor_id, date, time, notes, status) VALUES (?, ?, ?, ?, ?, ?)',
                (patient_id, doctor_id, date_str, time_str, notes, "Scheduled"))
    conn.commit()
    conn.close()

def insert_feedback(visit_id, doctor_id, patient_id, rating, comments):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO feedback (visit_id, doctor_id, patient_id, rating, comments) VALUES (?, ?, ?, ?, ?)',
                (visit_id, doctor_id, patient_id, rating, comments))
    conn.commit()
    conn.close()

# -----------------------
# Session State
# -----------------------
if "last_pdf" not in st.session_state:
    st.session_state["last_pdf"] = None
if "last_pdf_name" not in st.session_state:
    st.session_state["last_pdf_name"] = None

# -----------------------
# Patient Portal
# -----------------------
st.subheader("Patient Portal")
with st.form("patient_form"):
    patient_name = st.text_input("Patient Name")
    patient_age = st.number_input("Age", 0, 120, 25)
    patient_city = st.text_input("City")
    patient_language = st.text_input("Language")
    patient_symptoms = st.text_area("Symptoms")
    patient_severity = st.slider("Symptom Severity (1-5)", 1, 5, 3)
    submitted = st.form_submit_button("Find Doctors")

if submitted:
    conn = get_conn()
    try:
        patient_row = pd.read_sql_query("SELECT * FROM patients WHERE name = ?", conn, params=(patient_name,))
        if patient_row.empty:
            patient_id = insert_patient(patient_name, patient_age, patient_city, patient_language)
        else:
            patient_id = int(patient_row.iloc[0]["id"])
    finally:
        conn.close()

    predicted_specialty = detect_specialty(patient_symptoms)
    st.info(f"Detected Specialty: **{predicted_specialty}**")

    # Compute similarity
    p_text = f"{predicted_specialty} {patient_language} {patient_symptoms} {patient_city}".lower()
    p_emb = MODEL.encode(p_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(p_emb, doctor_embeddings)[0].cpu().numpy()

    results = []
    for i, score in enumerate(cos_scores):
        doc = doctors_df.iloc[i]
        boost = 0
        if normalize_text(doc["city"]) == normalize_text(patient_city):
            boost += 0.10
        spec_lower = str(doc.get("specialty","")).lower()
        if patient_age <= 12 and "pediatric" in spec_lower:
            boost += 0.15
        elif patient_age >= 60 and ("general" in spec_lower or "multi" in spec_lower):
            boost += 0.15
        elif 13 <= patient_age <= 59 and ("general" in spec_lower or "physician" in spec_lower):
            boost += 0.10
        if patient_language and patient_language.lower() in str(doc.get("languages","")).lower():
            boost += 0.05
        boost += patient_severity * 0.02
        rating_val = float(doc.get("rating",0) or 0)
        exp_val = float(doc.get("experience",0) or 0)
        boost += (rating_val/5)*0.02 + (exp_val/20)*0.02
        final_score = min(float(score)+boost, 1.0)
        results.append({
            "id": int(doc.get("id", -1)),
            "name": doc.get("name"),
            "specialty": doc.get("specialty"),
            "city": doc.get("city"),
            "experience": exp_val,
            "rating": rating_val,
            "score": final_score,
            "latitude": doc.get("latitude"),
            "longitude": doc.get("longitude")
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

    # -----------------------
    # Display Top 5 Doctors
    # -----------------------
    st.subheader("Top Doctor Matches (Top 5)")
    for rank, d in enumerate(results, start=1):
        st.markdown(
            f"""
            <div style='border:1px solid #ddd; padding:12px; border-radius:12px; margin-bottom:12px; background:#f0f8ff; color:#000;'>
                <h4 style='margin:0'>🏅 Rank {rank}: {d['name']}</h4>
                <div style='color:#333; margin-top:4px;'>
                    <b>Specialty:</b> {d['specialty']} &nbsp; | &nbsp;
                    <b>City:</b> {d['city']} &nbsp; | &nbsp;
                    <b>Experience:</b> {d['experience']} yrs &nbsp; | &nbsp;
                    <b>Rating:</b> {d['rating']}/5
                </div>
                <div style='color:#555; margin-top:6px;'>Match Score: {d['score']:.3f} | 
                <a href="https://www.google.com/maps/search/?api=1&query={d['latitude']},{d['longitude']}" target="_blank">Open in Google Maps</a></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Insert visit and capture visit_id (use return value of insert_visit)
        visit_id = insert_visit(patient_id, d["id"], patient_symptoms, predicted_specialty, d["score"])
        # (You may use visit_id later when inserting feedback)

    # -----------------------
    # PDF Generation
    # -----------------------
    if results:
        pdf_bytes = generate_pdf_bytes(
            {"name": patient_name, "age": patient_age, "city": patient_city, "language": patient_language},
            results[0],
            patient_symptoms,
            predicted_specialty
        )
        st.session_state["last_pdf"] = pdf_bytes
        st.session_state["last_pdf_name"] = f"Medical_Report_{patient_name.replace(' ','_')}.pdf"

# -----------------------
# PDF Download
# -----------------------
if st.session_state.get("last_pdf") is not None:
    st.download_button(
        label="📄 Download Last Medical Report",
        data=st.session_state["last_pdf"],
        file_name=st.session_state.get("last_pdf_name","medical_report.pdf"),
        mime="application/pdf"
    )

# ----------------------------------------------------------
# Part 2 – Appointments, Feedback, Doctor Sidebar, Dashboards
# ----------------------------------------------------------
# -----------------------
# Appointment Booking
# -----------------------
st.subheader("Book an Appointment")
with st.form("appointment_form"):
    app_patient = st.text_input("Patient Name for Appointment", key="appt_patient")
    app_doctor = st.selectbox("Select Doctor", doctors_df["name"].tolist(), key="appt_doctor")
    app_date = st.date_input("Date", key="appt_date")
    app_time = st.time_input("Time", key="appt_time")
    app_notes = st.text_area("Notes", key="appt_notes")
    app_submitted = st.form_submit_button("Book Appointment")

if app_submitted:
    conn = get_conn()
    try:
        patient_row = pd.read_sql_query("SELECT * FROM patients WHERE name = ?", conn, params=(app_patient,))
        doctor_row = doctors_df[doctors_df["name"] == app_doctor]
        if not patient_row.empty and not doctor_row.empty:
            patient_id = int(patient_row.iloc[0]["id"])
            doctor_id = int(doctor_row.iloc[0]["id"])
            insert_appointment(patient_id, doctor_id, str(app_date), str(app_time), app_notes)
            st.success(f"Appointment booked with Dr. {app_doctor} on {app_date} at {app_time}.")
        else:
            st.error("Patient or Doctor not found.")
    finally:
        conn.close()

# -----------------------
# Feedback Submission
# -----------------------
st.subheader("Submit Feedback")
with st.form("feedback_form"):
    fb_patient = st.text_input("Your Name", key="fb_patient")
    fb_doctor = st.selectbox("Doctor to Review", doctors_df["name"].tolist(), key="fb_doctor")
    fb_rating = st.slider("Rating (1-5)", 1, 5, 5, key="fb_rating")
    fb_comments = st.text_area("Comments", key="fb_comments")
    fb_submitted = st.form_submit_button("Submit Feedback")

if fb_submitted:
    conn = get_conn()
    try:
        patient_row = pd.read_sql_query("SELECT * FROM patients WHERE name = ?", conn, params=(fb_patient,))
        doctor_row = doctors_df[doctors_df["name"] == fb_doctor]
        if not patient_row.empty and not doctor_row.empty:
            patient_id = int(patient_row.iloc[0]["id"])
            doctor_id = int(doctor_row.iloc[0]["id"])

            # IMPORTANT: visits table uses 'visit_id' as primary key in your DB schema.
            visits = pd.read_sql_query(
                "SELECT * FROM visits WHERE patient_id = ? AND doctor_id = ? ORDER BY visit_date DESC LIMIT 1",
                conn, params=(patient_id, doctor_id)
            )

            # Correctly extract 'visit_id' (matches medintel_setup_db.py)
            visit_id = int(visits.iloc[0]["visit_id"]) if not visits.empty else None

            insert_feedback(visit_id, doctor_id, patient_id, fb_rating, fb_comments)
            st.success("Thank you! Your feedback has been submitted.")
        else:
            st.error("Patient or Doctor not found.")
    finally:
        conn.close()

# -----------------------
# Sidebar – Doctor Dashboard & Appointments
# -----------------------
st.sidebar.subheader("Doctor Dashboard")

# Remove duplicate doctor names for selection
doctors_unique = doctors_df.drop_duplicates(subset=["name"])

# Doctor selection
selected_doctor = st.sidebar.selectbox(
    "Select Doctor",
    doctors_unique["name"].tolist(),  # Unique names only
    key="dash_doc"
)

if selected_doctor:
    conn = get_conn()
    try:
        # Get all rows of doctors with this name
        selected_doctor_rows = doctors_df[doctors_df["name"] == selected_doctor]
        doc_ids = selected_doctor_rows["id"].tolist()  # all matching IDs

        # For display, take the first doctor row
        doc_info = selected_doctor_rows.iloc[0]

        # Doctor info
        st.sidebar.markdown(f"**Specialty:** {doc_info['specialty']}")

        # Total appointments for all doctor IDs with this name
        # Use parameterized query with correct number of placeholders
        placeholders = ",".join(["?"] * len(doc_ids))
        visits = pd.read_sql_query(
            f"SELECT * FROM visits WHERE doctor_id IN ({placeholders})",
            conn, params=doc_ids
        )
        st.sidebar.markdown(f"**Total Appointments:** {len(visits)}")

        # Feedback for all doctor IDs
        feedbacks = pd.read_sql_query(
            f"SELECT * FROM feedback WHERE doctor_id IN ({placeholders})",
            conn, params=doc_ids
        )

        if not feedbacks.empty:
            st.sidebar.markdown(f"**Average Rating:** {feedbacks['rating'].mean():.2f}/5")
            st.sidebar.markdown(f"**Number of Feedbacks:** {len(feedbacks)}")
            st.sidebar.markdown("**Recent Feedbacks:**")
            for idx, fb in feedbacks.tail(5).iterrows():
                st.sidebar.markdown(f"- {fb['comments']} (⭐ {fb['rating']}/5)")

        # Upcoming Appointments
        st.sidebar.subheader("Upcoming Appointments")
        appointments = pd.read_sql_query(
            f"""
            SELECT a.date, a.time, a.notes, p.name as patient_name
            FROM appointments a
            JOIN patients p ON a.patient_id = p.id
            WHERE a.doctor_id IN ({placeholders})
            ORDER BY a.date, a.time
            """,
            conn, params=doc_ids
        )

        if not appointments.empty:
            today = date.today()
            upcoming = appointments[appointments["date"] >= str(today)]
            st.sidebar.markdown(f"**Upcoming Appointments:** {len(upcoming)}")
            for idx, appt in upcoming.iterrows():
                appt_date = appt['date']
                color = "#007BFF" if appt_date != str(today) else "#28A745"
                st.sidebar.markdown(
                    f"""
                    <div style='border:1px solid #ccc; border-radius:8px; padding:8px; margin-bottom:6px; background-color:{color}20;'>
                        🗓️ <b>{appt['date']} {appt['time']}</b><br>
                        👤 {appt['patient_name']}<br>
                        📝 {appt['notes'] if appt['notes'] else 'No notes'}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.sidebar.info("No appointments scheduled.")

    finally:
        conn.close()

# -----------------------
# Doctor Performance Dashboard
# -----------------------
st.subheader("Doctors Performance Dashboard")
conn = get_conn()
try:
    feedbacks = pd.read_sql_query(
        "SELECT doctor_id, AVG(rating) as avg_rating, COUNT(*) as total_feedbacks "
        "FROM feedback GROUP BY doctor_id", conn
    )
    doctors_perf = pd.merge(
        doctors_df, feedbacks, left_on="id", right_on="doctor_id", how="left"
    ).fillna({"avg_rating": 0, "total_feedbacks": 0})
    st.dataframe(
        doctors_perf[
            ["name", "specialty", "city", "experience", "rating", "avg_rating", "total_feedbacks"]
        ].sort_values(by="avg_rating", ascending=False)
    )
finally:
    conn.close()