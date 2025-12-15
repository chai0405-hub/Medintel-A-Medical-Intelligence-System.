# ----------------------------------------------------------
# MedIntel – A Medical Intelligence System
# FINAL ERROR-FREE SINGLE FILE VERSION
# ----------------------------------------------------------

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date
from sentence_transformers import SentenceTransformer, util
import os

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="MedIntel",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 MedIntel – A Medical Intelligence System")

# -----------------------
# Database Setup
# -----------------------
DB_FILE = "medintel.db"

def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def setup_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        city TEXT,
        language TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        specialty TEXT,
        city TEXT,
        experience INTEGER,
        rating REAL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        doctor_id INTEGER,
        date TEXT,
        time TEXT,
        notes TEXT,
        status TEXT
    )""")

    conn.commit()
    conn.close()

setup_db()

# -----------------------
# Insert Sample Doctors
# -----------------------
def insert_doctors():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM doctors")
    if cur.fetchone()[0] == 0:
        doctors = [
            ("Dr. Amit Sharma", "Cardiology", "Mumbai", 12, 4.6),
            ("Dr. Neha Verma", "Neurology", "Mumbai", 10, 4.5),
            ("Dr. Raj Malhotra", "General Physician", "Mumbai", 15, 4.4),
            ("Dr. Pooja Iyer", "Pulmonology", "Mumbai", 8, 4.3)
        ]
        cur.executemany(
            "INSERT INTO doctors VALUES (NULL,?,?,?,?,?)", doctors
        )
    conn.commit()
    conn.close()

insert_doctors()

# -----------------------
# Load ML Model
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------
# Load Doctors
# -----------------------
@st.cache_data
def load_doctors():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM doctors", conn)
    conn.close()
    df["text"] = (df["specialty"] + " " + df["city"]).str.lower()
    return df

doctors_df = load_doctors()
doctor_embeddings = model.encode(doctors_df["text"].tolist(), convert_to_tensor=True)

# -----------------------
# Patient Input
# -----------------------
st.subheader("👤 Patient Details")

with st.form("patient_form"):
    name = st.text_input("Name")
    age = st.number_input("Age", 1, 100, 25)
    city = st.text_input("City", "Mumbai")
    language = st.selectbox("Language", ["English", "Hindi"])
    symptoms = st.text_area("Symptoms")
    severity = st.slider("Severity (1–5)", 1, 5, 3)
    submit = st.form_submit_button("Find Doctors")

# -----------------------
# Find Doctors
# -----------------------
if submit and name and symptoms:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("INSERT INTO patients VALUES (NULL,?,?,?,?)",
                (name, age, city, language))
    patient_id = cur.lastrowid
    conn.commit()

    query = model.encode(
        f"{symptoms} {city}".lower(),
        convert_to_tensor=True
    )

    scores = util.cos_sim(query, doctor_embeddings)[0].cpu().numpy()

    results = doctors_df.copy()
    results["score"] = scores + (severity * 0.05)
    results = results.sort_values("score", ascending=False)

    st.subheader("🏥 Recommended Doctors")

    for _, r in results.iterrows():
        st.markdown(f"### {r['name']} ({r['specialty']})")
        st.write(f"Experience: {r['experience']} years")
        st.write(f"Rating: ⭐ {r['rating']}")

        with st.expander("📅 Book Appointment"):
            d = st.date_input("Date", min_value=date.today(), key=f"d{r['id']}")
            t = st.time_input("Time", key=f"t{r['id']}")
            notes = st.text_area(
                "Notes",
                "Experiencing fever, headache and cough",
                key=f"n{r['id']}"
            )

            if st.button("Confirm Appointment", key=f"b{r['id']}"):
                cur.execute("""
                INSERT INTO appointments VALUES
                (NULL,?,?,?,?,?,?)
                """, (patient_id, r["id"], str(d), str(t), notes, "Scheduled"))
                conn.commit()
                st.success("✅ Appointment Booked Successfully")

    conn.close()

# -----------------------
# Show Appointments
# -----------------------
st.subheader("📋 My Appointments")

conn = get_conn()
appts = pd.read_sql("""
SELECT a.date, a.time, a.status, a.notes,
       d.name AS doctor, d.specialty
FROM appointments a
JOIN doctors d ON a.doctor_id = d.id
ORDER BY a.date
""", conn)
conn.close()

if appts.empty:
    st.info("No appointments yet.")
else:
    st.dataframe(appts, use_container_width=True)
