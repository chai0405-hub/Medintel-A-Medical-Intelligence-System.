
# ==========================================================
# MedIntel – Fully Integrated Medical Intelligence System
# ==========================================================

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from plyer import notification
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.linear_model import LinearRegression
       
# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="MedIntel", page_icon="🩺", layout="wide")

st.markdown("""
<h1 style='text-align:center; color:#2E86C1;'>🩺 MedIntel – Medical Intelligence System</h1>
<p style='text-align:center; font-size:18px;'>AI-powered health monitoring • Risk prediction • Smart care</p>
<hr>
""", unsafe_allow_html=True)

# ---------------- DATABASE SETUP ----------------
DB_FILE = "medintel.db"

def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    # Patients
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT, age INTEGER, city TEXT, language TEXT, password TEXT)''')
    # Doctors
    c.execute('''CREATE TABLE IF NOT EXISTS doctors
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT, specialty TEXT, city TEXT, languages TEXT, experience INTEGER, rating REAL, password TEXT,email TEXT)''')
    # Records
    c.execute('''CREATE TABLE IF NOT EXISTS records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id INTEGER,
                  visit_number INTEGER,
                  symptoms TEXT,
                  severity INTEGER,
                  heart_rate INTEGER,
                  sugar INTEGER,
                  bp TEXT,
                  steps INTEGER,
                  risk TEXT,
                  risk_score INTEGER,
                  specialty_risk TEXT,
                  date TEXT,
                  notes TEXT,
                  FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    # Medication
    c.execute('''CREATE TABLE IF NOT EXISTS medication
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id INTEGER,
                  medicine_name TEXT,
                  time TEXT,
                  taken INTEGER,
                  date TEXT,                 
                  confirmation TEXT,
                  FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS emergency
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 patient_id TEXT,
                 risk TEXT,
                 heart_rate INTEGER,
                 sugar INTEGER,
                 bp TEXT,
                 message TEXT,
                 date TEXT,
                 FOREIGN KEY(patient_id) REFERENCES patients(id))''')
     

    c.execute(''' CREATE TABLE IF NOT EXISTS chat 
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id INTEGER,
                  sender TEXT,
                  message TEXT,
                  timestamp TEXT
                  )
         ''')
    
   
    conn.commit()
    return conn, c

conn, c = init_db()

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.patient_id = None
    st.session_state.risk = ""
    st.session_state.specialty_risk = ""
    st.session_state.risk_score = 0
    st.session_state.chat_history = []

# ---------------- AUTHENTICATION ----------------
st.sidebar.header("🔐 Authentication")
role_input = st.sidebar.selectbox("Role", ["Patient", "Doctor", "Admin"])
auth_choice = st.sidebar.radio("Action", ["Login", "Register"])
username_input = st.sidebar.text_input("Username")
password_input = st.sidebar.text_input("Password", type="password")
login_btn = st.sidebar.button("Proceed")

if login_btn and username_input and password_input:
    if role_input == "Patient":
      row = pd.read_sql(
    "SELECT * FROM patients WHERE name=? AND password=?",
    conn,
    params=(username_input, password_input)
)
        if auth_choice == "Register":
            if row.empty:
                c.execute("INSERT INTO patients (name, age, city, language, password) VALUES (?,?,?,?,?)",
                          (username_input, 30, "Unknown", "English", password_input))
                conn.commit()
                st.success("Patient registered! Please login.")
            else:
                st.warning("Username already exists. Try logging in.")
        else:  # Login
            if not row.empty:
                st.session_state.logged_in = True
                st.session_state.username = username_input
                st.session_state.role = role_input
                st.success(f"Patient {username_input} logged in!")
            else:
                st.error("Invalid username or password.")

    elif role_input == "Doctor":
        row = pd.read_sql("SELECT * FROM doctors WHERE name=? AND password=?", conn, params=(username_input, password_input))
        if auth_choice == "Register":
            st.warning("Doctor registration must be done by Admin.")
        else:  # Login
            if not row.empty:
                st.session_state.logged_in = True
                st.session_state.username = username_input
                st.session_state.role = role_input
                st.success(f"Doctor {username_input} logged in!")
            else:
                st.error("Invalid username or password.")

    elif role_input == "Admin":
        if username_input == "admin" and password_input == "admin123":
            st.session_state.logged_in = True
            st.session_state.username = "Admin"
            st.session_state.role = "Admin"
            st.success("Admin logged in!")
        else:
            st.error("Invalid admin credentials.")

if not st.session_state.logged_in:
    st.info("Please login or register to continue")
    st.stop()

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.patient_id = None
    st.rerun()  # <-- immediately rerun the app

# ---------------- HELPER FUNCTIONS ----------------
def get_patient_id(name):
    row = pd.read_sql("SELECT * FROM patients WHERE name=?", conn, params=(name,))
    if row.empty:
        c.execute("INSERT INTO patients (name, age, city, language, password) VALUES (?,?,?,?,?)",
                  (name, 30, "Unknown", "English", "12345"))
        conn.commit()
        return c.lastrowid
    return int(row.iloc[0]["id"])

def calculate_risk(heart, sugar, sys, dia, steps, symptoms):
    score = 0
    specialty_risk = []

    # -------------------------------
    # 1. BLOOD SUGAR (WHO / ADA)
    # -------------------------------
    # Normal: <110
    # Pre-diabetes: 110–125 → GP
    # Diabetes: ≥126 → Endocrinology

    if sugar >= 126:
        score += 2
        specialty_risk.append("Endocrinology")
    elif 110 <= sugar < 126:
        score += 1
        specialty_risk.append("General Medicine")

    # -------------------------------
    # 2. BLOOD PRESSURE & HEART RATE (WHO)
    # -------------------------------
    # Hypertension ≥140/90
    # Tachycardia >100 bpm

    if sys >= 140 or dia >= 90 or heart > 100:
        score += 3
        specialty_risk.append("Cardiology")

    # Chest pain is always high-risk (WHO emergency)
    if "Chest Pain" in symptoms:
        score += 3
        specialty_risk.append("Cardiology")

    # -------------------------------
    # 3. RESPIRATORY SYSTEM
    # -------------------------------

    if "Shortness of Breath" in symptoms:
        score += 3
        specialty_risk.append("Respiratory")
    elif "Cough" in symptoms:
        score += 1
        specialty_risk.append("Respiratory")

    # -------------------------------
    # 4. NEUROLOGICAL SYSTEM
    # -------------------------------

    if "Headache" in symptoms:
        score += 1
        specialty_risk.append("Neurology")
    if "Fatigue" in symptoms:
        score += 1
        specialty_risk.append("General Medicine")

    # -------------------------------
    # 5. PHYSICAL ACTIVITY (WHO)
    # -------------------------------
    # <3000 steps → sedentary risk

    if steps < 3000:
        score += 1
        specialty_risk.append("General Medicine")

    # -------------------------------
    # 6. FINAL RISK LEVEL (WHO-STYLE TRIAGE)
    # -------------------------------

    if score >= 6:
        risk = "High"
    elif score >= 3:
        risk = "Medium"
    else:
        risk = "Low"

    # Remove duplicates
    specialty_risk = list(set(specialty_risk))

    # WHO principle: Primary care first if unclear
    if not specialty_risk:
        specialty_risk = ["General Medicine"]

    return risk, ", ".join(specialty_risk)

def ai_health_chat(query, patient_id):
    df = pd.read_sql(
        "SELECT * FROM records WHERE patient_id=? ORDER BY date DESC LIMIT 1",
        conn,
        params=(patient_id,)
    )

    if df.empty:
        return "Please enter your health data first so I can guide you better."

    row = df.iloc[0]

    risk = row["risk"]
    heart = row["heart_rate"]
    sugar = row["sugar"]
    bp = row["bp"]
    symptoms = row["symptoms"]

    response = []

    if risk == "High":
        response.append("🚨 Your current health risk is HIGH.")
        response.append("Immediate medical attention is strongly recommended.")
    elif risk == "Medium":
        response.append("⚠️ Your health risk is MEDIUM.")
        response.append("Please monitor your condition closely.")
    else:
        response.append("✅ Your health risk is LOW.")
        response.append("Maintain healthy habits and regular monitoring.")

    if heart > 120:
        response.append("Your heart rate is higher than normal.")
    if sugar > 180:
        response.append("Your blood sugar level is elevated.")
    if "Chest Pain" in symptoms:
        response.append("Chest pain is a serious symptom. Do not ignore it.")

    response.append("This AI guidance is supportive and does not replace your doctor.")

    return " ".join(response)

def get_last_risk(patient_id):
    df = pd.read_sql("SELECT * FROM records WHERE patient_id=? ORDER BY date DESC LIMIT 1", conn, params=(patient_id,))
    if not df.empty:
        st.session_state.risk = df.iloc[0]['risk']
        st.session_state.specialty_risk = df.iloc[0]['specialty_risk']
    else:
        st.session_state.risk = ""
        st.session_state.specialty_risk = ""
# ================= ML MODEL =================

SYMPTOM_INDEX = {
    # Cardiovascular
    "Chest Pain":0,"Palpitations":1,"Shortness of Breath":2,"Swelling in Legs":3,
    "High BP Symptoms":4,"Irregular Heartbeat":5,"Fainting":6,"Cold Sweats":7,
    "Jaw Pain":8,"Left Arm Pain":9,"Rapid Pulse":10,"Exercise Intolerance":11,
    "Orthopnea":12,"Cyanosis":13,"Chest Tightness":14,

    # Respiratory
    "Cough":15,"Wheezing":16,"Runny Nose":17,"Sore Throat":18,"Chest Congestion":19,
    "Asthma Attacks":20,"Night Cough":21,"Hemoptysis":22,"Snoring":23,"Sleep Apnea":24,
    "Rapid Breathing":25,"Hoarseness":26,"Postnasal Drip":27,"Pleural Pain":28,
    "Dry Cough":29,

    # Metabolic
    "Fatigue":30,"Frequent Thirst":31,"Frequent Urination":32,"Weight Gain":33,
    "Weight Loss":34,"Blurred Vision":35,"Slow Healing":36,"Cold Intolerance":37,
    "Heat Intolerance":38,"Hair Loss":39,"Excess Hunger":40,"Night Sweats":41,
    "Tremors":42,"Dehydration":43,"Low Energy":44,

    # Neurological
    "Headache":45,"Dizziness":46,"Numbness":47,"Memory Loss":48,"Confusion":49,
    "Seizures":50,"Speech Difficulty":51,"Balance Issues":52,"Tremors Neuro":53,
    "Vision Loss":54,"Tingling":55,"Sleep Disorders":56,"Behavior Change":57,
    "Blackouts":58,"Weakness":59,

    # Gastrointestinal
    "Nausea":60,"Abdominal Pain":61,"Diarrhea":62,"Constipation":63,
    "Heartburn":64,"Bloating":65,"Vomiting":66,"Blood in Stool":67,
    "Loss of Appetite":68,"Indigestion":69,"Acid Reflux":70,"Jaundice":71,
    "Gas":72,"Stomach Cramps":73,"Food Intolerance":74
}

SPECIALTIES = [
    "Cardiology","Respiratory","Endocrinology",
    "Neurology","Gastroenterology"
]

X, y = [], []
np.random.seed(42)

mapping = {
    "Cardiology": list(range(0,15)),
    "Respiratory": list(range(15,30)),
    "Endocrinology": list(range(30,45)),
    "Neurology": list(range(45,60)),
    "Gastroenterology": list(range(60,75))
}

for _ in range(1000):
    vec = np.zeros(len(SYMPTOM_INDEX))
    spec = np.random.choice(SPECIALTIES)

    for idx in mapping.get(spec, []):
        if np.random.rand() > 0.4:
            vec[idx] = 1

    severity = np.random.randint(1,4)
    duration = np.random.randint(1,4)

    X.append(np.concatenate([vec, [severity, duration]]))
    y.append(spec)

le = LabelEncoder()
y_enc = le.fit_transform(y)

ml_model = RandomForestClassifier(n_estimators=200)
ml_model.fit(X, y_enc)

def predict_specialty_ml(symptoms, severity, duration,sugar):
    if "Hair Loss" in symptoms and sugar < 126:
        return "General Medicine"
    vec = np.zeros(len(SYMPTOM_INDEX))
    for s in symptoms:
        if s in SYMPTOM_INDEX:
            vec[SYMPTOM_INDEX[s]] = 1
    features = np.concatenate([vec, [severity, duration]])
    pred = ml_model.predict([features])[0]
    return le.inverse_transform([pred])[0]
    

# ---------------- PATIENT PANEL ----------------
if st.session_state.role == "Patient":
    st.title(f"🧑‍⚕️ Patient Panel - {st.session_state.username}")
    patient_id = get_patient_id(st.session_state.username)
    st.session_state.patient_id = patient_id
    get_last_risk(patient_id)

    tabs = st.tabs(["Symptoms","Health Data","Risk","AI Advice","Medication","Emergency","History","Chat","Resources"])

    with tabs[0]:
        st.subheader("Select Your Symptoms")
    
        symptom_categories = {
    "Cardiovascular / Chest": [
        "Chest Pain","Palpitations","Shortness of Breath","Swelling in Legs",
        "High BP Symptoms","Irregular Heartbeat","Fainting","Cold Sweats",
        "Jaw Pain","Left Arm Pain","Rapid Pulse","Exercise Intolerance",
        "Orthopnea","Cyanosis","Chest Tightness"
    ],

    "Respiratory": [
        "Cough","Wheezing","Runny Nose","Sore Throat","Chest Congestion",
        "Asthma Attacks","Night Cough","Hemoptysis","Snoring","Sleep Apnea",
        "Rapid Breathing","Hoarseness","Postnasal Drip","Pleural Pain","Dry Cough"
    ],

    "Metabolic / General Health": [
        "Fatigue","Frequent Thirst","Frequent Urination","Weight Gain","Weight Loss",
        "Blurred Vision","Slow Healing","Cold Intolerance","Heat Intolerance",
        "Hair Loss","Excess Hunger","Night Sweats","Tremors","Dehydration","Low Energy"
    ],

    "Neurological / Head": [
        "Headache","Dizziness","Numbness","Memory Loss","Confusion","Seizures",
        "Speech Difficulty","Balance Issues","Tremors Neuro","Vision Loss",
        "Tingling","Sleep Disorders","Behavior Change","Blackouts","Weakness"
    ],

    "Gastrointestinal / Other": [
        "Nausea","Abdominal Pain","Diarrhea","Constipation","Heartburn","Bloating",
        "Vomiting","Blood in Stool","Loss of Appetite","Indigestion","Acid Reflux",
        "Jaundice","Gas","Stomach Cramps","Food Intolerance"
    ]
}

    # Initialize session_state for symptoms if not present
        if "symptoms" not in st.session_state:
            st.session_state.symptoms = []

    # Collect selected symptoms from all categories
        selected_symptoms = []
        for category, symptoms_list in symptom_categories.items():
            selected = st.multiselect(category, symptoms_list, default=[s for s in st.session_state.symptoms if s in symptoms_list])
            selected_symptoms.extend(selected)

    # Update session_state
        st.session_state.symptoms = list(set(selected_symptoms))

    with tabs[1]:
         st.session_state.heart_rate = st.number_input(
            "Heart Rate", 40, 180, st.session_state.get("heart_rate", 72)
        )
         st.session_state.sugar = st.number_input(
             "Blood Sugar", 70, 400, st.session_state.get("sugar", 110)
         )
         st.session_state.sys = st.number_input(
             "Systolic BP", 80, 200, st.session_state.get("sys", 120)
         )
         st.session_state.dia = st.number_input(
             "Diastolic BP", 50, 130, st.session_state.get("dia", 80)
         )
         st.session_state.steps = st.number_input(
             "Steps Today", 0, 20000, st.session_state.get("steps", 3000)
         )

         severity_level = st.selectbox(
             "Symptom Severity", ["Mild","Moderate","Severe"],
             index={"Mild":0,"Moderate":1,"Severe":2}[st.session_state.get("severity_level","Mild")]
         )
         symptom_duration = st.selectbox(
             "Symptom Duration", ["< 1 day","1–3 days","3–7 days","> 1 week"],
             index={"< 1 day":0,"1–3 days":1,"3–7 days":2,"> 1 week":3}[st.session_state.get("symptom_duration","< 1 day")]
         )

         severity_map = {"Mild":1,"Moderate":2,"Severe":3}
         duration_map = {"< 1 day":1,"1–3 days":2,"3–7 days":3,"> 1 week":4}

         st.session_state.severity_score = severity_map[severity_level]
         st.session_state.duration_score = duration_map[symptom_duration]

         st.session_state.severity_level = severity_level
         st.session_state.symptom_duration = symptom_duration

         st.info(
             f"Severity Score: {st.session_state.severity_score} | "
             f"Duration Score: {st.session_state.duration_score}"
         )
    with tabs[2]:
        if st.button("Calculate & Save Risk"):
        # Use severity and duration from Tab 2
            severity_score = st.session_state.severity_score
            duration_score = st.session_state.duration_score

        # Predict specialty using ML model
            specialty_risk = predict_specialty_ml(
                st.session_state.symptoms,
                severity_score,
                duration_score,
                st.session_state.sugar
            )

        # Calculate risk based on health data
            risk, _ = calculate_risk(
                st.session_state.heart_rate,
                st.session_state.sugar,
                st.session_state.sys,
                st.session_state.dia,
                st.session_state.steps,
                st.session_state.symptoms
            )

        # Store results in session state
            st.session_state.risk = risk
            st.session_state.specialty_risk = specialty_risk

        # Get visit count for this patient
            visit_count = pd.read_sql(
                "SELECT COUNT(*) as cnt FROM records WHERE patient_id=?",
                conn, params=(st.session_state.patient_id,)
            ).iloc[0]['cnt'] + 1

        # Insert record into database
            c.execute('''
                INSERT INTO records (
                    patient_id, visit_number, symptoms, severity,
                    heart_rate, sugar, bp, steps,
                    risk, risk_score, specialty_risk, date, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                st.session_state.patient_id,
                visit_count,
                ",".join(st.session_state.symptoms),
                0,  # severity column can remain 0 or store severity_score
                st.session_state.heart_rate,
                st.session_state.sugar,
                f"{st.session_state.sys}/{st.session_state.dia}",
                st.session_state.steps,
                risk,
                0,  # risk_score can be calculated later if needed
                specialty_risk,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                ""
            ))
            conn.commit()

            st.success(f"Health record saved! Visit Number: {visit_count}")
            st.write("Risk Level:", risk)
            st.write("Specialty Risk:", specialty_risk)

    with tabs[3]:  # Tab 4: AI Advice
        st.subheader("AI Health Advice")

        if st.session_state.risk != "":
        # Display risk warning
            if st.session_state.risk == "High":
                st.error("Immediate medical consultation required.")
            elif st.session_state.risk == "Medium":
                st.warning("Lifestyle changes and monitoring advised.")
            else:
                st.success("Maintain a healthy lifestyle.")
        
        # AI advice based on specialty
            tips = {
                "Cardiology": "Avoid heavy salt intake, monitor BP, do light exercises.",
                "Respiratory": "Avoid smoke, monitor breathing, consider inhaler if prescribed.",
                "Endocrinology": "Monitor sugar levels, maintain diet, regular exercise.",
                "Neurology": "Rest well, track headaches or dizziness, consult specialist if worsening.",
                "Gastroenterology": "Eat small frequent meals, avoid irritants, stay hydrated.",
                "General": "Maintain balanced diet, exercise, and regular health checkups."
            }

            specialty = st.session_state.specialty_risk
            if specialty in tips:
                st.info(f"💡 Advice for {specialty}: {tips[specialty]}")

        else:
            st.info("Calculate risk first to receive advice.")

    with tabs[4]:  # ---------------- TAB 4 : MEDICATION ----------------
        st.subheader("💊 Medication Management")

    # ---------------- ADD MEDICATION ----------------
        st.markdown("### ➕ Add Medicine Schedule")

        med_name = st.text_input("Medicine Name")
        days = st.number_input("Number of Days", min_value=1, max_value=30, value=1)
        times_per_day = st.number_input("Times Per Day", min_value=1, max_value=5, value=1)

        med_times = [
            st.time_input(f"Time {i+1}", key=f"med_time_{i}")
            for i in range(times_per_day)
        ]

        if st.button("Add Medicine Schedule") and med_name:
            for day in range(days):
                date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")
                for t in med_times:
                    c.execute("""
                        INSERT INTO medication
                        (patient_id, medicine_name, time, taken, date, confirmation)
                        VALUES (?, ?, ?, 0, ?, 'Pending')
                    """, (
                        st.session_state.patient_id,
                        med_name,
                        t.strftime("%H:%M"),
                        date
                    ))
            conn.commit()
            st.success(f"✅ {med_name} scheduled successfully")

    # ---------------- REMINDER NOTIFICATION ----------------
        st.markdown("### ⏰ Medication Reminder")

        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now()
        start_time = (now - timedelta(minutes=1)).strftime("%H:%M")
        end_time = (now + timedelta(minutes=1)).strftime("%H:%M")

        pending_meds = pd.read_sql("""
            SELECT * FROM medication
            WHERE patient_id=?
            AND date=?
            AND time BETWEEN ? AND ?
            AND confirmation='Pending'
        """, conn, params=(
            st.session_state.patient_id,
            today,
            start_time,
            end_time
        ))

    # ---- Prevent duplicate notifications ----
        if "notified_meds" not in st.session_state:
            st.session_state.notified_meds = []

        for _, row in pending_meds.iterrows():
            if row["id"] not in st.session_state.notified_meds:
                notification.notify(
                    title="💊 Medication Reminder",
                    message=f"Time to take {row['medicine_name']} at {row['time']}",
                    timeout=10
                )
                st.session_state.notified_meds.append(row["id"])
                c.execute(
                    "UPDATE medication SET confirmation='Sent' WHERE id=?",
                    (row["id"],)
                )
                conn.commit()

    # ---------------- TODAY'S MEDICATION LIST ----------------
        st.markdown("### 📅 Today's Medications")

        med_df = pd.read_sql("""
            SELECT * FROM medication
            WHERE patient_id=? AND date=?
            ORDER BY time
        """, conn, params=(
            st.session_state.patient_id,
            today
        ))

        if med_df.empty:
            st.info("No medications scheduled for today.")
        else:
            for _, row in med_df.iterrows():
                col1, col2, col3 = st.columns([3, 2, 2])

                col1.write(f"💊 **{row['medicine_name']}**")
                col2.write(f"⏰ {row['time']}")

                if row["taken"] == 1:
                    col3.success("✔ Taken")
                elif row["confirmation"] == "Missed":
                    col3.error("❌ Missed")
                else:
                    take = col3.button("Taken", key=f"take_{row['id']}")
                    miss = col3.button("Missed", key=f"miss_{row['id']}")

                    if take:
                        c.execute("""
                            UPDATE medication
                            SET taken=1, confirmation='Taken'
                            WHERE id=?
                        """, (row["id"],))
                        conn.commit()
                        st.rerun()

                    if miss:
                        c.execute("""
                            UPDATE medication
                            SET taken=0, confirmation='Missed'
                            WHERE id=?
                        """, (row["id"],))
                        conn.commit()
                        st.rerun()


    with tabs[5]:
        st.subheader("🚨 Emergency Alert System")

    # ---------------- CURRENT HEALTH DATA ----------------
        latest = pd.read_sql("""
            SELECT * FROM records
            WHERE patient_id=?
            ORDER BY date DESC
            LIMIT 1
        """, conn, params=(patient_id,))

        if latest.empty:
            st.warning("No health data available to evaluate emergency.")
        else:
            row = latest.iloc[0]

            st.markdown("### 🩺 Latest Health Readings")
            st.write(f"❤️ Heart Rate: {row['heart_rate']} bpm")
            st.write(f"🩸 Sugar Level: {row['sugar']} mg/dL")
            st.write(f"📟 BP: {row['bp']}")
            st.write(f"⚠️ Risk Level: **{row['risk']}**")

        # ---------------- AUTO EMERGENCY DETECTION ----------------
            emergency_detected = False
            reason = ""

            if row["risk"] == "High":
                emergency_detected = True
                reason = "High risk detected by AI"

            if row["heart_rate"] > 120 or row["heart_rate"] < 45:
                emergency_detected = True
                reason = "Abnormal heart rate detected"

            if row["sugar"] > 250 or row["sugar"] < 60:
                emergency_detected = True
                reason = "Critical sugar level detected"

        # ---------------- AUTO ALERT ----------------
            if emergency_detected:
                st.error("🚨 EMERGENCY CONDITION DETECTED!")
                st.write(f"**Reason:** {reason}")

                if st.button("🚑 Send Emergency Alert"):
                # Save emergency record
                    c.execute("""
                        INSERT INTO emergency
                        (patient_id, risk, heart_rate, sugar, bp, message, date)
                        VALUES (?,?,?,?,?,?,?)
                    """, (
                        patient_id,
                        row["risk"],
                        row["heart_rate"],
                        row["sugar"],
                        row["bp"],
                        reason,
                        datetime.now().strftime("%Y-%m-%d %H:%M")
                    ))
                    conn.commit()

                # Fetch doctor(s) matching the patient's specialty_risk
                doctors_df = pd.read_sql(
                    "SELECT * FROM doctors WHERE LOWER(specialty) LIKE ?",
                     conn,
                     params=(f"%{st.session_state.specialty_risk.lower()}%",)
                 )


        # ---------------- MANUAL EMERGENCY ----------------
            st.markdown("### 🆘 Manual Emergency Trigger")

            manual_msg = st.text_area(
                "Describe the emergency (optional)",
                 placeholder="Patient unconscious, chest pain, etc."
             )

            if st.button("🚨 Trigger Emergency Manually"):
                c.execute("""
                    INSERT INTO emergency
                    (patient_id, risk, heart_rate, sugar, bp, message, date)
                    VALUES (?,?,?,?,?,?,?)
                """, (
                   patient_id,
                   row["risk"],
                   row["heart_rate"],
                   row["sugar"],
                   row["bp"],
                   manual_msg if manual_msg else "Manual emergency triggered",
            datetime.now().strftime("%Y-%m-%d %H:%M")
                ))
                conn.commit()

    # Send to doctor(s)
                doctors_df = pd.read_sql(
                    "SELECT * FROM doctors WHERE LOWER(specialty) LIKE ?",
                     conn,
                     params=(f"%{st.session_state.specialty_risk.lower()}%"
                    ,)
                     )

                if not doctors_df.empty:
                    for _, doc in doctors_df.iterrows():
                        send_email(
                            doc["email"],
                                "🚨 Manual Emergency Triggered for Your Patient",
                                f"""
                                Manual emergency triggered for patient ID: {patient_id}
                                Patient Name: {st.session_state.username}
                                Message: {manual_msg if manual_msg else "No message provided"}
                                Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
                         """
                       )
                    st.success(f"🚑 Manual emergency alert sent to {len(doctors_df)} doctor(s)")
                else:
                    st.warning("No doctor found for the patient's specialty risk!")

    # ---------------- EMERGENCY HISTORY ----------------
        st.markdown("### 📜 Emergency History")

        emergency_df = pd.read_sql("""
            SELECT date, risk, heart_rate, sugar, bp, message
            FROM emergency
            WHERE patient_id=?
            ORDER BY date DESC
        """, conn, params=(patient_id,))

        if emergency_df.empty:
            st.info("No emergency records found.")
        else:
            st.dataframe(emergency_df)

    with tabs[6]:
        st.subheader("🩺 Health History & Predictions")
    
    # Fetch all records for this patient
        history_df = pd.read_sql("SELECT * FROM records WHERE patient_id=? ORDER BY date ASC", conn,params=(patient_id,))
    
        if history_df.empty:
            st.info("No health records available.")
        else:
        # Show full history table
            st.markdown("### 📜 Complete Health Records")
            st.dataframe(history_df[['visit_number','date','symptoms','heart_rate','sugar','bp','steps','risk','specialty_risk']])
        
        # ---------------- CHARTS ----------------
            st.markdown("### 📊 Health Trends")
            chart_df = history_df.set_index("date")[["heart_rate","sugar","steps"]]
            st.line_chart(chart_df)
        
        # ---------------- RISK TREND PREDICTION ----------------
            st.markdown("### 🔮 Predicted Risk for Next 5 Visits")
        
        
        # Encode risk to numeric for prediction
            history_df = history_df.copy()
            history_df['risk_score_numeric'] = history_df['risk'].map({'Low':1, 'Medium':2, 'High':3})
        
            if len(history_df) >3:  # Require at least 4 records to predict
                X = np.arange(len(history_df)).reshape(-1,1)  # visit index
                y = history_df['risk_score_numeric'].values

                model = LinearRegression()
                model.fit(X, y)

            # Predict next 5 visits
                future_idx = np.arange(len(history_df), len(history_df)+5).reshape(-1,1)
                predicted_risk_numeric = model.predict(future_idx)
                predicted_risk_labels = ['Low' if r<1.5 else 'Medium' if r<2.5 else 'High' for r in predicted_risk_numeric]

                st.markdown("### 🔮 Predicted Risk for Next 5 Visits")
                for i, risk in enumerate(predicted_risk_labels, start=1):
                    if risk == "High":
                        st.markdown(f"**Visit {i}:** 🔴 {risk}")
                    elif risk == "Medium":
                        st.markdown(f"**Visit {i}:** 🟠 {risk}")
                    else:
                        st.markdown(f"**Visit {i}:** 🟢 {risk}")

            # Show predicted risk chart
                pred_chart_df = pd.DataFrame({
                    "Predicted Risk": [1 if r=="Low" else 2 if r=="Medium" else 3 for r in predicted_risk_labels],
                    "Visit": np.arange(len(history_df)+1, len(history_df)+6)
                }).set_index("Visit")
                st.line_chart(pred_chart_df)
            
            # Highlight high risk prediction
                if "High" in predicted_risk_labels:
                    st.warning("⚠️ High risk predicted in upcoming visits. Consult your doctor!")
                else:
                    st.info("Not enough data to predict future risk. Add more records.")

        # ---------------- DOWNLOAD HISTORY ----------------
            st.markdown("### 💾 Download Health History")
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{st.session_state.username}_health_history.csv",
                mime='text/csv'
            )

    with tabs[7]:
        st.subheader("💬 Medical Chat")
        patient_id = st.session_state.patient_id

    # Fetch latest chat messages for this patient every time
        chat_df = pd.read_sql(
            "SELECT * FROM chat WHERE patient_id=? ORDER BY timestamp",
            conn,
            params=(patient_id,)
        )

    # Display messages
        if chat_df.empty:
            st.info("No messages yet. Start the conversation with your doctor or AI.")
        else:
            for _, row in chat_df.iterrows():
                sender = row["sender"].strip().lower()
                timestamp = row["timestamp"]
                if sender == "patient":
                    st.markdown(f"🧑 **You [{timestamp}]:** {row['message']}")
                elif sender == "doctor":
                    st.markdown(f"👨‍⚕️ **Doctor [{timestamp}]:** {row['message']}")
                else:
                    st.markdown(f"🤖 **AI [{timestamp}]:** {row['message']}")

        st.markdown("---")

    # Input box
        query = st.text_input("Ask your health question", key="patient_query")
        if st.button("Send", key="send_patient_msg"):
            if query.strip():
            # Save patient message
                c.execute(
                    "INSERT INTO chat (patient_id, sender, message, timestamp) VALUES (?,?,?,?)",
                    (patient_id, "Patient", query.strip(), datetime.now().strftime("%Y-%m-%d %H:%M"))
                )

            # Generate AI response
                ai_reply = ai_health_chat(query, patient_id)

            # Save AI response
                c.execute(
                    "INSERT INTO chat (patient_id, sender, message, timestamp) VALUES (?,?,?,?)",
                    (patient_id, "AI", ai_reply, datetime.now().strftime("%Y-%m-%d %H:%M"))
                )
                conn.commit()
                st.rerun()  # Refresh to show new messages

    with tabs[8]:
        st.subheader("📚 Personalized Health Resources")

        patient_id = st.session_state.patient_id

    # Fetch latest health record
        latest = pd.read_sql("""
            SELECT risk, specialty_risk, heart_rate, sugar, bp
            FROM records
            WHERE patient_id=?
            ORDER BY date DESC
            LIMIT 1
        """, conn, params=(patient_id,))

        if latest.empty:
            st.info("No health data available yet. Please submit your health details.")
            st.stop()

        risk = latest.iloc[0]["risk"]
        specialty = latest.iloc[0]["specialty_risk"]

    # ---------------- RISK STATUS ----------------
        st.markdown("### 🧠 Health Status")
        st.markdown(f"**Specialty Focus:** {specialty}")
        st.markdown(f"**Risk Level:** {risk}")

        if risk == "High":
            st.error("🚨 High risk detected. Immediate medical attention is advised.")
        elif risk == "Medium":
            st.warning("⚠️ Moderate risk detected. Regular monitoring recommended.")
        else:
            st.success("✅ Low risk. Keep maintaining healthy habits.")

    # ---------------- PREVENTIVE TIPS ----------------
        st.markdown("### 🛡️ Preventive Care Tips")

        preventive_tips = {
            "Cardiology": [
                "Reduce salt and oily food intake",
                "Monitor blood pressure daily",
                "Avoid smoking and alcohol",
                "Do light cardio exercises"
            ],
            "Respiratory": [
                "Avoid dust and pollution",
                "Practice deep breathing exercises",
                "Use masks in polluted areas",
                "Stay hydrated"
            ],
            "Neurology": [
                "Ensure adequate sleep",
                "Manage stress levels",
                "Limit screen time",
                "Seek help for frequent headaches"
            ],
            "Endocrinology": [
                "Control sugar intake",
                "Eat small frequent meals",
                "Exercise regularly",
                "Monitor glucose levels"
            ],
            "Gastroenterology": [
                "Avoid spicy and oily food",
                "Eat fiber-rich meals",
                "Drink enough water",
                "Avoid late-night eating"
            ],
            "General": [
                "Maintain balanced diet",
                "Exercise regularly",
                "Stay hydrated",
                "Get routine health checkups"
            ]
        }

        tips = preventive_tips.get(specialty, preventive_tips["General"])
        for tip in tips:
            st.write("•", tip)

    # ---------------- RESOURCE TYPE FILTER ----------------
        st.markdown("### 📖 Educational Resources")
        resource_type = st.selectbox(
            "Choose Resource Type",
            ["Articles", "Videos", "Clinical Guidelines"]
        )

    # ---------------- RESOURCE LINKS ----------------
        resources = {
            "Cardiology": {
                "Articles": [
                    ("Heart Disease Overview", "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)")
                ],
                "Videos": [
                    ("Heart Health Basics", "https://www.youtube.com/watch?v=iU-uE3xvQF0")
                ],
                "Clinical Guidelines": [
                    ("WHO Cardiovascular Guidelines", "https://www.who.int/health-topics/cardiovascular-diseases")
                ]
            },
            "Respiratory": {
                "Articles": [
                    ("Air Pollution & Lungs", "https://www.who.int/health-topics/air-pollution")
                ],
                "Videos": [
                    ("Breathing Exercises", "https://www.youtube.com/watch?v=TgcyiVQnVBs")
                ],
                "Clinical Guidelines": [
                    ("WHO Air Quality Guidelines", "https://www.who.int/publications/i/item/9789240034228")
                ]
            },
            "Endocrinology": {
                "Articles": [
                    ("Diabetes Management", "https://www.who.int/news-room/fact-sheets/detail/diabetes")
                ],
                "Videos": [
                    ("Managing Blood Sugar", "https://www.youtube.com/watch?v=eWHH9je2zG4")
                ],
                "Clinical Guidelines": [
                    ("WHO Diabetes Guidelines", "https://www.who.int/health-topics/diabetes")
                ]
            },
            "Neurology": {
                "Articles": [
                    ( "Neurological Disorders – WHO", "https://www.who.int/health-topics/brain-health" )
                ],
                "Videos": [
                    ("Brain Health Tips", "https://www.youtube.com/watch?v=qPix_X-9t7E")
                ],
                "Clinical Guidelines": [
                    ("WHO Neurology Guidelines", "https://www.who.int/publications/i/item/9789240084278")
                ]
            },
            "Gastroenterology": {
                "Articles": [
                    ("Digestive Health", "https://www.britannica.com/science/gastroenterology")
                ],
                "Videos": [
                    ("Gut Health Explained", "https://www.youtube.com/watch?v=yIoTRGfcMqMsss")
                ],
                "Clinical Guidelines": [
                    ("WHO Digestive Health", "https://www.worldgastroenterology.org/guidelines")
                ]
            }
        }

        selected_resources = resources.get(specialty, {}).get(resource_type, [])

        if not selected_resources:
            st.info("No resources available for this category.")
        else:
            for title, link in selected_resources:
                st.markdown(f"🔗 **{title}**")
                st.markdown(f"[Open Resource]({link})")

    # ---------------- NEARBY HOSPITALS (ONLY IF HIGH RISK) ----------------
        if risk == "High":
            st.markdown("### 🏥 Nearby Clinics / Hospitals")

            hospitals = [
                ("City Health Clinic", "https://maps.google.com/?q=City+Health+Clinic"),
                ("Central Hospital", "https://maps.google.com/?q=Central+Hospital"),
                ("Metro Medical Center", "https://maps.google.com/?q=Metro+Medical+Center"),
                ("Lifecare Hospital", "https://maps.google.com/?q=Lifecare+Hospital"),
                ("Emergency Care Unit", "https://maps.google.com/?q=Emergency+Care+Unit")
            ]

            for name, link in hospitals:
                st.markdown(f"🏥 **{name}** — [View Map]({link})")

# ---------------- DOCTOR PANEL ----------------
# ---------------- DOCTOR PANEL ----------------
elif st.session_state.role == "Doctor":
    st.title(f"Doctor Panel - {st.session_state.username}")

    # ---------------- Specialty Input ----------------
    specialty_input = st.text_input("Enter your Specialty")
    if specialty_input:
        specialty = specialty_input.strip().lower()

        # Load all records with patient names
        df = pd.read_sql("""
            SELECT r.id, r.patient_id, p.name, r.symptoms, r.heart_rate, r.sugar, r.bp, r.steps,
                   r.risk, r.specialty_risk, r.date, r.notes
            FROM records r
            JOIN patients p ON r.patient_id=p.id
        """, conn)

        # Filter only records matching the doctor's specialty
        df_filtered = df[df["specialty_risk"].str.lower() == specialty]

        if not df_filtered.empty:
            # ---------------- Select Patient ----------------
            selected_patient = st.selectbox("Select Patient", df_filtered["name"].unique())
            patient_df = df[df["name"] == selected_patient].sort_values("date")
            latest_record = patient_df.iloc[-1]
            patient_id = latest_record["patient_id"]
            # ---------------- Patient Records ----------------
            st.subheader("Patient Records")
            st.dataframe(patient_df[['id','patient_id','name','symptoms','heart_rate','sugar','bp','steps','risk','specialty_risk','date','notes']])

            # ---------------- Health Trends ----------------
            st.subheader("Health Trends")
            st.line_chart(patient_df.set_index("date")[["heart_rate","sugar","steps"]])

            # ---------------- Notes for Latest Record ----------------
            latest_record = patient_df.iloc[-1]
            note = st.text_area("Enter Notes for Latest Record", value=latest_record["notes"])
            if st.button("Save Note", key=f"save_note_{patient_id}"):
                c.execute("UPDATE records SET notes=? WHERE id=?", (note, latest_record["id"]))
                conn.commit()
                st.success("Note saved successfully!")

            # ---------------- Patient Messages (Chat) -----------------
            st.subheader("💬 Patient Messages")

            # Fetch all chat messages for this patient
            chat_df = pd.read_sql(
                "SELECT * FROM chat WHERE patient_id=? ORDER BY timestamp",
                conn, params=(patient_id,)
            )

            if chat_df.empty:
                st.info("No messages yet from this patient.")
            else:
                # Display chat messages
                for _, row in chat_df.iterrows():
                    timestamp = row['timestamp']
                    sender = str(row["sender"]).strip().lower()
                    if sender == "patient":
                        st.markdown(f"🧑 **Patient [{timestamp}]:** {row['message']}")
                    elif sender == "doctor":
                        st.markdown(f"👨‍⚕️ **Doctor [{timestamp}]:** {row['message']}")
                    else:
                        st.markdown(f"🤖 **AI [{timestamp}]:** {row['message']}")

            # ---------------- Doctor Reply -----------------
            reply = st.text_input("Reply to patient", key=f"reply_{patient_id}")
            if st.button("Send Reply", key=f"send_{patient_id}"):
                if reply.strip():
                    # Insert doctor's message into chat
                    c.execute(
                        "INSERT INTO chat (patient_id, sender, message, timestamp) VALUES (?,?,?,?)",
                        (patient_id, "Doctor", reply.strip(), datetime.now().strftime("%Y-%m-%d %H:%M"))
                    )
                    conn.commit()
                    st.rerun()  # Refresh to show new message immediately

        else:
            st.info("No patients found for this specialty yet.")

# ---------------- ADMIN PANEL ----------------
elif st.session_state.role == "Admin":
    st.title("Admin Panel - Manage Doctors")
    st.subheader("Add Doctor")
    doc_name = st.text_input("Doctor Name", key="doc_name")
    specialty = st.text_input("Specialty", key="specialty")
    city = st.text_input("City", key="city")
    languages = st.text_input("Languages", key="languages")
    experience = st.number_input("Experience (Years)", 0, 50, 5)
    password = st.text_input("Password", type="password", key="doc_pass")
    email = st.text_input("Email", key="doc_email")
    
    if st.button("Add Doctor"):
        if doc_name and specialty and password and email:
            c.execute("INSERT INTO doctors (name, specialty, city, languages, experience, rating, password,email) VALUES (?,?,?,?,?,0,?,?)",
                      (doc_name, specialty, city, languages, experience, password,email))
            conn.commit()
            st.success(f"Doctor {doc_name} added successfully!")

    st.subheader("Remove Doctor")
    doctors_df = pd.read_sql("SELECT * FROM doctors", conn)
    if not doctors_df.empty:
        doc_to_remove = st.selectbox("Select Doctor to Remove", doctors_df["name"])
        if st.button("Remove Doctor"):
            c.execute("DELETE FROM doctors WHERE name=?", (doc_to_remove,))
            conn.commit()
            st.success(f"Doctor {doc_to_remove} removed successfully!")

    st.subheader("All Doctors")
    st.dataframe(doctors_df)

    st.subheader("All Patients (Health Data Included)")

# Join patients with latest record data
    patients_df = pd.read_sql('''
        SELECT p.name,
               r.visit_number,
               r.symptoms,
               r.heart_rate,
               r.sugar,
               r.bp,
               r.steps,
               r.risk,
               r.specialty_risk,
               r.date
        FROM patients p
        LEFT JOIN records r ON p.id = r.patient_id
        WHERE r.id IN (
            SELECT MAX(id) FROM records GROUP BY patient_id
        )
    ''', conn)

    if not patients_df.empty:
        st.dataframe(patients_df)
    else:
        st.info("No patient records available.")

    st.subheader("High Risk Patients")
    records_df = pd.read_sql("SELECT r.id,r.patient_id,p.name,r.symptoms,r.heart_rate,r.sugar,r.bp,r.steps,r.risk,r.specialty_risk,r.date,r.notes FROM records r JOIN patients p ON r.patient_id=p.id", conn)
    st.dataframe(records_df[records_df["risk"]=="High"])

    st.subheader("Risk Distribution")
    st.bar_chart(records_df["risk"].value_counts())

st.markdown("---")
st.caption("MedIntel | Fully Integrated Medical Intelligence System")