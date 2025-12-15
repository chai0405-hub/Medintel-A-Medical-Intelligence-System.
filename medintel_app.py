# ----------------------------------------------------------
# MedIntel – A Medical Intelligence System (Streamlit App)
# ----------------------------------------------------------

import streamlit as st
import sqlite3
from datetime import datetime

# -----------------------
# Database Setup
# -----------------------
conn = sqlite3.connect("medintel.db")
c = conn.cursor()

# Create tables if not exists
c.execute('''
CREATE TABLE IF NOT EXISTS patients(
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender TEXT,
    contact TEXT
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS doctors(
    doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    specialty TEXT,
    contact TEXT
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS appointments(
    appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT,
    doctor_name TEXT,
    date TEXT,
    time TEXT
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS feedbacks(
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT,
    doctor_name TEXT,
    feedback TEXT
)
''')

conn.commit()

# -----------------------
# Streamlit App
# -----------------------
st.set_page_config(page_title="MedIntel - Medical Intelligence System", layout="wide")
st.title("🩺 MedIntel – Medical Intelligence System")

# Sidebar
menu = ["Patient Portal", "Doctor Information", "Book Appointment", "Feedback"]
choice = st.sidebar.selectbox("Menu", menu)

# -----------------------
# Patient Portal
# -----------------------
if choice == "Patient Portal":
    st.subheader("Patient Information")
    
    with st.form("add_patient_form"):
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=0, max_value=120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        contact = st.text_input("Contact Number")
        submit = st.form_submit_button("Add Patient")
        
        if submit:
            c.execute("INSERT INTO patients (name, age, gender, contact) VALUES (?, ?, ?, ?)", 
                      (name, age, gender, contact))
            conn.commit()
            st.success(f"Patient {name} added successfully!")

    st.subheader("All Patients")
    patients_df = c.execute("SELECT * FROM patients").fetchall()
    st.table(patients_df)

# -----------------------
# Doctor Information
# -----------------------
elif choice == "Doctor Information":
    st.subheader("Find Doctor by Specialty")
    
    specialty = st.text_input("Enter Specialty (e.g., Cardiology, Pediatrics)")
    if st.button("Search"):
        doctors = c.execute("SELECT * FROM doctors WHERE specialty LIKE ?", ('%'+specialty+'%',)).fetchall()
        if doctors:
            st.table(doctors)
        else:
            st.warning("No doctors found for this specialty.")
    
    st.subheader("All Doctors")
    all_doctors = c.execute("SELECT * FROM doctors").fetchall()
    st.table(all_doctors)

# -----------------------
# Book Appointment
# -----------------------
elif choice == "Book Appointment":
    st.subheader("Book an Appointment")
    
    patient_name = st.text_input("Patient Name")
    doctor_name = st.text_input("Doctor Name")
    date = st.date_input("Select Date")
    time = st.time_input("Select Time")
    
    if st.button("Book Appointment"):
        if patient_name and doctor_name:
            c.execute("INSERT INTO appointments (patient_name, doctor_name, date, time) VALUES (?, ?, ?, ?)",
                      (patient_name, doctor_name, str(date), str(time)))
            conn.commit()
            st.success(f"Appointment booked for {patient_name} with Dr. {doctor_name} on {date} at {time}.")
        else:
            st.warning("Please enter both patient and doctor name.")

    st.subheader("All Appointments")
    appointments = c.execute("SELECT * FROM appointments").fetchall()
    st.table(appointments)

# -----------------------
# Feedback
# -----------------------
elif choice == "Feedback":
    st.subheader("Give Feedback")
    
    patient_name = st.text_input("Patient Name")
    doctor_name = st.text_input("Doctor Name")
    feedback_text = st.text_area("Your Feedback")
    
    if st.button("Submit Feedback"):
        if patient_name and doctor_name and feedback_text:
            c.execute("INSERT INTO feedbacks (patient_name, doctor_name, feedback) VALUES (?, ?, ?)",
                      (patient_name, doctor_name, feedback_text))
            conn.commit()
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please fill all the fields.")

    st.subheader("All Feedbacks")
    feedbacks = c.execute("SELECT * FROM feedbacks").fetchall()
    st.table(feedbacks)