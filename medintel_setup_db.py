import sqlite3

DB_FILE = "medintel.db"

conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# ===================== PATIENTS =====================
c.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    age INTEGER,
    city TEXT,
    language TEXT,
    password TEXT NOT NULL
)
""")

# ===================== DOCTORS =====================
c.execute("""
CREATE TABLE IF NOT EXISTS doctors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    specialty TEXT,
    city TEXT,
    languages TEXT,
    experience INTEGER,
    rating REAL DEFAULT 0,
    password TEXT NOT NULL,
    email TEXT
)
""")

# ===================== RECORDS =====================
c.execute("""
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
""")

# ===================== MEDICATION =====================
c.execute("""
CREATE TABLE IF NOT EXISTS medication (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    medicine_name TEXT,
    time TEXT,
    taken INTEGER DEFAULT 0,
    date TEXT,
    confirmation TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
""")

# ===================== EMERGENCY =====================
c.execute("""
CREATE TABLE IF NOT EXISTS emergency (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    risk TEXT,
    heart_rate INTEGER,
    sugar INTEGER,
    bp TEXT,
    message TEXT,
    date TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
""")

# ===================== CHAT =====================
c.execute("""
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    sender TEXT,
    message TEXT,
    timestamp TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
""")

conn.commit()
conn.close()

print("====================================")
print("✅ medintel.db CREATED SUCCESSFULLY")
print("====================================")