import streamlit as st
import sqlite3
import pandas as pd

USERNAME = "admin"
PASSWORD = "admin123"

st.title("🔐 Admin Dashboard")

with st.form("admin_login"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login = st.form_submit_button("Login")

if login:
    if username == USERNAME and password == PASSWORD:
        st.success("Login successful ✅")
        
        conn = sqlite3.connect("patient_reports.db")
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                patient_id TEXT,
                dob TEXT,
                gender TEXT,
                timestamp TEXT,
                pdf BLOB
            )
        """)
        conn.commit()

        cursor.execute("SELECT patient_name, patient_id, dob, gender, timestamp FROM reports ORDER BY timestamp DESC")
        data = cursor.fetchall()

        if data:
            st.subheader("📄 Patient Reports Summary")
            df = pd.DataFrame(data, columns=["Patient Name", "Patient ID", "DOB", "Gender", "Prediction Timestamp"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No reports available.")

        conn.close()
    else:
        st.error("Invalid credentials ❌")
