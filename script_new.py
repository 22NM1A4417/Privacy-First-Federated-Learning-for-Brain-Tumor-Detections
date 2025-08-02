import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from datetime import datetime, date
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import os
import io
import sqlite3
from PyPDF2 import PdfReader, PdfWriter
import pandas as pd

# Constants
USERNAME = "admin"
PASSWORD = "admin123"

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Load model
@st.cache_resource
def load_model_once():
    return load_model('global_model_round5.h5')

model = load_model_once()
class_names = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',
    3: 'Pituitary'
}

# Use tabs instead of sidebar
tab1, tab2 = st.tabs(["🏠 Home", "🔐 Admin Login"])

# ---------------------- HOME TAB ----------------------
with tab1:
    st.title("🧠 Brain Tumor Detection Report")
    st.markdown("Upload an MRI image and get the prediction report.")

    patient_name = st.text_input("Enter Patient Name")
    patient_id = st.text_input("Enter Patient ID")

    col1, col2 = st.columns(2)
    with col1:
        dob = st.date_input("Date of Birth", min_value=date(1960, 1, 1), max_value=date.today())
        dob_str = dob.strftime("%d-%m-%Y")
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(299, 299))
        st.image(img, caption='Uploaded MRI Image', width=300)

        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner('Analyzing image...'):
            preds = model.predict(img_array)
            predicted_index = np.argmax(preds)
            predicted_label = class_names[predicted_index]

        if predicted_label == 'No Tumor':
            st.success("✅ No tumor detected.")
        else:
            st.error("⚠️ Tumor detected.")
            st.markdown(f"""
                <div style="background-color: #fefefe; border-left: 4px solid #d9534f;
                    padding: 10px 14px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
                    width: 60%; margin-top: -5px; margin-bottom: 20px;">
                    <p style="margin: 0; font-size: 16px; color: #d9534f;">
                        <strong>Tumor Type:</strong> {predicted_label}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Tumor Info Section (smaller heading)
            st.markdown(f"<h5>📘 About {predicted_label}</h5>", unsafe_allow_html=True)
            if predicted_label == "Glioma":
                st.info("Gliomas are tumors that originate in the brain’s glial cells. They are often aggressive and require early treatment.")
            elif predicted_label == "Meningioma":
                st.info("Meningiomas grow in the membranes around the brain/spinal cord. Usually benign, but can cause issues due to pressure.")
            elif predicted_label == "Pituitary":
                st.info("Pituitary tumors are located in the pituitary gland, often affecting hormones. Usually treatable through surgery or medication.")

        # PDF Generation
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "🧠 Brain Tumor Detection Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, height - 90, f"Patient Name: {patient_name or 'N/A'}")
        c.drawString(50, height - 110, f"Patient ID: {patient_id or 'N/A'}")
        c.drawString(50, height - 130, f"Date of Birth: {dob_str}")
        c.drawString(50, height - 150, f"Gender: {gender}")
        c.drawString(50, height - 170, f"Prediction Time: {timestamp}")

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 210, "Prediction Summary:")
        c.setFont("Helvetica", 12)
        c.drawString(70, height - 230, f"Tumor Status: {'Detected' if predicted_label != 'No Tumor' else 'No Tumor'}")
        c.drawString(70, height - 250, f"Tumor Type: {predicted_label}")

        img_pil = Image.open(uploaded_file).convert("RGB")
        image_path = "temp_img.jpg"
        img_pil.save(image_path)
        c.drawImage(ImageReader(image_path), 350, height - 370, width=180, height=180)
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, 80, "Disclaimer: This is an AI-generated prediction. Consult a medical professional for confirmation.")

        c.save()
        buffer.seek(0)

        reader = PdfReader(buffer)
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        writer.encrypt(str(dob.year))

        final_pdf = io.BytesIO()
        writer.write(final_pdf)
        final_pdf.seek(0)
        os.remove(image_path)

        if not os.path.exists("reports"):
            os.makedirs("reports")
        filename = f"{patient_id}_{timestamp.replace(':','-').replace(' ','_')}.pdf"
        filepath = os.path.join("reports", filename)
        with open(filepath, "wb") as f:
            f.write(final_pdf.read())
        final_pdf.seek(0)

        # Store in database
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
        cursor.execute("""
            INSERT INTO reports (patient_name, patient_id, dob, gender, timestamp, pdf)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            patient_name,
            patient_id,
            dob_str,
            gender,
            timestamp,
            final_pdf.read()
        ))
        conn.commit()
        conn.close()

        st.download_button(
            label="⬇️ Download PDF Report",
            data=final_pdf,
            file_name=filename,
            mime="application/pdf"
        )

# ---------------------- ADMIN LOGIN TAB ----------------------
with tab2:
    st.header("🔐 Admin Login")

    if not st.session_state.logged_in:
        with st.form("admin_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login = st.form_submit_button("Login")

        if login:
            if username == USERNAME and password == PASSWORD:
                st.success("Login successful ✅")
                st.session_state.logged_in = True
                st.experimental_rerun()
            else:
                st.error("Invalid credentials ❌")

    if st.session_state.logged_in:
        st.subheader("📄 Patient Reports Summary")
        conn = sqlite3.connect("patient_reports.db")
        cursor = conn.cursor()
        cursor.execute("SELECT patient_name, patient_id, dob, gender, timestamp FROM reports ORDER BY timestamp DESC")
        data = cursor.fetchall()
        conn.close()

        if data:
            df = pd.DataFrame(data, columns=["Patient Name", "Patient ID", "DOB", "Gender", "Prediction Timestamp"])
            search = st.text_input("Search by name or ID")
            if search:
                df = df[df["Patient Name"].str.contains(search, case=False) | df["Patient ID"].str.contains(search, case=False)]
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No reports available.")

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()

# ---------------------- Global Disclaimer ----------------------
st.markdown("---")
st.markdown(
    "<p style='font-size: 13px; color: grey;'>"
    "<strong>Disclaimer:</strong> This tool is intended for educational and preliminary analysis purposes only. "
    "It should not replace professional medical advice, diagnosis, or treatment."
    "</p>",
    unsafe_allow_html=True
)
