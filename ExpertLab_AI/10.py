import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import io
import google.generativeai as genai
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Configure Google Gemini AI API
os.environ['GOOGLE_API_KEY'] = "AIzaSyAmsRxXrJoDTEzBoUTZaZykpu65IWZHtWI"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return text

# Function to generate a PDF summary
def generate_pdf(title, content):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)

    # Title
    pdf.drawString(100, 750, title)
    y = 730

    # Ensure content is not None
    if content is None:
        content = "No comparison available."

    # Write Content
    for line in content.split("\n"):
        pdf.drawString(100, y, line)
        y -= 20

    pdf.save()
    buffer.seek(0)
    return buffer

# Function to save the graph as a PDF
def save_graph_as_pdf(fig, filename="graph.pdf"):
    pdf = PdfPages(filename)
    pdf.savefig(fig, bbox_inches="tight")
    pdf.close()

# Ensure session state variables exist
for key in ['extracted_text', 'extracted_text2', 'Compare', 'medical_data', 'df_filtered']:
    if key not in st.session_state:
        st.session_state[key] = None

# Streamlit UI Setup
st.set_page_config(page_title="AI Lab Report Analyzer", page_icon="📑", layout="wide")
st.markdown(
    """
    <style>
           .sidebar .stButton>button {
            transition: all 0.3s ease-in-out;
            border: 2px solid transparent;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
        }
        
           .sidebar .stButton>button:hover {
            background: linear-gradient(90deg, #a1c4fd, #c2e9fb);
            color: #000000;
            border: 2px solid #ffffff;
            transform: scale(1.1) rotate(1deg);
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        }

        .stFileUploader {
            transition: all 0.2s ease-in-out;
            border: 2px solid transparent;
            padding: 10px;
            border-radius: 10px;
            background: transparent;
            font-size: 14px;
        }
        .stFileUploader:hover {
            transform: scale(1.01);
            box-shadow: 0px 6px 18px rgba(0, 0, 0, 0.25);
        }
        .stFileUploader:focus-visible {
            outline: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📑 AI-Powered Lab Report Analyzer")

# Sidebar Navigation
with st.sidebar:
    st.header("🔍 Navigation")
    if st.button("🏠 Home"):
        st.session_state.clear()
    if st.button("📄 Summarize"):
        st.session_state['page'] = 'summarize'
    if st.button("💬 Chat"):
        st.session_state['page'] = 'chat'
    if st.button("Compare"):
        st.session_state['page'] = 'compare'
    if st.button("❌ Exit"):
        st.stop()

if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Home Page
if st.session_state['page'] == 'home':
    st.subheader("Welcome to the AI Lab Report Analyzer! 🩺")
    st.write("This tool helps you analyze lab reports, summarize findings, and identify potential risks.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image("C:/Users/VENKAT/Downloads/image1.png", use_container_width=True)
    
    with col2:
        st.subheader("About This Application:")
        st.write("**Key Features:**")
        st.write("- Extract meaningful insights from lab reports.")
        st.write("- Summarize complex information into simple language.")
        st.write("- Detect and analyze potential health risks.")
        st.write("- Chat with AI to gain more insights about your report.")

# Summary Page
if st.session_state['page'] == 'summarize':
    uploaded_file = st.file_uploader("📄 Upload a PDF Lab Report", type=["pdf"])
    
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
        st.write("📝 Extracting text...")
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.session_state['extracted_text'] = extracted_text
    
    if 'extracted_text' in st.session_state:
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
        input_text = f"Summarize this lab report: {st.session_state['extracted_text']}"
        summary = model.generate_content(input_text)
        st.session_state['summary'] = summary.text
        
        st.subheader("📝 Lab Report Summary")
        st.markdown(summary.text)
        
        if st.button("🔬 Show Possible Disease Risks"):
            disease = model.generate_content(summary.text + " What are the possible disease risks?")
            st.session_state['disease_text'] = disease.text
        
        if 'disease_text' in st.session_state:
            with st.expander("⚠ Possible Disease Risks"):
                st.markdown(st.session_state['disease_text'])
        
        # ✅ Visualize Button
        if st.button("📊 Visualize Medical Data"):
            with st.spinner("Extracting medical data..."):
                prompt = f"""
                Extract the following details from this medical lab report:
                1. Test Name
                2. Normal Range (ALWAYS provide a valid numeric range, like "13.5-17.5". If unknown, use average values from medical literature.)
                3. Current Value
                4. Unit
                
                Respond with **only** valid JSON in this format:
                [
                    {{"Test Name": "Hemoglobin", "Current Value": 12.2, "Normal Range": "13.5-17.5", "Unit": "g/dL"}},
                    {{"Test Name": "WBC", "Current Value": 8500, "Normal Range": "4000-11000", "Unit": "cells/µL"}}
                ]
                Lab Report:
                {st.session_state['extracted_text']}
                """
                values = model.generate_content(prompt)

            try:
                medical_data = json.loads(values.text.strip())
                if not isinstance(medical_data, list):
                    raise ValueError("Response is not a valid JSON list")
            except (json.JSONDecodeError, ValueError):
                json_start = values.text.find("[")
                json_end = values.text.rfind("]")
                if json_start != -1 and json_end != -1:
                    json_fixed = values.text[json_start:json_end + 1]
                    try:
                        medical_data = json.loads(json_fixed)
                    except json.JSONDecodeError:
                        medical_data = []

            if medical_data:
                st.session_state['medical_data'] = medical_data
                df = pd.DataFrame(medical_data)
                st.write("### 📋 Extracted Medical Data:", df)
                df.to_csv("medical_data.csv", index=False)
                st.success("✅ CSV file saved successfully!")

                # 🔹 Data Processing
                df["Current Value"] = df["Current Value"].astype(str)
                df_filtered = df[~df["Current Value"].str.contains("/", regex=False)].copy()
                df_filtered[["Normal Low", "Normal High"]] = df_filtered["Normal Range"].str.extract(r'([\d.]+)-([\d.]+)').astype(float)
                df_filtered["Current Value"] = df_filtered["Current Value"].astype(float)
                st.session_state['df_filtered'] = df_filtered

        # 🔹 Allow user to select features to display
        if 'df_filtered' in st.session_state and st.session_state['df_filtered'] is not None:
            df_filtered = st.session_state['df_filtered']
            features_to_display = st.multiselect(
                "Select features to display:",
                options=df_filtered["Test Name"].unique(),
                default=df_filtered["Test Name"].unique()
            )

            # Filter the dataframe based on selected features
            df_filtered = df_filtered[df_filtered["Test Name"].isin(features_to_display)]

            # 🔹 Visualization
            if not df_filtered.empty:
                plt.figure(figsize=(10, 6))
                sns.barplot(x="Test Name", y="Normal High", data=df_filtered, color="lightgreen", label="Normal High")
                sns.barplot(x="Test Name", y="Normal Low", data=df_filtered, color="lightblue", label="Normal Low")
                sns.scatterplot(x=df_filtered["Test Name"], y=df_filtered["Current Value"], color="red", s=150, label="Current Value", zorder=3)

                plt.xticks(rotation=45)
                plt.ylabel("Values")
                plt.title("Test Values vs Normal Ranges")
                plt.legend()
                plt.grid(axis="y", linestyle="--", alpha=0.7)

                st.pyplot(plt)

                # 🔹 Download Graph as PDF
                if st.button("📥 Download Graph as PDF"):
                    with st.spinner("Saving graph as PDF..."):
                        save_graph_as_pdf(plt.gcf(), filename="medical_graph.pdf")
                        st.success("✅ Graph saved as PDF!")
                        with open("medical_graph.pdf", "rb") as file:
                            st.download_button(
                                label="📥 Download PDF",
                                data=file,
                                file_name="medical_graph.pdf",
                                mime="application/pdf"
                            )
            else:
                st.warning("No data available for the selected features.")

        # ✅ Download Summary as PDF
        pdf_buffer = generate_pdf(st.session_state['summary'], st.session_state.get('disease_text', 'No risks detected.'))
        st.download_button(
            label="📥 Download summary",
            data=pdf_buffer,
            file_name="Lab_Report_Summary.pdf",
            mime="application/pdf"
        )


# Chatbot Page
if st.session_state['page'] == 'chat':
    st.subheader("💬 Chat with AI")
    if st.button("🔙 Back to Summary"):
        st.session_state['page'] = 'summarize'
        st.rerun()
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    for chat in st.session_state["chat_history"]:
        st.chat_message(chat["role"]).write(chat["message"])
    
    chat_input = st.chat_input("ask me anything related to the clinical report")
    
    if chat_input:
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
        prompt = f"Lab summary: {st.session_state.get('summary', '')}, {st.session_state.get('disease_text', '')}\nUser: {chat_input}\nAI: strictly answer only if it is present in the clinical report, you are not allowed to answer to questions that are not related to lab report. given that you are a robot which should only follow these parameters "
        response = model.generate_content(prompt)
        
        st.session_state["chat_history"].append({"role": "User", "message": chat_input})
        st.session_state["chat_history"].append({"role": "Assistant", "message": response.text})
        
        st.chat_message("User").write(chat_input)
        st.chat_message("Assistant").write(response.text)


if st.session_state['page'] == 'compare':
    st.title("📊 Lab Report Comparison")

    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')

    uploaded_file = st.file_uploader("📄 Upload a PDF Lab Report for comparison", type=["pdf"])
    uploaded_file2 = st.file_uploader("📄 Upload another PDF Lab Report for comparison", type=["pdf"])

    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
        st.write("📝 Extracting Text...")
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.session_state['extracted_text'] = extracted_text

    if uploaded_file2 is not None:
        st.session_state['uploaded_file2'] = uploaded_file2
        extracted_text2 = extract_text_from_pdf(uploaded_file2)
        st.session_state['extracted_text2'] = extracted_text2

    if uploaded_file and uploaded_file2:
        if 'Compare' not in st.session_state or st.session_state['Compare'] is None:
            compare_input = f"Compare these lab reports and describe whether health has improved or worsened, including date references: {st.session_state['extracted_text'], st.session_state['extracted_text2']}"
            compare_model = model.generate_content(compare_input)
            st.session_state['Compare'] = compare_model.text

    if 'Compare' in st.session_state and st.session_state['Compare']:  
        st.subheader("📑 Lab Report Comparison Result")
        st.markdown(st.session_state['Compare'])

        # Generate PDF
        compare_pdf = generate_pdf("Lab Report Comparison", st.session_state['Compare'])

        # Download Button
        st.download_button(
            label="📥 Download Comparison Report",
            data=compare_pdf,
            file_name="Lab_Report_Comparison.pdf",
            mime="application/pdf"
        )