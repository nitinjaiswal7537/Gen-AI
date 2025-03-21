import streamlit as st
import PyPDF2
import pandas as pd
import os
import pymssql
import chromadb
import asyncio
from datetime import datetime
from sentence_transformers import SentenceTransformer
from ai21 import AI21Client
from ai21.models.chat import ChatMessage

# Set AI21 API Key
AI21_API_KEY = "2cumBEggCPBC2tY9XN9tLkJeLGZhNCPk"
os.environ["AI21_API_KEY"] = AI21_API_KEY
client = AI21Client() if AI21_API_KEY else None

# Initialize ChromaDB client
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="medical_docs")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# UI Layout
st.set_page_config(page_title="Medical Insights System", layout="wide")
st.title("Multi-Doc RAG System with SQL Integration")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            font-size: 16px;
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
            padding: 10px;
        }
        .stMarkdown {
            font-size: 16px;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with options
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to:", ["Home", "Upload Document", "Claim Insights", "Patient History Queries"])

# File uploader
if page == "Upload Document":
    st.subheader("Upload Medical Policy PDF")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    if pdf_file:
        with st.spinner("Processing PDF..."):
            pdf = PyPDF2.PdfReader(pdf_file)
            pdf_text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            pdf_chunks = [" ".join(pdf_text.split()[i:i+512]) for i in range(0, len(pdf_text.split()), 512)]
            for i, chunk in enumerate(pdf_chunks):
                text_embedding = embedding_model.encode(chunk).tolist()
                collection.add(documents=[chunk], embeddings=[text_embedding], ids=[f"pdf_{i}_{datetime.now().timestamp()}"])
            st.success("PDF successfully processed and stored!")

# Database connection function
def get_db_connection():
    return pymssql.connect(
        server='demo-trial.database.windows.net', 
        user='server@demo-trial', 
        password='abcd123@', 
        database='demo-trial'
    )

# Fetch data
def fetch_sql_data(query, params=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Error fetching data from database: {str(e)}")
        return []

# Query input
if page == "Home":
    st.subheader("Ask Medical Queries")
    predefined_queries = [
        "What is the most common disease?",
        "Show me the most given drug details.",
        "Brief about the medical policy",
        "Retrieve latest diagnosis details."
    ]
    query = st.selectbox("Select a query", ["Enter custom query..."] + predefined_queries)
    if query == "Enter custom query...":
        query = st.text_input("Type your query:")
    if query:
        with st.spinner("Retrieving information..."):
            query_embedding = embedding_model.encode(query).tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=5)
            relevant_docs = "\n".join(results["documents"][0]) if results["documents"] else "No relevant documents found."
            sql_response = fetch_sql_data("SELECT * FROM Drug_Dispense_and_Diagnosis_Report WHERE Diagnosis LIKE %s", [f"%{query}%"])
            sql_response_text = "\n".join([" ".join(map(str, row)) for row in sql_response]) if sql_response else "No matching records found."
            if client:
                try:
                    response = client.chat.completions.create(
                        model="jamba-instruct-preview",
                        messages=[ChatMessage(
                            role="user",
                            content=f"User Query: {query}\n\nDatabase Data:\n{sql_response_text}\n\nProvide a structured response."
                        )],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    st.subheader("Response")
                    st.markdown(f"**Query:** {query}")
                    st.markdown(f"**Database Information:**\n{sql_response_text}")
                    st.markdown(f"**AI Response:**\n{response.choices[0].message.content}")
                except Exception as e:
                    st.error(f"Error generating AI21 response: {str(e)}")
            else:
                st.error("AI21 API key is not set or client initialization failed.")

# Claim Insights
if page == "Claim Insights":
    st.subheader("Claim Insights")
    claim_query = st.text_input("Enter claim-related query:")
    if claim_query:
        claim_data = fetch_sql_data("SELECT * FROM Claims WHERE Description LIKE %s", [f"%{claim_query}%"])
        if claim_data:
            df = pd.DataFrame(claim_data, columns=["ClaimID", "PatientID", "Amount", "Status", "Date"])
            st.table(df)
        else:
            st.info("No claims found matching your query.")

# Patient History Queries
if page == "Patient History Queries":
    st.subheader("Retrieve Patient History")
    patient_id = st.text_input("Enter Patient ID:")
    if patient_id:
        patient_data = fetch_sql_data("SELECT * FROM Patient_History WHERE PatientID = %s", [patient_id])
        if patient_data:
            df = pd.DataFrame(patient_data, columns=["PatientID", "Diagnosis", "Treatment", "Doctor", "Date"])
            st.table(df)
        else:
            st.info("No history found for the given Patient ID.")

# Fix for Streamlit and asyncio compatibility
def fix_asyncio_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

fix_asyncio_event_loop()
