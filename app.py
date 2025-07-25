import streamlit as st
import os
import google.generativeai as genai

from data_pipeline import fetch_data, create_knowledge_base
from rag_logic import hybrid_retrieval, generate_summary

# Configure Google Generative AI with Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.set_page_config(page_title="Pediatric Drug Safety Explorer", page_icon="🔬")
st.title("🔬 Pediatric Drug Safety Explorer")

# --- User Inputs ---
drug_name = st.text_input("Enter the drug name:")
user_question = st.text_input(
    "What do you want to know?",
    placeholder="e.g., what are the most common skin issues in toddlers?"
)
audience = st.radio(
    "Select your audience:",
    ["Parent / Caregiver", "Medical Professional"]
)

if st.button("Generate Analysis"):
    # First, check for inputs
    if not drug_name or not user_question:
        st.warning("Please enter both a drug name and a question.")
    
    # If inputs are present, run the whole pipeline
    else: 
        with st.spinner("Running analysis. This may take a few moments..."):
            
            reports = fetch_data(drug_name)
            if not reports:
                st.error("No reports found for this drug. Please try another drug name.")
            else:
                df, faiss_index, reaction_list = create_knowledge_base(reports)
                if df is None or faiss_index is None:
                    st.error("Failed to process the data for this drug.")
                else:
                    context_df = hybrid_retrieval(user_question, df, faiss_index, reaction_list, drug_name)
                    summary = generate_summary(context_df, audience)
                    
                    st.subheader("Summary")
                    st.write(summary)
                    
                    with st.expander("View Retrieved Data Context"):
                        st.dataframe(context_df)