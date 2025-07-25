import requests
import datetime  # Add this line
import os
from config import *
import pandas as pd
import numpy as np
import faiss
# If using Google Generative AI embeddings, import the relevant library
# import google.generativeai as genai


def fetch_data(drug_name: str):
    """
    Fetches pediatric adverse event data for a given drug from the openFDA API.
    """
    # 1. CORRECT DATE FORMAT: YYYYMMDD
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=DAYS_BACK)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    # 2. CONSTRUCT THE CORRECT SEARCH QUERY
    search_query = (
        f'patient.drug.medicinalproduct:"{drug_name.lower()}"'
        f'+AND+receiptdate:[{start_str}+TO+{end_str}]'
        f'+AND+patient.patientonsetage:[0+TO+17]'
    )

    api_url = f"https://api.fda.gov/drug/event.json?search={search_query}&limit={REPORT_LIMIT}"

    print(f"Querying API: {api_url}") # Added for debugging

    try:
        # Increased timeout for potentially large queries
        response = requests.get(api_url, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes (like 404 or 500)
        data = response.json()
        return data.get('results', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []


def create_knowledge_base(reports):
    """
    Processes raw JSON reports into a DataFrame, creates REAL embeddings for unique reactions,
    and builds a normalized FAISS index.
    """
    import google.generativeai as genai
    
    processed_data = []
    print("Processing raw reports into a structured format...")
    for report in reports:
        try:
            # Find the primary suspect drug (drugcharacterization == '1')
            primary_drug_obj = next((d for d in report.get('patient', {}).get('drug', []) if d.get('drugcharacterization') == '1'), None)
            
            if not primary_drug_obj:
                continue

            # Safely get all the data points we need
            drug_name = primary_drug_obj.get('medicinalproduct')
            patient_data = report.get('patient', {})
            age = patient_data.get('patientonsetage')
            sex = patient_data.get('patientsex') # '1' is Male, '2' is Female
            
            # Check for any seriousness flag
            is_serious = any(report.get(key) == '1' for key in [
                'seriousnesshospitalization', 'seriousnessdeath', 
                'seriousnesslifethreatening', 'seriousnessdisabling',
                'seriousnesscongenitalanomali', 'seriousnessother'
            ])

            # A single report can have multiple reactions
            for reaction in patient_data.get('reaction', []):
                reaction_term = reaction.get('reactionmeddrapt')
                if reaction_term:
                    processed_data.append({
                        'drug_name': drug_name,
                        'reaction': reaction_term.strip(),
                        'age': age,
                        'sex': 'Female' if sex == '2' else 'Male' if sex == '1' else 'Unknown',
                        'is_serious': is_serious
                    })
        except KeyError as e:
            # This can happen if a report is structured unexpectedly
            print(f"Skipping a malformed report due to missing key: {e}")
            continue

    if not processed_data:
        print("No valid data could be processed.")
        return None, None, None

    df = pd.DataFrame(processed_data)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    unique_reactions = df['reaction'].unique().tolist()
    
    # --- THIS IS THE CORRECTED EMBEDDING STEP ---
    print(f"Creating embeddings for {len(unique_reactions)} unique reactions...")
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=unique_reactions,
            task_type="RETRIEVAL_DOCUMENT",
            title="Pediatric Drug Adverse Event Reactions"
        )
        embeddings = np.array(result['embedding'], dtype='float32')
        
        # Normalize vectors and build the index
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        print("Embeddings and FAISS index created successfully.")
        return df, index, unique_reactions

    except Exception as e:
        print(f"🔴 ERROR: Failed to create embeddings. {e}")
        return df, None, unique_reactions

# if __name__ == "__main__":
#     import google.generativeai as genai

#     # Set your API key from environment variable for this test
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         print("🔴 ERROR: GOOGLE_API_KEY environment variable not set.")
#     else:
#         genai.configure(api_key=api_key)
        
#         # Step 1: Fetch the data
#         drug_to_test = "Amoxicillin"
#         print(f"--- Running test for '{drug_to_test}' ---")
#         reports = fetch_data(drug_to_test)
        
#         if reports:
#             # Step 2: Structure the data into our knowledge base
#             df, index, reactions = create_knowledge_base(reports)
            
#             if df is not None and index is not None:
#                 print("\n✅ Knowledge Base created successfully!")
#                 print(f"Total unique reactions found: {len(reactions)}")
#                 print(f"FAISS index contains {index.ntotal} vectors.")
#                 print("\n--- Sample of the structured DataFrame: ---")
#                 print(df.head()) # Print the first 5 rows of our clean table
#         else:
#             print("\n❌ Fetch failed or no reports found.")