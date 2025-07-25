import os
import json
import faiss
import numpy as np
import pandas as pd
import google.generativeai as genai
from config import *

def hybrid_retrieval(user_query: str, df: pd.DataFrame, faiss_index: faiss.Index, reaction_list: list, drug_name: str) -> pd.DataFrame:
    """
    Analyzes the user's query to extract concepts and filters, then performs a hybrid search.
    """
    print("1. Deconstructing user query with AI...")
    # LLM Call #1: Deconstruct the query
    prompt = (
        "You are a medical query parser. Your task is to extract the core medical concept and any demographic or severity filters from a user's question. "
        "Valid filters are 'serious', 'boys', 'girls', 'toddlers' (age 1-3), and 'teens' (age 13-17). "
        "Return your answer as a single, clean JSON object with two keys: 'concept' and 'filters'. "
        f"For example, for the query 'are there serious skin issues in young boys', you should return: {{\"concept\": \"skin issues\", \"filters\": [\"serious\", \"boys\"]}}.\n"
        f"User query: {user_query}"
    )
    
    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL)
        response = model.generate_content(prompt)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        parsed_response = json.loads(json_text)
        concept = parsed_response.get("concept", user_query)
        filters = parsed_response.get("filters", [])
        print(f"   - Concept extracted: '{concept}'")
        print(f"   - Filters found: {filters}")

    except (json.JSONDecodeError, AttributeError, Exception) as e:
        print(f"   - Warning: Could not parse AI response, using full query. Error: {e}")
        concept = user_query
        filters = []

    # Embed the concept and search FAISS
    print("2. Performing semantic search...")
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=concept,
        task_type="RETRIEVAL_QUERY"
    )
    
    query_vector = np.array([result['embedding']], dtype='float32')
    faiss.normalize_L2(query_vector)
    
    _, indices = faiss_index.search(query_vector, 10)
    top_reactions = [reaction_list[i] for i in indices[0]]
    
    # Filter DataFrame based on retrieved reactions and the specific drug
    context_df = df[df['reaction'].isin(top_reactions)].copy()
    context_df = context_df[context_df['drug_name'].str.contains(drug_name, case=False, na=False)]
    
    # Apply structured filters
    print("3. Applying structured filters...")
    for f in filters:
        f_lower = f.lower()
        if f_lower == 'serious':
            context_df = context_df[context_df['is_serious'] == True]
        elif f_lower in ['boy', 'boys', 'male', 'males']:
            context_df = context_df[context_df['sex'] == 'Male']
        elif f_lower in ['girl', 'girls', 'female', 'females']:
            context_df = context_df[context_df['sex'] == 'Female']
        elif f_lower == 'toddlers':
            context_df = context_df[context_df['age'].between(1, 3)]
        elif f_lower == 'teens':
            context_df = context_df[context_df['age'].between(13, 17)]
    
    print(f"   - Final context size after filtering: {len(context_df)} records.")
    return context_df

def generate_summary(context_df: pd.DataFrame, audience: str) -> str:
    """Generates a summary tailored to the audience based on the provided context."""
    
    if context_df.empty:
        return "No specific data was found for this query. This could mean no adverse events matching your criteria have been reported."

    print("4. Generating final summary with AI...")
    context_sample = context_df.head(20).to_dict(orient='records')
    context_json = json.dumps(context_sample, indent=2)
    
    prompt = (
        f"You are a skilled medical communicator. Your audience is: **{audience}**. "
        f"Based *only* on the following JSON data of reported adverse events, write a concise and helpful summary. "
        "Do not use any outside knowledge. If the data is sparse, say so. "
        "Structure your response clearly.\n"
        f"Context data:\n{context_json}\n\n"
        "Summary:"
    )
    
    model = genai.GenerativeModel(GENERATIVE_MODEL)
    # CORRECTED SYNTAX: The prompt is the first argument
    response = model.generate_content(prompt)
    
    return response.text.strip()

# --- Test Block ---
# The existing __main__ block you have is fine, but ensure it's at the end of the file.
# if __name__ == '__main__':
#     from data_pipeline import fetch_data, create_knowledge_base
    
#     api_key_test = os.getenv("GOOGLE_API_KEY")
#     if not api_key_test:
#         print("🔴 ERROR: GOOGLE_API_KEY environment variable not set for the test.")
#     else:
#         print("--- Building a test knowledge base for 'Ibuprofen' ---")
#         reports = fetch_data("Ibuprofen")
#         if not reports:
#             print("❌ Could not fetch data for test.")
#         else:
#             df, index, reactions = create_knowledge_base(reports)
#             if df is not None and index is not None:
#                 print("✅ Test knowledge base created successfully.")
                
#                 test_query = "what are the most serious stomach issues reported in young boys?"
#                 test_audience = "Parent / Caregiver"
                
#                 print(f"\n--- Testing hybrid_retrieval with query: '{test_query}' ---")
#                 context_df = hybrid_retrieval(test_query, df, index, reactions)
                
#                 print("\n--- Retrieved Context DataFrame (Top 5 Rows): ---")
#                 if not context_df.empty:
#                     print(context_df.head())
#                 else:
#                     print("No context was retrieved.")

#                 print(f"\n--- Testing generate_summary for audience: '{test_audience}' ---")
#                 summary = generate_summary(context_df, test_audience)
                
#                 print("\n--- Final AI-Generated Summary: ---")
#                 print(summary)
#             else:
#                 print("❌ Failed to create knowledge base for test.")