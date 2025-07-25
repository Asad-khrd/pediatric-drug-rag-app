# pediatric-drug-rag-app
A RAG system to analyze and summarize the latest pediatric adverse drug events from the FDA.

# 🔬 Pediatric Drug Safety Explorer

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.2-E1523D?style=for-the-badge&logo=pandas)
![FAISS](https://img.shields.io/badge/FAISS-GPU-blue?style=for-the-badge)
![Google Gemini](https://img.shields.io/badge/Google-Gemini_Pro-4285F4?style=for-the-badge&logo=google)

An advanced, on-demand RAG (Retrieval-Augmented Generation) system that analyzes real-time FDA data to generate safety profiles for drugs used in pediatric populations (ages 0-17).

**Live Demo:** ``

---

## 🎯 Problem Statement

Medical professionals and caregivers need access to the most current, real-world data on adverse drug events in children. Standard resources are often outdated, and public AI models cannot access or analyze this live, specialized data due to knowledge cutoffs and an inability to query specific databases.

This project solves that problem by providing an intelligent tool to query, analyze, and summarize the latest reports from the FDA's Adverse Event Reporting System (FAERS) in real-time.

## ✨ Key Features

* **Live Data Analysis:** Connects directly to the openFDA API to use data from the last 365 days, ensuring insights are timely and relevant.
* **Hybrid RAG Pipeline:** Implements a sophisticated RAG system that uses an LLM to deconstruct user queries, performs a hybrid search using both semantic (FAISS) and structured (Pandas) retrieval, and generates grounded, verifiable summaries.
* **LLM-Powered Query Deconstruction:** A unique feature where a preliminary LLM call analyzes the user's natural language query to separate the core semantic `concept` from structured `filters` (like age, sex, or seriousness).
* **Persona-Based Summaries:** The final output is tailored to the selected audience ("Parent / Caregiver" or "Medical Professional"), demonstrating advanced prompt engineering for targeted communication.
* **GPU-Accelerated:** Utilizes a GPU-powered FAISS index for high-speed semantic search on the in-memory knowledge base during local development.

## ⚙️ Architecture: On-Demand RAG

The application does not rely on a static, pre-built database. Instead, it performs the entire RAG pipeline in real-time for every user request.

User Input (Drug Name, Query, Audience)
│
▼
[data_pipeline.py] -> Fetches live FDA data for the specific drug and pediatric age group.
│
▼
[data_pipeline.py] -> Creates an in-memory Knowledge Base (Pandas DataFrame + FAISS Index).
│
▼
[rag_logic.py] -> LLM Call #1: Deconstructs the user's query into a semantic 'concept' and structured 'filters'.
│
▼
[rag_logic.py] -> Hybrid Retrieval:
1. FAISS semantic search on the 'concept'.
2. Pandas filtering on the DataFrame using the 'filters'.
│
▼
[rag_logic.py] -> LLM Call #2: Generates a persona-based summary using the retrieved, filtered context.
│
▼
[app.py] -> Displays the final summary and structured data on the Streamlit UI.


## 🛠️ Tech Stack

* **Language:** Python
* **AI & Data Science:** Google Gemini Pro (`gemini-1.5-flash`, `text-embedding-004`), FAISS, Pandas, NumPy
* **Web Framework:** Streamlit
* **Data Source:** openFDA API
* **Environment Management:** Conda

## 🚀 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/pediatric-drug-rag-app.git](https://github.com/your-username/pediatric-drug-rag-app.git)
    cd pediatric-drug-rag-app
    ```

2.  **Create and Activate the Conda Environment**
    This single command creates the environment and installs all dependencies.
    ```bash
    conda create --name pedia-rag -c pytorch -c conda-forge faiss-gpu streamlit google-generativeai pandas numpy -y
    conda activate pedia-rag
    ```

3.  **Add Your API Key**
    * Create a file at this exact path: `.streamlit/secrets.toml`.
    * Add your Gemini API key to the file like this:
        ```toml
        GEMINI_API_KEY = "YOUR_API_KEY_HERE"
        ```

4.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
