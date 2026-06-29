# ClinicalAI

![ClinicalAI UI](clinicalAI_UI.png)

An agentic AI system that takes any healthcare dataset, automatically trains the most appropriate machine learning model, and generates clinical insight  reports grounded in US quality standards through a RAG pipeline.

## What It Does

- Upload any healthcare CSV dataset and describe your use case
- Choose between Prediction and Clustering
- The system selects and trains the appropriate model
- A RAG pipeline retrieves relevant guidelines from a curated knowledge base
- Generates a clinical insight report with cited TPO recommendations

## Setup

1. Clone the repository
2. Create a virtual environment and activate it
3. Install dependencies:
   pip install -r requirements.txt
4. Create a .env file in the root folder with your OpenRouter API key:
   OPENROUTER_API_KEY=your_key_here
5. Run the app:
   streamlit run app.py

## Knowledge Base

The RAG pipeline uses five curated documents covering U.S. payer regulatory and quality standards including CMS Program Integrity guidelines, HEDIS measure specifications, Transitional Care Management, and Chronic Care Management requirements.

## Tech Stack

- Streamlit — frontend UI
- DeepSeek V4 Flash via OpenRouter — LLM
- scikit-learn and XGBoost — ML engine
- ChromaDB and sentence-transformers — RAG vector store