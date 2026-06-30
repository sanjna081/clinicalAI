import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "deepseek/deepseek-v4-flash"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}


def call_llm(messages, temperature=0.3):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# LLM CALL #1 — Clinical Analyst Agent

def analyze_dataset(dataset_summary, clinical_context):
    system_prompt = """You are an expert clinical data scientist specializing in
healthcare analytics for Treatment, Payment, and Operations (TPO) use cases.

Your job is to analyze a healthcare dataset summary and recommend the best
machine learning approach.

You must respond ONLY with a valid JSON object. No explanation, no markdown,
no code blocks. Just raw JSON.

The JSON must follow this exact structure:
{
    "task_type": "classification" or "clustering",
    "recommended_model": one of ["logistic_regression", "random_forest", "xgboost", "kmeans", "hdbscan"],
    "target_column": "column name if classification, null if clustering",
    "n_clusters": number if clustering (between 3 and 6), null if classification,
    "plot_columns": ["col_name_1", "col_name_2"],
    "clinical_rationale": "2-3 sentences explaining why this model fits the clinical use case",
    "key_features": ["list", "of", "up to 5", "most relevant", "column names"],
    "evaluation_metric": "the most appropriate metric e.g. AUC-ROC, F1, accuracy",
    "preprocessing_notes": "brief note on any important preprocessing considerations"
}

For plot_columns: pick exactly 2 column names from the dataset that are:
- Genuinely numeric with meaningful clinical values (e.g. age, num_medications, time_in_hospital)
- NOT diagnosis codes, IDs, or columns encoded from text categories
- Columns where differences between values have real clinical meaning
- Columns that would visually show meaningful separation between clusters or groups
- Both columns must exist in the dataset columns list provided"""

    user_prompt = f"""Clinical Context provided by user:
{clinical_context}

Dataset Summary:
{dataset_summary}

Based on the clinical context and dataset characteristics, recommend the best
ML approach. Return only the JSON object."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw = call_llm(messages, temperature=0.2)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "task_type": "classification",
            "recommended_model": "random_forest",
            "target_column": None,
            "n_clusters": None,
            "plot_columns": [],
            "clinical_rationale": "Defaulting to Random Forest as a robust general-purpose classifier.",
            "key_features": [],
            "evaluation_metric": "AUC-ROC",
            "preprocessing_notes": "Standard preprocessing applied.",
        }

    return result


# LLM CALL #2 — Clinical Insight Generator (RAG-powered)

def generate_insights(model_results, dataset_summary, clinical_context,
                      llm_recommendation, retrieved_context=""):
    """
    Generates clinical insights grounded in retrieved knowledge base context.
    retrieved_context comes from rag_engine.retrieve_context().
    If empty, falls back to LLM general knowledge.
    """

    # Build RAG context block
    rag_block = ""
    if retrieved_context:
        rag_block = f"""
RELEVANT CLINICAL GUIDELINES AND FRAMEWORKS (retrieved from knowledge base):
{retrieved_context}

Use the above guidelines to ground your recommendations in evidence-based
standards. Cite the source document when making specific recommendations.
"""

    system_prompt = """You are a senior clinical analytics consultant writing
an insight report for healthcare executives and care managers.

Your job is to translate machine learning results into clear, actionable
clinical insights tied to Treatment, Payment, and Operations (TPO).

Write in plain English. Avoid jargon. Be specific and actionable.
Where relevant, reference the clinical guidelines provided to ground
your recommendations in established standards.

Structure your report with these exact sections:

## Executive Summary
2-3 sentences summarizing the most important finding.

## Key Findings
3-5 bullet points of the most important results.

## Clinical Insights
Detailed interpretation of the model results in the context of the clinical
use case. What do the top features tell us? What patterns emerge?

## TPO Recommendations
Specific recommendations for:
- Treatment: what clinical actions should be taken?
- Payment: what payment integrity or cost implications exist?
- Operations: what operational changes or monitoring should be implemented?

## Limitations & Caveats
2-3 honest limitations of this analysis."""

    user_prompt = f"""Clinical Context:
{clinical_context}

Dataset Summary:
{dataset_summary}

Model Used: {llm_recommendation.get('recommended_model', 'Unknown')}
Clinical Rationale: {llm_recommendation.get('clinical_rationale', '')}

Model Results:
{json.dumps(model_results, indent=2, default=str)}
{rag_block}
Write a comprehensive clinical insight report based on these results."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return call_llm(messages, temperature=0.4)