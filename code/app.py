import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json

from agent import analyze_dataset, generate_insights
from ml_engine import run_model
from rag_engine import build_knowledge_base, retrieve_context, build_retrieval_query

# PAGE CONFIG

st.set_page_config(
    page_title="ClinicalAI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CUSTOM CSS

st.markdown("""
<style>
/* ── Global ── */
.stApp {
    background-color: #FFFFFF;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Hide sidebar toggle ── */
[data-testid="collapsedControl"] {
    display: none;
}

/* ── Main content ── */
.main .block-container {
    padding: 3rem 2rem 2rem 2rem;
    max-width: 860px;
}

/* ── Page title ── */
h1 {
    color: #1A1D2E !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    text-align: center;
}

h2, h3 {
    color: #1A1D2E !important;
    font-weight: 600 !important;
}

/* ── Mode selection cards ── */
.mode-card {
    background: #FFFFFF;
    border: 2px solid #E8ECF4;
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    height: 100%;
}

.mode-card:hover {
    border-color: #6C5CE7;
    box-shadow: 0 4px 20px rgba(108, 92, 231, 0.12);
}

.mode-card.selected {
    border-color: #6C5CE7;
    background: #F5F3FF;
    box-shadow: 0 4px 20px rgba(108, 92, 231, 0.15);
}

.mode-icon {
    font-size: 2.2rem;
    margin-bottom: 0.75rem;
}

.mode-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1A1D2E;
    margin-bottom: 0.4rem;
}

.mode-desc {
    font-size: 0.82rem;
    color: #8A94A6;
    line-height: 1.5;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border: 1px solid #E8ECF4;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.75rem;
    font-weight: 500;
    color: #8A94A6;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1A1D2E;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C5CE7, #8B7CF6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    transition: opacity 0.2s ease;
    width: 100%;
}

.stButton > button[kind="primary"]:hover {
    opacity: 0.9;
}

/* ── Secondary buttons ── */
.stButton > button:not([kind="primary"]) {
    background-color: #FFFFFF;
    border: 2px solid #E8ECF4;
    border-radius: 10px;
    color: #1A1D2E;
    font-weight: 600;
    font-size: 0.875rem;
    padding: 0.6rem 1.2rem;
    transition: all 0.2s ease;
    width: 100%;
}

.stButton > button:not([kind="primary"]):hover {
    border-color: #6C5CE7;
    color: #6C5CE7;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #F4F6FB;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #E8ECF4;
    gap: 2px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.875rem;
    color: #8A94A6;
    padding: 0.5rem 1.25rem;
}

.stTabs [aria-selected="true"] {
    background-color: #6C5CE7 !important;
    color: white !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background-color: #FAFAFA;
    border: 2px dashed #C8D0E0;
    border-radius: 10px;
    padding: 0.5rem;
}

/* ── Text area ── */
textarea {
    border-radius: 8px !important;
    border: 1px solid #E8ECF4 !important;
    font-size: 0.875rem !important;
    background-color: #FFFFFF !important;
}

/* ── Select box ── */
[data-testid="stSelectbox"] > div > div {
    border-radius: 8px !important;
    border: 1px solid #E8ECF4 !important;
}

/* ── Status box ── */
[data-testid="stStatusWidget"] {
    border-radius: 10px;
    border: 1px solid #E8ECF4;
}

/* ── Divider ── */
hr {
    border-color: #E8ECF4;
    margin: 1.5rem 0;
}

/* ── Caption ── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #8A94A6 !important;
    font-size: 0.8rem !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background-color: #FFFFFF;
    border: 1px solid #6C5CE7;
    color: #6C5CE7;
    border-radius: 8px;
    font-weight: 500;
}

[data-testid="stDownloadButton"] > button:hover {
    background-color: #6C5CE7;
    color: white;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E8ECF4;
}

/* ── Input label ── */
.input-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #8A94A6;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
}

/* ── Section divider with label ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #8A94A6;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    margin-top: 1.25rem;
}
</style>
""", unsafe_allow_html=True)

# KNOWLEDGE BASE — load once

@st.cache_resource(show_spinner="Loading clinical knowledge base...")
def load_knowledge_base():
    return build_knowledge_base()

collection = load_knowledge_base()

# SESSION STATE

if "mode" not in st.session_state:
    st.session_state.mode = None

# CONSTANTS

MISSING_VALUES = [
    "?", "NA", "N/A", "n/a", "na", "NULL", "null",
    "None", "none", "NaN", "nan", "", " ", "--", "-",
    "unknown", "Unknown", "UNKNOWN", "missing", "Missing"
]

# HEADER

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; margin-bottom: 0.25rem;">
    <span style="display:inline-block; width:10px; height:10px;
                 background:#6C5CE7; border-radius:50%;
                 margin-right:8px; vertical-align:middle;"></span>
    <span style="font-size:0.8rem; font-weight:600; color:#8A94A6;
                 text-transform:uppercase; letter-spacing:0.1em;">
        Healthcare Analytics Platform
    </span>
</div>
""", unsafe_allow_html=True)

st.title("ClinicalAI")
st.markdown("""
<p style="text-align:center; color:#8A94A6; font-size:0.95rem; margin-top:0.25rem;">
    Upload a healthcare dataset. Get predictive models and clinical insights grounded in CMS and HEDIS guidelines.
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# MODE SELECTION

if st.session_state.mode is None:
    st.markdown("""
    <p style="text-align:center; font-size:0.85rem; font-weight:600;
              color:#8A94A6; text-transform:uppercase; letter-spacing:0.08em;
              margin-bottom:1rem;">
        Choose Analysis Type
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <style>
        div[data-testid="stButton"][id="pred_wrap"] > button,
        .big-card-btn-pred > button {
            height: 220px !important;
            white-space: normal !important;
            background: #FFFFFF !important;
            border: 2px solid #E8ECF4 !important;
            border-radius: 16px !important;
            color: #1A1D2E !important;
            font-size: 0.95rem !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
            padding: 1.5rem !important;
        }
        .big-card-btn-pred > button:hover {
            border-color: #6C5CE7 !important;
            box-shadow: 0 4px 20px rgba(108,92,231,0.12) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="big-card-btn-pred">', unsafe_allow_html=True)
        if st.button("\n\nPrediction\n\nTrain a classification model to predict readmission risk, disease likelihood, or payment status.", key="btn_prediction"):
            st.session_state.mode = "prediction"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <style>
        .big-card-btn-clus > button {
            height: 220px !important;
            white-space: normal !important;
            background: #FFFFFF !important;
            border: 2px solid #E8ECF4 !important;
            border-radius: 16px !important;
            color: #1A1D2E !important;
            font-size: 0.95rem !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
            padding: 1.5rem !important;
        }
        .big-card-btn-clus > button:hover {
            border-color: #6C5CE7 !important;
            box-shadow: 0 4px 20px rgba(108,92,231,0.12) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="big-card-btn-clus">', unsafe_allow_html=True)
        if st.button("\n\nClustering\n\nSegment patients into distinct groups based on clinical and utilization patterns to support population health management.", key="btn_clustering"):
            st.session_state.mode = "clustering"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# INPUT FORM - shown after mode is selected

mode = st.session_state.mode
mode_label = "Prediction" if mode == "prediction" else "Clustering"
mode_icon = "📈" if mode == "prediction" else "🔵"

# Mode indicator + back button
col_back, col_mode = st.columns([1, 4])
with col_back:
    if st.button("Back"):
        st.session_state.mode = None
        st.rerun()
with col_mode:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; padding:0.5rem 0;">
        <span style="font-size:1.3rem">{mode_icon}</span>
        <span style="font-weight:700; color:#1A1D2E; font-size:1rem">{mode_label} Mode</span>
        <span style="background:#EDE9FD; color:#6C5CE7; font-size:0.7rem;
                     font-weight:600; padding:2px 8px; border-radius:20px;
                     text-transform:uppercase; letter-spacing:0.05em;">Active</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Dataset upload ──
st.markdown('<div class="section-label">Upload Dataset</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    label_visibility="collapsed"
)
st.caption("Supported format: CSV")

# ── Target column — only for prediction ──
target_col = None
if mode == "prediction" and uploaded_file:
    st.markdown('<div class="section-label">Target Column</div>', unsafe_allow_html=True)
    try:
        df_preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        column_names = ["Let AI decide"] + list(df_preview.columns)
        selected = st.selectbox(
            "Select target column",
            column_names,
            label_visibility="collapsed"
        )
        target_col = None if selected == "Let AI decide" else selected
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Clinical context
st.markdown('<div class="section-label">Clinical Context</div>', unsafe_allow_html=True)
placeholder = (
    "e.g. This is hospital EHR data. I want to predict 30-day readmission risk."
    if mode == "prediction"
    else "e.g. This is hospital EHR data. I want to segment patients into distinct groups based on clinical characteristics."
)
clinical_context = st.text_area(
    "Clinical context",
    placeholder=placeholder,
    height=110,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)
run_button = st.button("Run Analysis", type="primary")

# HELPERS

def build_dataset_summary(df):
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "numeric_stats": df.describe().round(2).to_dict(),
        "categorical_columns": list(df.select_dtypes(include=["object"]).columns),
        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
    }
    return json.dumps(summary, default=str)


def split_insight_report(report):
    tpo_marker = "## TPO Recommendations"
    if tpo_marker not in report:
        return report, ""
    parts = report.split(tpo_marker)
    before_tpo = parts[0].strip()
    tpo_and_after = parts[1]
    next_section = tpo_and_after.find("\n## ")
    if next_section != -1:
        tpo_content = tpo_and_after[:next_section].strip()
        after_tpo = tpo_and_after[next_section:].strip()
        main_report = before_tpo + "\n\n" + after_tpo
    else:
        tpo_content = tpo_and_after.strip()
        main_report = before_tpo
    return main_report, tpo_content


def plot_feature_importance(feature_importances):
    features = list(feature_importances.keys())[:10]
    scores = list(feature_importances.values())[:10]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    colors = ['#6C5CE7' if i == 0 else '#A29BFE' if i < 3 else '#DFD8FD'
              for i in range(len(features))]
    ax.barh(features[::-1], scores[::-1], color=colors[::-1], height=0.6)
    ax.set_xlabel("Importance Score", color='#8A94A6', fontsize=10)
    ax.set_title("Top 10 Feature Importances", color='#1A1D2E',
                 fontsize=12, fontweight='600', pad=15)
    ax.tick_params(colors='#636E72', labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color('#E8ECF4')
    ax.spines["bottom"].set_color('#E8ECF4')
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap=sns.light_palette("#6C5CE7", as_cmap=True),
        ax=ax, linewidths=0.5, linecolor='#F4F6FB'
    )
    ax.set_xlabel("Predicted", color='#636E72', fontsize=10)
    ax.set_ylabel("Actual", color='#636E72', fontsize=10)
    ax.set_title("Confusion Matrix", color='#1A1D2E',
                 fontsize=12, fontweight='600', pad=15)
    ax.tick_params(colors='#636E72', labelsize=9)
    plt.tight_layout()
    return fig


def plot_cluster_distribution(cluster_labels):
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    colors = ['#6C5CE7', '#A29BFE', '#DFD8FD', '#EFEDFB']
    fig = px.bar(
        x=[f"Cluster {i}" for i in cluster_counts.index],
        y=cluster_counts.values,
        labels={"x": "Cluster", "y": "Number of Patients"},
        title="Patient Distribution Across Clusters",
        color=[f"Cluster {i}" for i in cluster_counts.index],
        color_discrete_sequence=colors
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#636E72', size=11),
        title_font=dict(color='#1A1D2E', size=13),
        xaxis=dict(showgrid=False, linecolor='#E8ECF4'),
        yaxis=dict(gridcolor='#F4F6FB', linecolor='#E8ECF4'),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    return fig


def plot_cluster_scatter(X_pca, cluster_labels):
    if X_pca is None:
        return None
    plot_df = X_pca.copy()
    plot_df["Cluster"] = [f"Cluster {l}" for l in cluster_labels]
    fig = px.scatter(
        plot_df,
        x="PCA Component 1",
        y="PCA Component 2",
        color="Cluster",
        title="Patient Clusters (PCA Projection)",
        color_discrete_sequence=['#6C5CE7', '#A29BFE', '#00B894', '#FDCB6E'],
        opacity=0.65
    )
    fig.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#636E72', size=11),
        title_font=dict(color='#1A1D2E', size=13),
        xaxis=dict(gridcolor='#F4F6FB', linecolor='#E8ECF4'),
        yaxis=dict(gridcolor='#F4F6FB', linecolor='#E8ECF4'),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    return fig


# MAIN PIPELINE

if run_button:
    if not uploaded_file:
        st.error("Please upload a CSV file.")
        st.stop()

    if not clinical_context.strip():
        st.error("Please describe your clinical use case.")
        st.stop()

    # Switch to wide layout for results
    st.markdown("""
    <style>
    .main .block-container { max-width: 1200px; }
    </style>
    """, unsafe_allow_html=True)

    df = pd.read_csv(uploaded_file, na_values=MISSING_VALUES, keep_default_na=True)

    if len(df) > 10000:
        df = df.sample(10000, random_state=42)
        st.info("Dataset trimmed to 10,000 rows for performance.")

    status = st.status("Running ClinicalAI Pipeline...", expanded=True)

    with status:
        st.write("Step 1: Analyzing dataset with AI...")
        dataset_summary = build_dataset_summary(df)

        # Inject mode into context so LLM knows what task to do
        mode_hint = (
            "The user has selected PREDICTION mode. Recommend a classification model."
            if mode == "prediction"
            else "The user has selected CLUSTERING mode. Recommend a clustering model."
        )
        enriched_context = f"{clinical_context}\n\n{mode_hint}"

        try:
            llm_recommendation = analyze_dataset(dataset_summary, enriched_context)
        except Exception as e:
            st.error(f"LLM Call #1 failed: {e}")
            st.stop()
        st.write("Step 1 complete - model selected.")

        # Override model type based on user selection
        if mode == "prediction" and llm_recommendation.get("task_type") == "clustering":
            llm_recommendation["task_type"] = "classification"
            llm_recommendation["recommended_model"] = "random_forest"

        if mode == "clustering" and llm_recommendation.get("task_type") == "classification":
            llm_recommendation["task_type"] = "clustering"
            llm_recommendation["recommended_model"] = "kmeans"
            target_col = None

        if target_col is None and llm_recommendation.get("target_column"):
            target_col = llm_recommendation["target_column"]
            if target_col not in df.columns:
                target_col = None

        recommended_model = llm_recommendation.get("recommended_model", "random_forest")
        n_clusters = llm_recommendation.get("n_clusters", 4) or 4

        st.write(f"Step 2: Training {recommended_model.replace('_', ' ').title()}...")
        try:
            results = run_model(
                model_name=recommended_model,
                df=df,
                target_col=target_col,
                n_clusters=n_clusters
            )
        except Exception as e:
            st.error(f"ML Engine failed: {e}")
            st.stop()
        st.write("Step 2 complete - model trained.")

        st.write("Step 3: Retrieving relevant clinical guidelines...")
        try:
            retrieval_query = build_retrieval_query(
                results, clinical_context, llm_recommendation
            )
            retrieved_context = retrieve_context(collection, retrieval_query)
        except Exception as e:
            st.warning(f"RAG retrieval failed, continuing without context: {e}")
            retrieved_context = ""
        st.write("Step 3 complete - guidelines retrieved.")

        st.write("Step 4: Generating clinical insights...")
        try:
            insight_report = generate_insights(
                model_results=results,
                dataset_summary=dataset_summary,
                clinical_context=clinical_context,
                llm_recommendation=llm_recommendation,
                retrieved_context=retrieved_context
            )
        except Exception as e:
            st.error(f"LLM Call #2 failed: {e}")
            st.stop()
        st.write("Step 4 complete - insights ready.")
        status.update(label="ClinicalAI Pipeline Complete!", state="complete")

    main_report, tpo_section = split_insight_report(insight_report)

    # Summary metrics
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Mode", mode_label)
    c2.metric("Model", recommended_model.replace("_", " ").title())
    c3.metric("Records Analyzed", f"{len(df):,}")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("AI Model Recommendation Details"):
        st.json(llm_recommendation)

    if retrieved_context:
        with st.expander("Retrieved Clinical Guidelines"):
            st.text(retrieved_context)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Explorer",
        "Model Results",
        "Clinical Insights",
        "Recommendations"
    ])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Sample Records")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Statistical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        task_type = llm_recommendation.get("task_type", "classification")

        if task_type == "classification":
            st.subheader(f"Model: {results.get('model')}")
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", results.get("accuracy"))
            col2.metric("AUC-ROC", results.get("auc"))
            col3.metric("F1 Score", results.get("f1"))
            st.markdown("<br>", unsafe_allow_html=True)
            col_left, col_right = st.columns(2)
            with col_left:
                if results.get("feature_importances"):
                    st.pyplot(plot_feature_importance(results["feature_importances"]))
            with col_right:
                if results.get("confusion_matrix"):
                    st.pyplot(plot_confusion_matrix(results["confusion_matrix"]))
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("Full Classification Report"):
                st.text(results.get("classification_report", ""))

        elif task_type == "clustering":
            st.subheader(f"Model: {results.get('model')}")
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.metric("Clusters Found", results.get("n_clusters"))
            col2.metric("Total Patients", f"{len(results.get('cluster_labels', [])):,}")
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(
                plot_cluster_distribution(results["cluster_labels"]),
                use_container_width=True
            )
            fig_scatter = plot_cluster_scatter(
                results.get("X_pca"),
                results["cluster_labels"]
            )
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
            with st.expander("Cluster Profiles"):
                profiles = results.get("cluster_profiles", {})
                if profiles:
                    st.dataframe(
                        pd.DataFrame(profiles).T.round(2),
                        use_container_width=True
                    )

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(main_report)

    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Actionable recommendations grounded in clinical guidelines for Treatment, Payment, and Operations.")
        st.markdown("<br>", unsafe_allow_html=True)
        if tpo_section:
            st.markdown(tpo_section)
        else:
            st.info("No TPO recommendations were generated. Try re-running the analysis.")
        st.divider()
        st.download_button(
            label="Download Full Report",
            data=insight_report,
            file_name="clinical_insight_report.txt",
            mime="text/plain"
        )