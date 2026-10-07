"""
Single-Page Phishing URL Analyzer
ML + Cybersecurity + Explainable AI (XAI) Application
Built with Streamlit, trained Random Forest, and SHAP.
"""

import sys
import os

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np

from src.predict import analyze_url, create_shap_plot
from src.features import FEATURE_NAMES, FEATURE_DESCRIPTIONS, FEATURE_CATEGORIES

# Streamlit Page Configuration
st.set_page_config(
    page_title="Phishing URL Analyzer | ML + Cybersecurity + XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling (Dark Cybersecurity Theme)
st.markdown(
    """
    <style>
    /* Global dark aesthetic */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header hero styling */
    .hero-container {
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.5) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.35rem;
    }
    .badge-tag {
        display: inline-block;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin-right: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Result Badges */
    .result-card-phishing {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
    }
    .result-card-legit {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    }
    .result-title-phishing {
        font-size: 2rem;
        font-weight: 900;
        color: #ef4444;
        margin: 0;
        letter-spacing: 1px;
    }
    .result-title-legit {
        font-size: 2rem;
        font-weight: 900;
        color: #10b981;
        margin: 0;
        letter-spacing: 1px;
    }
    
    /* Top 5 Feature Cards */
    .top-feature-box {
        background: #111827;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #64748b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .top-feature-box.phishing-driver {
        border-left-color: #ef4444;
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.08) 0%, #111827 100%);
    }
    .top-feature-box.legit-driver {
        border-left-color: #10b981;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.08) 0%, #111827 100%);
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.65rem 1.75rem;
        font-size: 1rem;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
        transform: translateY(-1px);
    }
    
    /* Table styling */
    .dataframe {
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown(
    """
    <div class="hero-container">
        <div>
            <span class="badge-tag">Machine Learning</span>
            <span class="badge-tag">Cybersecurity</span>
            <span class="badge-tag">Explainable AI (SHAP)</span>
            <span class="badge-tag">Random Forest 99.7% Acc</span>
        </div>
        <h1 class="hero-title">🛡️ Phishing URL Analyzer</h1>
        <p class="hero-subtitle">
            Inspect any web link in real-time. Features are extracted dynamically, evaluated against a trained Random Forest classifier, and explained using SHAP (Shapley Additive Explanations).
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Pre-populated URL sample selector for quick demo
SAMPLES = {
    "Select a pre-filled sample...": "",
    "🔴 Phishing: IP-Based Host (No Domain)": "http://34.149.138.117/",
    "🔴 Phishing: Fake PayPal Login": "http://paypal-security-update.com/login.php",
    "🔴 Phishing: Workers Dev Deceptive Host": "http://2f3a.reicrut-chat.workers.dev/",
    "🔴 Phishing: Suspicious Account SMS Harvesting": "http://akareal.com.vn/wp-admin/js/hku/sa/netfluinca/chun-g/account/sms2.php",
    "🟢 Legitimate: Google Homepage": "https://www.google.com",
    "🟢 Legitimate: Wikipedia Knowledge Base": "https://www.wikipedia.org",
    "🟢 Legitimate: GitHub Platform": "https://www.github.com",
    "🟢 Legitimate: Southbank Mosaics (From Dataset)": "https://www.southbankmosaics.com"
}

sample_choice = st.selectbox(
    "💡 Or try a pre-configured sample URL:",
    options=list(SAMPLES.keys()),
    index=0
)

# URL Input Field
default_url = SAMPLES[sample_choice] if sample_choice != "Select a pre-filled sample..." else "https://www.wikipedia.org"

col_input, col_btn = st.columns([5, 1])

with col_input:
    url_input = st.text_input(
        "Enter URL to analyze:",
        value=default_url,
        placeholder="e.g. https://example.com/login or http://192.168.1.1/verify",
        label_visibility="collapsed"
    )

with col_btn:
    check_btn = st.button("🔍 Check URL", use_container_width=True)

# Run Analysis
if url_input:
    try:
        with st.spinner("Analyzing URL structure and calculating SHAP contributions..."):
            result = analyze_url(url_input)

        is_phishing = result["is_phishing"]
        confidence = result["confidence"]
        prob_phishing = result["prob_phishing"]
        prob_legit = result["prob_legitimate"]
        top_5 = result["top_5_features"]
        features_dict = result["features"]

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # 1. Result & Probability Banner
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if is_phishing:
                st.markdown(
                    f"""
                    <div class="result-card-phishing">
                        <div style="font-size: 2.5rem; margin-bottom: 0.25rem;">⚠️</div>
                        <h2 class="result-title-phishing">PHISHING DETECTED</h2>
                        <p style="color: #fca5a5; margin-top: 0.5rem; font-size: 1.1rem; font-weight: 500;">
                            High probability of malicious intent, credential harvesting, or deceptive origin.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card-legit">
                        <div style="font-size: 2.5rem; margin-bottom: 0.25rem;">✅</div>
                        <h2 class="result-title-legit">LEGITIMATE URL</h2>
                        <p style="color: #6ee7b7; margin-top: 0.5rem; font-size: 1.1rem; font-weight: 500;">
                            Features match benign web architecture and safe registration profiles.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with res_col2:
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 1.35rem 1.5rem; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.75rem;">
                        <span style="font-weight: 700; color: #f8fafc; font-size: 1.1rem;">Model Confidence</span>
                        <span style="font-size: 1.6rem; font-weight: 800; color: {'#ef4444' if is_phishing else '#10b981'};">{confidence:.1%}</span>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.35rem;">
                            <span>Phishing Risk: <b>{prob_phishing:.1%}</b></span>
                            <span>Legitimate Prob: <b>{prob_legit:.1%}</b></span>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.75rem;">
                        <div style="background: #0f172a; padding: 0.75rem; border-radius: 8px; border: 1px solid {'rgba(239,68,68,0.4)' if is_phishing else '#1e293b'};">
                            <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Phishing Score</span>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #f87171;">{prob_phishing:.1%}</div>
                        </div>
                        <div style="background: #0f172a; padding: 0.75rem; border-radius: 8px; border: 1px solid {'rgba(16,185,129,0.4)' if not is_phishing else '#1e293b'};">
                            <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Legitimate Score</span>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #34d399;">{prob_legit:.1%}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

        # 2. Top 5 Features Influencing the Prediction
        st.subheader("🎯 Top 5 Features Influencing the Prediction")
        st.caption("Derived from local SHAP contribution values for this specific URL.")

        for rank, feat in enumerate(top_5, 1):
            pushes_phishing = feat["shap_value"] > 0
            badge_color = "#ef4444" if pushes_phishing else "#10b981"
            badge_text = "Increases Phishing Risk" if pushes_phishing else "Supports Legitimate"
            icon = "🚨" if pushes_phishing else "🛡️"
            box_class = "phishing-driver" if pushes_phishing else "legit-driver"

            val_display = (
                f"{feat['value']:.4g}"
                if isinstance(feat["value"], float) and not feat["value"].is_integer()
                else f"{int(feat['value'])}"
            )

            st.markdown(
                f"""
                <div class="top-feature-box {box_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <div>
                            <span style="font-weight: 800; font-size: 1.05rem; color: #f8fafc;">#{rank}. {feat['feature']}</span>
                            <span style="background: #1e293b; color: #94a3b8; font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 4px; margin-left: 0.5rem;">{feat['category']}</span>
                        </div>
                        <div>
                            <span style="font-weight: 700; color: #f1f5f9; margin-right: 0.75rem;">Value: <code>{val_display}</code></span>
                            <span style="font-size: 0.8rem; font-weight: 700; color: {badge_color}; background: rgba(0,0,0,0.3); padding: 0.2rem 0.6rem; border-radius: 6px; border: 1px solid {badge_color};">
                                {icon} {badge_text} ({feat['shap_value']:+.4f})
                            </span>
                        </div>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.45;">
                        <b>Security Analysis:</b> {feat['interpretation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

        # 3. SHAP Explanation & Visual Breakdown
        st.subheader("🔬 SHAP Explanation (Explainable AI)")
        st.caption("How each lexical and structural signal pushed the model toward or away from a phishing verdict.")

        shap_col1, shap_col2 = st.columns([3, 2])

        with shap_col1:
            fig = create_shap_plot(result, max_display=10)
            st.pyplot(fig)

        with shap_col2:
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 1.25rem;">
                    <h4 style="margin-top: 0; color: #38bdf8;">How to Interpret SHAP Values:</h4>
                    <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; padding-left: 1.2rem;">
                        <li><b>Baseline Model Prior:</b> Across the dataset, the prior probability for phishing is <b>{result['base_value_phishing']:.1%}</b>.</li>
                        <li><b>Red Bars (+SHAP):</b> Features whose values increased the probability of this URL being classified as <span style="color:#f87171; font-weight:bold;">Phishing</span>.</li>
                        <li><b>Green Bars (-SHAP):</b> Features whose values supported a <span style="color:#34d399; font-weight:bold;">Legitimate</span> diagnosis.</li>
                        <li><b>Magnitude:</b> The length of the bar reflects the relative impact of that feature on this individual decision.</li>
                    </ul>
                    <hr style="border-color: #1e293b; margin: 1rem 0;">
                    <div style="color: #94a3b8; font-size: 0.85rem;">
                        <i>Model: RandomForestClassifier (100 estimators, Gini criterion) trained on 235k PhiUSIIL URLs.</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

        # 4. Collapsible Section for 18 Extracted Features
        with st.expander("📊 View All 18 Extracted Features (Collapsible Table)", expanded=False):
            feature_rows = []
            for name in FEATURE_NAMES:
                val = features_dict[name]
                desc = FEATURE_DESCRIPTIONS.get(name, "")
                cat = FEATURE_CATEGORIES.get(name, "")
                
                # Find shap value if available
                matching_impact = next((i for i in result["all_impacts"] if i["feature"] == name), None)
                shap_str = f"{matching_impact['shap_value']:+.4f}" if matching_impact else "N/A"
                impact_label = matching_impact["direction"] if matching_impact else "N/A"

                feature_rows.append({
                    "Feature": name,
                    "Category": cat,
                    "Extracted Value": val,
                    "SHAP Impact": shap_str,
                    "Tendency": impact_label,
                    "Description": desc
                })

            df_display = pd.DataFrame(feature_rows)
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"Error analyzing URL: {str(e)}")

else:
    st.info("👆 Please enter a URL or select a pre-configured sample above to start analysis.")
