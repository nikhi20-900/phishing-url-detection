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

# Custom Styling (Refined Dark Cybersecurity Aesthetic)
st.markdown(
    """
    <style>
    /* Global modern dark typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header hero styling */
    .hero-container {
        padding: 1.25rem 1.5rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(30, 41, 59, 0.45) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 0.85rem;
        backdrop-filter: blur(10px);
    }
    .hero-badges {
        margin-bottom: 0.45rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 750;
        color: #f8fafc;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.4px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0;
    }
    .badge-tag {
        display: inline-block;
        font-size: 0.72rem;
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.25);
        margin-right: 0.4rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* Visual status metadata line */
    .tech-status-line {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.65rem;
        font-size: 0.78rem;
        color: #64748b;
        padding: 0.35rem 0.75rem;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        margin-bottom: 1.15rem;
    }
    .tech-status-line b {
        color: #94a3b8;
    }
    .status-sep {
        color: #334155;
    }

    /* Unified URL input & check button */
    div[data-testid="stTextInput"] > div > div > input {
        height: 46px !important;
        background-color: #0f172a !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
        font-size: 0.95rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stTextInput"] > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
    }

    div.stButton > button:first-child {
        height: 46px !important;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff;
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.25);
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.45);
        border-color: #60a5fa;
        transform: translateY(-1px);
    }

    /* Refined Result Cards */
    .result-card-phishing {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(185, 28, 28, 0.04) 100%);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 12px;
        padding: 1.35rem 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.12);
    }
    .result-card-legit {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.04) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 12px;
        padding: 1.35rem 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.12);
    }
    .result-tag-phishing {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        color: #ef4444;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-tag-legit {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        color: #10b981;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-title-phishing {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f87171;
        margin: 0;
        letter-spacing: -0.2px;
    }
    .result-title-legit {
        font-size: 1.75rem;
        font-weight: 800;
        color: #34d399;
        margin: 0;
        letter-spacing: -0.2px;
    }
    .result-desc {
        color: #94a3b8;
        margin-top: 0.4rem;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    /* Probability Container */
    .prob-container {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.35rem 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .prob-meter-bg {
        width: 100%;
        height: 7px;
        background: #1e293b;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0 0.85rem 0;
    }
    .prob-meter-phish {
        height: 100%;
        background: linear-gradient(90deg, #f87171, #ef4444);
        border-radius: 4px;
    }
    .prob-meter-legit {
        height: 100%;
        background: linear-gradient(90deg, #34d399, #10b981);
        border-radius: 4px;
    }
    
    /* Compact Top 5 Feature Cards */
    .top-feature-box {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 0.65rem 0.95rem;
        margin-bottom: 0.45rem;
        border-left: 3px solid #64748b;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }
    .top-feature-box.phishing-driver {
        border-left-color: #ef4444;
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.05) 0%, #0f172a 100%);
    }
    .top-feature-box.legit-driver {
        border-left-color: #10b981;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.05) 0%, #0f172a 100%);
    }
    .feat-rank {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748b;
        margin-right: 0.35rem;
    }
    .feat-name {
        font-weight: 700;
        font-size: 0.92rem;
        color: #f1f5f9;
    }
    .feat-category {
        background: rgba(148, 163, 184, 0.1);
        color: #94a3b8;
        font-size: 0.7rem;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        margin-left: 0.4rem;
    }
    .feat-val code {
        background: rgba(0, 0, 0, 0.35) !important;
        color: #cbd5e1 !important;
        font-size: 0.82rem !important;
        padding: 0.15rem 0.4rem !important;
        border-radius: 4px !important;
    }
    .feat-badge {
        font-size: 0.74rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 5px;
    }
    .badge-phish {
        color: #f87171;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .badge-legit {
        color: #34d399;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .feat-note {
        color: #94a3b8;
        font-size: 0.83rem;
        line-height: 1.4;
        margin-top: 0.25rem;
    }

    /* Section Subheaders */
    h3 {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.2rem !important;
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
        <div class="hero-badges">
            <span class="badge-tag">Machine Learning</span>
            <span class="badge-tag">Cybersecurity</span>
            <span class="badge-tag">Explainable AI (SHAP)</span>
            <span class="badge-tag">Random Forest • 99.7% Acc</span>
        </div>
        <h1 class="hero-title">🛡️ Phishing URL Analyzer</h1>
        <p class="hero-subtitle">
            Inspect web links in real-time. Features are extracted dynamically, evaluated against a trained Random Forest classifier, and explained using SHAP (Shapley Additive Explanations).
        </p>
    </div>
    <div class="tech-status-line">
        <span>⚡ <b>ML Prediction:</b> Random Forest Classifier (100 Trees)</span>
        <span class="status-sep">•</span>
        <span>🔬 <b>XAI:</b> Local TreeExplainer Attribution</span>
        <span class="status-sep">•</span>
        <span>🛡️ <b>Signals:</b> 18 Lexical & Structural Features</span>
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

# URL Input Field & Unified Button Bar
default_url = SAMPLES[sample_choice] if sample_choice != "Select a pre-filled sample..." else "https://www.wikipedia.org"

col_input, col_btn = st.columns([5, 1], gap="small", vertical_alignment="bottom")

with col_input:
    url_input = st.text_input(
        "Enter URL to analyze:",
        value=default_url,
        placeholder="e.g. https://example.com/login or http://192.168.1.1/verify",
        label_visibility="collapsed"
    )

with col_btn:
    check_btn = st.button("🔍 Check URL", width="stretch")

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

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # 1. Result & Probability Cards (Balanced Hierarchy)
        res_col1, res_col2 = st.columns([1, 1], gap="medium")

        with res_col1:
            if is_phishing:
                st.markdown(
                    f"""
                    <div class="result-card-phishing">
                        <div><span class="result-tag-phishing">⚠️ Malicious Pattern Detected</span></div>
                        <h2 class="result-title-phishing">PHISHING URL</h2>
                        <p class="result-desc">
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
                        <div><span class="result-tag-legit">✅ Safe Profile Verified</span></div>
                        <h2 class="result-title-legit">LEGITIMATE URL</h2>
                        <p class="result-desc">
                            Structural and lexical patterns match authentic, trusted web architecture.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with res_col2:
            meter_class = "prob-meter-phish" if is_phishing else "prob-meter-legit"
            meter_width = confidence * 100
            val_color = "#f87171" if is_phishing else "#34d399"

            st.markdown(
                f"""
                <div class="prob-container">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <span style="font-weight: 600; color: #94a3b8; font-size: 0.95rem;">Prediction Probability</span>
                            <span style="font-size: 1.5rem; font-weight: 800; color: {val_color};">{confidence:.1%}</span>
                        </div>
                        <div class="prob-meter-bg">
                            <div class="{meter_class}" style="width: {meter_width:.1f}%;"></div>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem; margin-top: 0.5rem;">
                        <div style="background: #0f172a; padding: 0.65rem 0.85rem; border-radius: 8px; border: 1px solid {'rgba(239,68,68,0.35)' if is_phishing else 'rgba(255,255,255,0.05)'};">
                            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Phishing Score</div>
                            <div style="font-size: 1.15rem; font-weight: 750; color: #f87171;">{prob_phishing:.1%}</div>
                        </div>
                        <div style="background: #0f172a; padding: 0.65rem 0.85rem; border-radius: 8px; border: 1px solid {'rgba(16,185,129,0.35)' if not is_phishing else 'rgba(255,255,255,0.05)'};">
                            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Legitimate Score</div>
                            <div style="font-size: 1.15rem; font-weight: 750; color: #34d399;">{prob_legit:.1%}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # 2. Top 5 Features Influencing the Prediction (Compact Cards)
        st.subheader("🎯 Top 5 Features Influencing the Prediction")
        st.caption("Derived from local SHAP contribution values for this specific URL.")

        for rank, feat in enumerate(top_5, 1):
            pushes_phishing = feat["shap_value"] > 0
            badge_class = "badge-phish" if pushes_phishing else "badge-legit"
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
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center;">
                            <span class="feat-rank">#{rank}</span>
                            <span class="feat-name">{feat['feature']}</span>
                            <span class="feat-category">{feat['category']}</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.6rem;">
                            <span class="feat-val"><code>{val_display}</code></span>
                            <span class="feat-badge {badge_class}">
                                {icon} {badge_text} ({feat['shap_value']:+.4f})
                            </span>
                        </div>
                    </div>
                    <div class="feat-note">
                        {feat['interpretation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # 3. SHAP Explanation & Visual Breakdown
        st.subheader("🔬 SHAP Explanation (Explainable AI)")
        st.caption("How each lexical and structural signal pushed the model toward or away from a phishing verdict.")

        shap_col1, shap_col2 = st.columns([3, 2], gap="medium")

        with shap_col1:
            fig = create_shap_plot(result, max_display=10)
            st.pyplot(fig)

        with shap_col2:
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 1.25rem;">
                    <h4 style="margin-top: 0; color: #38bdf8; font-size: 1rem; font-weight: 700;">How to Interpret SHAP Values:</h4>
                    <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; padding-left: 1.15rem; margin-bottom: 0.75rem;">
                        <li><b>Baseline Model Prior:</b> Prior probability across dataset for phishing is <b>{result['base_value_phishing']:.1%}</b>.</li>
                        <li><b>Red Bars (+SHAP):</b> Signals that increased the probability of a <span style="color:#f87171; font-weight:bold;">Phishing</span> verdict.</li>
                        <li><b>Green Bars (-SHAP):</b> Signals that supported a <span style="color:#34d399; font-weight:bold;">Legitimate</span> verdict.</li>
                        <li><b>Bar Length:</b> Indicates relative feature contribution weight for this specific URL.</li>
                    </ul>
                    <hr style="border-color: rgba(255, 255, 255, 0.08); margin: 0.75rem 0;">
                    <div style="color: #64748b; font-size: 0.8rem;">
                        <i>Model: RandomForestClassifier (100 estimators, Gini criterion) trained on 235k PhiUSIIL URLs.</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

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
                width="stretch",
                hide_index=True
            )

    except Exception as e:
        st.error(f"Error analyzing URL: {str(e)}")

else:
    st.info("👆 Please enter a URL or select a pre-configured sample above to start analysis.")
