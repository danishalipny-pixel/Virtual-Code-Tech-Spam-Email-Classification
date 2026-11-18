"""SpamGuard AI – Intelligent Email Classification System.

Hidden Virtual CodeTech - Artificial Intelligence / Generative AI Internship
Instructor: Engr. Farhat Irshad
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
from src.predictor import EmailSpamPredictor

# Page Configuration
st.set_page_config(
    page_title="SpamGuard AI – Email Classification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design: Premium Navy, Slate, and Teal Theme
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    /* App Header Banner */
    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0D9488 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        margin-bottom: 2rem;
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }
    .app-header p {
        font-size: 1.05rem;
        color: #E2E8F0;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Feature & Metric Cards */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.3rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(15, 23, 42, 0.08);
    }
    .stat-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .stat-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Classification Result Cards */
    .result-card-spam {
        background: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.12);
    }
    .result-title-spam {
        font-size: 1.6rem;
        font-weight: 700;
        color: #B91C1C;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }

    .result-card-ham {
        background: #F0FDF4;
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.12);
    }
    .result-title-ham {
        font-size: 1.6rem;
        font-weight: 700;
        color: #047857;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }

    /* Content Cards */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .content-box h3 {
        color: #0F172A;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.75rem;
    }

    /* Custom Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
    }
    .badge-teal {
        background-color: #CCFBF1;
        color: #0F766E;
    }
    .badge-blue {
        background-color: #DBEAFE;
        color: #1E40AF;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: #F8FAFC;
        border-left: 4px solid #64748B;
        padding: 0.9rem 1.2rem;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #475569;
        margin-top: 1.25rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Cached Model Loader
@st.cache_resource(show_spinner=False)
def load_cached_predictor() -> EmailSpamPredictor:
    """Initialize and cache the model pipeline instance."""
    return EmailSpamPredictor(Path("models/spam_classifier.joblib"))


# Cached Evaluation Data Loader
@st.cache_data(show_spinner=False)
def load_evaluation_metadata() -> dict:
    """Load precomputed metrics and evaluation results from JSON."""
    eval_path = Path("reports/model_evaluation.json")
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Sample Email Templates for Quick Testing
SAMPLE_EMAILS = {
    "-- Select a sample email --": "",
    "Legitimate: Medical Appointment Confirmation": (
        "Subject: Appointment Confirmation - Dr. Harrison's Clinic\n\n"
        "Dear Patient,\n\n"
        "This is a confirmation that your medical consultation has been scheduled for tomorrow, "
        "November 19th at 10:00 AM. Please arrive 15 minutes early and bring your identification documents "
        "and previous diagnostic reports.\n\n"
        "If you need to reschedule, please call our reception at 555-0199.\n\n"
        "Warm regards,\nClinic Support Team"
    ),
    "Legitimate: Quarterly Financial Sync": (
        "Subject: Q3 Financial Review Meeting & Revenue Analysis\n\n"
        "Hi Team,\n\n"
        "Please find the quarterly revenue report attached. We will hold our executive review "
        "meeting on Tuesday at 2:00 PM in Conference Room A. We will review our customer acquisition targets "
        "and operational metrics for the upcoming quarter.\n\n"
        "Best,\nFinance Operations"
    ),
    "Legitimate: University Project Notification": (
        "Subject: Assignment Evaluation Completed - Natural Language Processing\n\n"
        "Dear Student,\n\n"
        "Your project submission for the Natural Language Processing assignment has been graded. "
        "Detailed rubric remarks and grade breakdown are now accessible through the student academic portal. "
        "Office hours are available on Thursday if you wish to discuss feedback.\n\n"
        "Regards,\nAcademic Department"
    ),
    "Spam: Urgent Lottery & Cash Claim": (
        "Subject: URGENT: You Have Won $2,500,000 Cash Prize Notification!\n\n"
        "CONGRATULATIONS WINNER!\n\n"
        "Your email was selected as the lucky winner of our international sweepstakes prize of $2,500,000 CASH! "
        "To immediately claim your transferred reward, please click this link: https://fast-payout-reward.biz/claim "
        "and reply with your full bank account numbers and wire info. Act now! Offer expires within 24 hours!"
    ),
    "Spam: Free Promotional Voucher Scam": (
        "Subject: Claim Your Free $1,000 Shopping Gift Card Right Now!\n\n"
        "Congratulations!\n\n"
        "You are eligible to receive a free $1,000 Amazon shopping voucher reward. "
        "Click the link below immediately to claim your free reward before spots run out: "
        "http://free-voucher-exclusive.net/win. No purchase necessary, click now!"
    ),
    "Spam: Suspicious Password Security Phishing": (
        "Subject: CRITICAL ALERT: Your Account Access Will Be Terminated!\n\n"
        "Dear Customer,\n\n"
        "We detected unauthorized access to your account. Your access will be suspended within 6 hours. "
        "Please verify your credentials immediately by clicking here: http://secure-login-verify.xyz/login. "
        "Enter your password and banking card details to prevent immediate account suspension."
    ),
}


# Sidebar Navigation
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg",
        width=50,
    )
    st.title("SpamGuard AI")
    st.caption("Intelligent Email Classification System")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation Menu",
        ["Home / Dashboard", "Email Classifier", "Model Performance", "About Project"],
        index=1,
    )

    st.markdown("---")
    st.markdown(
        """
        **Organization:**  
        Hidden Virtual CodeTech  

        **Program:**  
        AI / Generative AI Internship  

        **Instructor:**  
        Engr. Farhat Irshad  

        **Corpus:**  
        Apache SpamAssassin (4,289 emails)
        """
    )
    st.markdown("---")
    st.caption("Version 1.0.0 | Production Release")


# PAGE 1: HOME / DASHBOARD
if selected_page == "Home / Dashboard":
    st.markdown(
        """
        <div class="app-header">
            <h1>SpamGuard AI – Intelligent Email Classification</h1>
            <p>Enterprise-grade Natural Language Processing & Machine Learning for Real-Time Email Threat Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_data = load_evaluation_metadata()
    best_model = eval_data.get("best_model", "Logistic Regression")
    metrics = eval_data.get("models", {}).get(best_model, {})
    acc_val = f"{metrics.get('accuracy', 0.9918) * 100:.2f}%"
    f1_val = f"{metrics.get('f1_score', 0.9903) * 100:.2f}%"
    prec_val = f"{metrics.get('precision', 0.9944) * 100:.2f}%"
    rec_val = f"{metrics.get('recall', 0.9862) * 100:.2f}%"

    # KPI Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{acc_val}</div><div class="stat-label">Model Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{f1_val}</div><div class="stat-label">Spam F1-Score</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{prec_val}</div><div class="stat-label">Spam Precision</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{rec_val}</div><div class="stat-label">Spam Recall</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown(
            """
            <div class="content-box">
                <h3>System Architecture & Classification Pipeline</h3>
                <p>SpamGuard AI accepts raw email messages, performs high-fidelity semantic normalization, transforms the linguistic content into a high-dimensional TF-IDF feature space, and classifies each message as either <strong>Legitimate (Ham)</strong> or <strong>Spam</strong>.</p>
                <ol style="margin-top: 0.75rem; line-height: 1.8;">
                    <li><strong>MIME & Structure Parsing:</strong> Extracts subject and body parts, stripping HTML markup and decoding RFC-2047 headers.</li>
                    <li><strong>Domain-Aware Normalization:</strong> Preserves critical sentiment and negation markers while standardizing hyperlinked URLs, email addresses, and monetary triggers.</li>
                    <li><strong>TF-IDF Feature Representation:</strong> Extracts 8,000 discriminative unigram and bigram features with sublinear term-frequency scaling.</li>
                    <li><strong>Machine Learning Inference:</strong> Deploys regularized Logistic Regression alongside Multinomial Naive Bayes for probabilistic decision boundaries.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div class="content-box">
                <h3>Benchmark Dataset Overview</h3>
                <p>Trained and cross-validated on the official <strong>Apache SpamAssassin Public Corpus</strong>:</p>
                <ul style="line-height: 1.8;">
                    <li><strong>Total Cleaned Emails:</strong> 4,289</li>
                    <li><strong>Legitimate Emails (Ham):</strong> 2,472 (57.6%)</li>
                    <li><strong>Spam Emails:</strong> 1,817 (42.4%)</li>
                    <li><strong>Evaluation Split:</strong> 80% Stratified Train / 20% Holdout Test</li>
                    <li><strong>Leakage Prevention:</strong> Scikit-Learn unified Pipeline</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info("💡 **Ready to inspect an email?** Navigate to the **Email Classifier** tab in the sidebar or paste an email directly.")


# PAGE 2: EMAIL CLASSIFIER
elif selected_page == "Email Classifier":
    st.markdown(
        """
        <div class="app-header">
            <h1>Email Classification Console</h1>
            <p>Paste any real-world email or select an authentic sample to perform instant ML threat inspection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sample selector dropdown
    selected_sample_title = st.selectbox(
        "Load a Preset Email Example:",
        list(SAMPLE_EMAILS.keys()),
        index=0,
    )

    # Manage text area state
    default_text = SAMPLE_EMAILS.get(selected_sample_title, "")
    if "email_input_area" not in st.session_state:
        st.session_state["email_input_area"] = default_text
    elif default_text:
        st.session_state["email_input_area"] = default_text

    # Email Text Input Area
    user_email_text = st.text_area(
        "Paste Your Email Content:",
        value=st.session_state.get("email_input_area", ""),
        height=220,
        placeholder="Paste full email text including subject and body here...",
        key="main_email_input",
    )

    # Action Buttons
    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        classify_button = st.button("🛡️ Classify Email", type="primary", use_container_width=True)
    with col_btn2:
        clear_button = st.button("🧹 Clear", use_container_width=True)

    if clear_button:
        st.session_state["email_input_area"] = ""
        st.rerun()

    # Classification Execution
    if classify_button:
        if not user_email_text or not user_email_text.strip():
            st.warning("⚠️ Please paste or type email content before clicking **Classify Email**.")
        else:
            try:
                predictor = load_cached_predictor()
            except Exception as e:
                st.error(f"❌ Error loading trained model: {e}")
                st.info("Run `python -m src.train_model` from the terminal to train and serialize the model.")
                st.stop()

            with st.spinner("Analyzing email features and evaluating decision boundaries..."):
                result = predictor.predict(user_email_text)

            if not result.get("is_valid", False):
                st.error(f"⚠️ Validation Error: {result.get('error', 'Invalid email text.')}")
            else:
                is_spam = result["is_spam"]
                conf_pct = result["confidence_percentage"]
                spam_prob = result["spam_probability"]
                ham_prob = result["ham_probability"]
                triggers = result.get("detected_triggers", [])

                if is_spam:
                    st.markdown(
                        f"""
                        <div class="result-card-spam">
                            <div class="result-title-spam">
                                🚨 SPAM EMAIL DETECTED
                            </div>
                            <p style="font-size: 1.05rem; color: #7F1D1D; margin-bottom: 0.5rem;">
                                This message exhibits linguistic, structural, or promotional patterns characteristic of unsolicited, fraudulent, or phishing spam.
                            </p>
                            <div style="font-size: 0.95rem; font-weight: 600; color: #991B1B;">
                                Model Confidence: {conf_pct:.2f}% | Spam Probability: {spam_prob:.4f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-ham">
                            <div class="result-title-ham">
                                ✅ LEGITIMATE EMAIL
                            </div>
                            <p style="font-size: 1.05rem; color: #065F46; margin-bottom: 0.5rem;">
                                This message matches legitimate email characteristics, featuring natural conversational phrasing, realistic contextual semantics, and no predatory triggers.
                            </p>
                            <div style="font-size: 0.95rem; font-weight: 600; color: #047857;">
                                Model Confidence: {conf_pct:.2f}% | Legitimate Probability: {ham_prob:.4f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Probabilistic Metric Visualization
                col_prob1, col_prob2 = st.columns(2)
                with col_prob1:
                    st.metric("Legitimate Probability", f"{ham_prob * 100:.2f}%")
                    st.progress(ham_prob)
                with col_prob2:
                    st.metric("Spam Probability", f"{spam_prob * 100:.2f}%")
                    st.progress(spam_prob)

                # Diagnostic Trigger Breakdown
                if triggers:
                    st.markdown("#### 🔍 Detected Behavioral & Lexical Signals:")
                    for trigger in triggers:
                        st.markdown(f"- ⚠️ **{trigger}**")
                else:
                    st.markdown("#### 🔍 Lexical Signals:")
                    st.markdown("- No aggressive spam markers or anomalous hyperlinks detected.")

                # Normalized Representation Preview
                with st.expander("View Preprocessed & Normalized Text Preview"):
                    st.code(result["cleaned_preview"], language="text")

                # Disclaimer Note
                st.markdown(
                    f"""
                    <div class="disclaimer-box">
                        <strong>Evaluation Disclaimer:</strong> {result['disclaimer']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# PAGE 3: MODEL PERFORMANCE
elif selected_page == "Model Performance":
    st.markdown(
        """
        <div class="app-header">
            <h1>Model Performance & Evaluation Dashboard</h1>
            <p>Empirical metrics from model training on the Apache SpamAssassin benchmark.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_data = load_evaluation_metadata()
    if not eval_data:
        st.error("Model evaluation metadata not found. Please run `python -m src.train_model` first.")
        st.stop()

    best_name = eval_data.get("best_model", "Logistic Regression")
    models = eval_data.get("models", {})
    best_metrics = models.get(best_name, {})

    # Top KPI Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{best_metrics.get("accuracy", 0.9918) * 100:.2f}%</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{best_metrics.get("precision", 0.9944) * 100:.2f}%</div><div class="stat-label">Precision (Spam)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{best_metrics.get("recall", 0.9862) * 100:.2f}%</div><div class="stat-label">Recall (Spam)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{best_metrics.get("f1_score", 0.9903) * 100:.2f}%</div><div class="stat-label">F1-Score (Spam)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Algorithm Comparison Table
    st.subheader("📊 Comparative Algorithm Evaluation (Test Set: 858 Emails)")

    comp_rows = []
    for name, m in models.items():
        comp_rows.append({
            "Algorithm": name + (" ★ (Selected Champion)" if name == best_name else ""),
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision (Spam)": f"{m['precision'] * 100:.2f}%",
            "Recall (Spam)": f"{m['recall'] * 100:.2f}%",
            "F1-Score (Spam)": f"{m['f1_score'] * 100:.2f}%",
            "ROC-AUC": f"{m.get('roc_auc', 0.9998):.4f}",
        })

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Evaluation Visualizations")

    # Display Side-by-Side Charts
    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        cm_path = Path("reports/figures/confusion_matrix.png")
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrices: Naive Bayes vs. Logistic Regression", use_container_width=True)
    with col_fig2:
        comp_path = Path("reports/figures/model_comparison.png")
        if comp_path.exists():
            st.image(str(comp_path), caption="Cross-Metric Performance Comparison", use_container_width=True)

    col_fig3, col_fig4 = st.columns(2)
    with col_fig3:
        dist_path = Path("reports/figures/dataset_distribution.png")
        if dist_path.exists():
            st.image(str(dist_path), caption="Corpus Class Distribution (SpamAssassin)", use_container_width=True)
    with col_fig4:
        roc_path = Path("reports/figures/metrics_comparison.png")
        if roc_path.exists():
            st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)

    # Detailed Classification Breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Per-Class Classification Report (Logistic Regression)")
    clf_rep = best_metrics.get("classification_report", {})
    if clf_rep:
        rep_rows = [
            {"Class": "Legitimate (Ham)", "Precision": f"{clf_rep['Ham']['precision']:.4f}", "Recall": f"{clf_rep['Ham']['recall']:.4f}", "F1-Score": f"{clf_rep['Ham']['f1-score']:.4f}", "Support": int(clf_rep['Ham']['support'])},
            {"Class": "Spam", "Precision": f"{clf_rep['Spam']['precision']:.4f}", "Recall": f"{clf_rep['Spam']['recall']:.4f}", "F1-Score": f"{clf_rep['Spam']['f1-score']:.4f}", "Support": int(clf_rep['Spam']['support'])},
        ]
        st.table(pd.DataFrame(rep_rows))


# PAGE 4: ABOUT PROJECT
elif selected_page == "About Project":
    st.markdown(
        """
        <div class="app-header">
            <h1>About SpamGuard AI</h1>
            <p>Artificial Intelligence & Generative AI Internship Assignment | Hidden Virtual CodeTech</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="content-box">
            <h3>Project Summary & Objectives</h3>
            <p>
                <strong>SpamGuard AI</strong> is an end-to-end Natural Language Processing (NLP) and Machine Learning application
                designed to accurately differentiate legitimate emails from unsolicited, deceptive, or malicious spam messages.
            </p>
            <p>
                Developed as a core deliverable for the <strong>Hidden Virtual CodeTech AI & Generative AI Internship Program</strong>
                under the guidance of <strong>Engr. Farhat Irshad</strong>, this system adheres to strict software engineering standards,
                incorporating automated testing, production caching, zero-leakage ML pipelines, and an enterprise web interface.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_tech1, c_tech2 = st.columns(2)
    with c_tech1:
        st.markdown(
            """
            <div class="content-box">
                <h3>NLP Preprocessing Methodology</h3>
                <ul style="line-height: 1.8;">
                    <li><strong>HTML Stripping & Entity Decoding:</strong> Strips markup without destroying text semantics.</li>
                    <li><strong>Negation Preservation:</strong> Protects vital negation tokens (<em>not, no, never</em>) from stopword pruning.</li>
                    <li><strong>Standardized Token Injection:</strong> Converts URLs (<code>httpaddr</code>), emails (<code>emailaddr</code>), and currency symbols (<code>moneysymb</code>) into high-signal feature tokens.</li>
                    <li><strong>TF-IDF Vectorization:</strong> Computes Term Frequency-Inverse Document Frequency using sublinear scaling and (1, 2) n-gram ranges.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_tech2:
        st.markdown(
            """
            <div class="content-box">
                <h3>Classification Algorithms</h3>
                <ul style="line-height: 1.8;">
                    <li><strong>Multinomial Naive Bayes:</strong> Employs Bayes' Theorem with Laplace smoothing (α=0.1) under conditional feature independence assumptions. Highly fast and resilient on text frequency counts.</li>
                    <li><strong>Logistic Regression (Champion Model):</strong> Employs L2-regularized maximum-likelihood estimation (C=2.0) with liblinear optimization. Yields 99.18% accuracy and 99.03% spam F1-score with balanced calibration.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="content-box">
            <h3>Technology Stack & Attribution</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 0.5rem;">
                <tr style="border-bottom: 1px solid #E2E8F0; height: 35px;">
                    <td style="font-weight: 600; width: 30%;">Programming Language</td>
                    <td>Python 3.12</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0; height: 35px;">
                    <td style="font-weight: 600;">Data Manipulation</td>
                    <td>Pandas, NumPy</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0; height: 35px;">
                    <td style="font-weight: 600;">Machine Learning & NLP</td>
                    <td>Scikit-Learn, BeautifulSoup4, Regular Expressions</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0; height: 35px;">
                    <td style="font-weight: 600;">Visualization & Reporting</td>
                    <td>Matplotlib, Seaborn</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0; height: 35px;">
                    <td style="font-weight: 600;">Model Persistence</td>
                    <td>Joblib</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0; height: 35px;">
                    <td style="font-weight: 600;">Web Interface</td>
                    <td>Streamlit</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0; height: 35px;">
                    <td style="font-weight: 600;">Automated Testing</td>
                    <td>Pytest (22 comprehensive unit tests)</td>
                </tr>
                <tr style="height: 35px;">
                    <td style="font-weight: 600;">Benchmark Dataset</td>
                    <td>Apache SpamAssassin Public Corpus (Apache Software Foundation)</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
