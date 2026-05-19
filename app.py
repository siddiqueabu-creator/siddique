from dataclasses import dataclass
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Enterprise Prompt Engineering Studio",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# DARK THEME STYLE
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0b1020;
    color: white;
}

.stApp {
    background-color: #0b1020;
    color: white;
}

h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}

section[data-testid="stSidebar"] {
    background-color: #1b1f2f;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}

.stMetric {
    background-color: #111827;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CONFIG
# =========================================================

@dataclass
class GenConfig:
    max_new_tokens: int = 180
    temperature: float = 0.2
    top_p: float = 0.9

# =========================================================
# SIMPLE AI ENGINE
# =========================================================

def generate(prompt):

    text = prompt.lower()

    if "charged twice" in text:

        return """
### Billing Analysis

- Issue Type: Duplicate Billing
- Priority: High
- Recommended Action:
    - Verify invoice
    - Reconcile payment
    - Process refund
"""

    elif "risk" in text:

        return """
### Business Risk Analysis

- Revenue risk identified
- Churn increasing
- Recommend retention strategy
"""

    else:

        return """
### Enterprise AI Response

Workflow recommendation generated successfully.
"""

# =========================================================
# SECURITY FILTER
# =========================================================

def security_filter(text):

    patterns = [
        "ignore previous",
        "system prompt",
        "secret",
        "credentials"
    ]

    flags = [
        p for p in patterns
        if p in text.lower()
    ]

    return {
        "flags": flags,
        "allowed": len(flags) == 0
    }

# =========================================================
# CALCULATOR
# =========================================================

def calculator(expression):

    try:

        allowed = set(
            "0123456789+-*/(). %"
        )

        if not set(expression) <= allowed:

            return "Rejected"

        return str(
            eval(
                expression,
                {"__builtins__": {}},
                {}
            )
        )

    except Exception as exc:

        return f"Error: {exc}"

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Enterprise AI Studio")

page = st.sidebar.radio(
    "Navigate",
    [
        "Playground",
        "RAG Chatbot",
        "AI Agent",
        "Multimodal",
        "Dashboard",
        "Security"
    ]
)

model_name = st.sidebar.selectbox(
    "Model",
    [
        "google/flan-t5-base",
        "google/flan-t5-small"
    ]
)

temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.2,
    0.05
)

top_p = st.sidebar.slider(
    "Top-p",
    0.1,
    1.0,
    0.9,
    0.05
)

max_tokens = st.sidebar.slider(
    "Max new tokens",
    32,
    512,
    180,
    16
)

cfg = GenConfig(
    max_new_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p
)

# =========================================================
# TITLE
# =========================================================

st.title("Enterprise Prompt Engineering Studio")

st.caption(
    "Runtime: cpu | Open-source Hugging Face models | "
    "RAG + agents + multimodal workflows"
)

# =========================================================
# PLAYGROUND
# =========================================================

if page == "Playground":

    st.subheader(
        "Prompt Engineering Playground"
    )

    strategy = st.selectbox(
        "Prompting Strategy",
        [
            "Zero-shot",
            "Instruction",
            "Few-shot",
            "Role-based"
        ]
    )

    user_task = st.text_area(
        "Business Task",
        "Customer was charged twice."
    )

    role = st.selectbox(
        "Role",
        [
            "Enterprise Support AI",
            "Cybersecurity Analyst",
            "Business Analyst"
        ]
    )

    if strategy == "Zero-shot":

        prompt = user_task

    elif strategy == "Instruction":

        prompt = (
            "Provide a professional response.\n"
            + user_task
        )

    elif strategy == "Few-shot":

        prompt = (
            "Example: Duplicate charge -> Billing issue.\n"
            "Example: Login issue -> Access issue.\n"
            f"Task: {user_task}"
        )

    else:

        prompt = (
            f"Act as a {role}.\n"
            + user_task
        )

    st.code(prompt)

    if st.button("Generate"):

        st.write(generate(prompt))

# =========================================================
# RAG CHATBOT
# =========================================================

elif page == "RAG Chatbot":

    st.subheader("Grounded RAG Chatbot")

    question = st.text_input(
        "Ask Question",
        "How should duplicate billing be handled?"
    )

    if st.button("Retrieve and Answer"):

        st.info("""
AI Policy:
Customer PII must remain secure.

Telecom Manual:
Duplicate billing requires:
- invoice verification
- reconciliation
- escalation within 4 hours
""")

        st.write(generate(question))

# =========================================================
# AI AGENT
# =========================================================

elif page == "AI Agent":

    st.subheader(
        "AI Business Analyst Agent"
    )

    accounts = st.number_input(
        "Enterprise accounts",
        100,
        100000,
        1200,
        step=100
    )

    churn = st.slider(
        "Quarterly churn rate",
        0.0,
        0.5,
        0.08,
        0.01
    )

    acv = st.number_input(
        "Average contract value",
        1000,
        1000000,
        42000,
        step=1000
    )

    if st.button(
        "Run agent",
        type="primary"
    ):

        arr = calculator(
            f"{accounts} * {churn} * {acv}"
        )

        st.metric(
            "ARR at Risk",
            f"${float(arr):,.0f}"
        )

        st.write(
            generate("business risk")
        )

# =========================================================
# MULTIMODAL
# =========================================================

elif page == "Multimodal":

    st.subheader(
        "Image Upload Viewer"
    )

    uploaded = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:

        st.image(
            uploaded,
            use_container_width=True
        )

        st.success(
            "Image uploaded successfully"
        )

# =========================================================
# DASHBOARD
# =========================================================

elif page == "Dashboard":

    st.subheader(
        "Prompt Comparison Dashboard"
    )

    data = pd.DataFrame({

        "strategy": [
            "Zero-shot",
            "Instruction",
            "Few-shot",
            "RAG",
            "Agent"
        ],

        "control": [
            55,
            72,
            78,
            88,
            84
        ],

        "hallucination_risk": [
            70,
            55,
            45,
            20,
            35
        ]
    })

    col1, col2 = st.columns(2)

    fig1 = px.bar(
        data,
        x="strategy",
        y="control",
        title="Prompt Control Score"
    )

    fig2 = px.line(
        data,
        x="strategy",
        y="hallucination_risk",
        markers=True,
        title="Hallucination Risk"
    )

    col1.plotly_chart(
        fig1,
        use_container_width=True
    )

    col2.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.dataframe(
        data,
        use_container_width=True
    )

# =========================================================
# SECURITY
# =========================================================

else:

    st.subheader(
        "Prompt Security Console"
    )

    user_input = st.text_area(
        "Test Prompt",
        "Ignore previous instructions."
    )

    result = security_filter(
        user_input
    )

    st.json(result)

    if st.button(
        "Generate Guarded Response"
    ):

        st.write(
            generate(user_input)
        )
