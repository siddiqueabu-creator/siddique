import streamlit as st

st.set_page_config(
    page_title="Enterprise AI Studio",
    layout="wide"
)

st.title("Enterprise Prompt Engineering Studio")

st.success("App running successfully ✅")

menu = st.sidebar.selectbox(
    "Navigate",
    [
        "Home",
        "AI Agent",
        "Dashboard"
    ]
)

# =====================================================
# HOME
# =====================================================

if menu == "Home":

    st.header("Welcome")

    st.write("""
Enterprise AI Studio Demo
""")

# =====================================================
# AI AGENT
# =====================================================

elif menu == "AI Agent":

    st.header("AI Business Analyst Agent")

    accounts = st.number_input(
        "Enterprise accounts",
        100,
        100000,
        1200
    )

    churn = st.slider(
        "Quarterly churn rate",
        0.0,
        0.5,
        0.08
    )

    acv = st.number_input(
        "Average contract value",
        1000,
        1000000,
        42000
    )

    if st.button("Run agent"):

        arr = accounts * churn * acv

        st.metric(
            "ARR at Risk",
            f"${arr:,.0f}"
        )

        st.write("""
### Business Risk Analysis

- Revenue risk identified
- Churn increasing
- Recommend customer retention strategy
""")

# =====================================================
# DASHBOARD
# =====================================================

else:

    st.header("Dashboard")

    st.bar_chart({
        "Control Score": [55, 72, 88]
    })
