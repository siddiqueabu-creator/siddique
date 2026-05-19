
!pip install streamlit -q
import streamlit as st
from transformers import pipeline

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="International Business Marketing Prompt App",
    page_icon="🌍",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🌍 International Business Marketing Prompt Application")
st.markdown(
    "Generate professional global marketing content using Generative AI."
)

# -----------------------------
# Load Hugging Face Model
# -----------------------------
@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    return generator

generator = load_model()

# -----------------------------
# User Input
# -----------------------------
product_name = st.text_input(
    "Enter Product Name",
    placeholder="Example: Smart Water Bottle"
)

# -----------------------------
# Generate Button
# -----------------------------
if st.button("Generate Marketing Content"):

    if product_name.strip() == "":
        st.warning("Please enter a product name.")
    else:

        # -----------------------------
        # Prompt Engineering
        # -----------------------------
        prompt = f"""
You are an International Business Marketing Expert.

Generate professional global marketing content for the product: {product_name}

Provide the output in the following format:

1. Global-Ready Product Title
2. Powerful Marketing Slogan
3. Advertising Description from:
- Digital Marketing Expert
- Luxury Brand Strategist
- Emotional Storytelling Expert

The content should:
- Follow international marketing standards
- Be persuasive and professional
- Appeal to global audiences
- Include emotional engagement
- Be suitable for worldwide advertising campaigns
"""

        # -----------------------------
        # AI Response
        # -----------------------------
        with st.spinner("Generating AI Marketing Content..."):

            response = generator(
                prompt,
                max_new_tokens=350,
                temperature=0.8,
                do_sample=True
            )

            result = response[0]["generated_text"]

        # -----------------------------
        # Display Output
        # -----------------------------
        st.success("Marketing Content Generated Successfully!")

        st.markdown("## 📢 Generated Marketing Content")
        st.write(result)
