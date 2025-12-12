# frontend.py
import streamlit as st
import requests
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="MCQ Generator",
    page_icon="📘",
    layout="wide"
)

BACKEND_URL = "http://127.0.0.1:8000/generate_mcqs"

# Add sidebar with instructions and API key setup
with st.sidebar:
    st.header("📘 MCQ Generator")
    st.write("Generate multiple-choice questions from your PDF documents using AI")

    with st.expander("📝 How to use"):
        st.markdown("""
        **Steps to generate MCQs:**
        1. Prepare answers you want questions about
        2. Upload a PDF document
        3. Enter your answers (comma-separated)
        4. Click "Generate MCQs"
        5. Review and download the results
        """)


    st.info("💡 **Tip:** The more specific your answers are, the better the generated questions will be!")

st.title("📘 MCQ Generator")
st.markdown("Upload a PDF document and specify answers to generate multiple-choice questions")

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], help="Upload a PDF document to extract content from")

    if uploaded_file:
        st.success(f"✅ File selected: {uploaded_file.name}")
        st.info(f"📄 Size: {uploaded_file.size / 1024:.2f} KB")

    st.subheader("✍️ Specify Answers")
    answers_input = st.text_area(
        "Enter answers to generate questions for",
        height=150,
        placeholder="Enter answers separated by commas (e.g., photosynthesis, mitosis, Newton's laws)",
        help="List the key concepts, terms, or facts you want questions about"
    )

    if answers_input:
        answers_list = [a.strip() for a in answers_input.split(",") if a.strip()]
        st.info(f"Number of answers: {len(answers_list)}")
        with st.expander("View answers"):
            for i, ans in enumerate(answers_list, 1):
                st.write(f"{i}. {ans}")

with col2:
    st.subheader("⚙️ Generate MCQs")

    if uploaded_file and answers_input:
        st.info("Ready to generate MCQs!")
        generate_btn = st.button("🚀 Generate MCQs", type="primary", use_container_width=True)
    else:
        st.warning("Please upload a PDF and enter answers to proceed")
        generate_btn = st.button("🚀 Generate MCQs", type="primary", use_container_width=True, disabled=True)

    if generate_btn and uploaded_file and answers_input:
        with st.spinner("🧠 AI is generating questions... This may take 30-60 seconds depending on document size ⏳"):
            try:
                response = requests.post(BACKEND_URL, files={"file": uploaded_file}, data={"answers": answers_input})

                if response.status_code == 200:
                    result = response.json()
                    mcqs = result["mcqs"]

                    st.success(f"✅ Successfully generated {len(mcqs)} MCQs!")

                    st.subheader("📋 Generated Questions")

                    # Display MCQs in a more organized way
                    for i, row in enumerate(mcqs, 1):
                        with st.container():
                            st.markdown(f"### {i}. {row['question']}")

                            # Create columns for options
                            cols = st.columns(2)

                            cols[0].button(f"🇦 {row['option_1']}", key=f"opt1_{i}", use_container_width=True, disabled=True)
                            cols[1].button(f"🇧 {row['option_2']}", key=f"opt2_{i}", use_container_width=True, disabled=True)

                            cols2 = st.columns(2)
                            cols2[0].button(f"🇨 {row['option_3']}", key=f"opt3_{i}", use_container_width=True, disabled=True)
                            cols2[1].button(f"🇩 {row['option_4']}", key=f"opt4_{i}", use_container_width=True, disabled=True)

                            # Show correct answer
                            correct_option_map = {"option_1": row['option_1'], "option_2": row['option_2'],
                                                "option_3": row['option_3'], "option_4": row['option_4']}
                            correct_answer = correct_option_map[row['correct_option']]

                            st.success(f"✅ **Correct Answer:** {row['answer']} (Option: {correct_answer})")
                            st.markdown("---")

                    # Prepare CSV for download
                    df = pd.DataFrame(mcqs)

                    # Add additional columns for better CSV
                    df_display = df.copy()
                    df_display = df_display.rename(columns={
                        'option_1': 'Option A',
                        'option_2': 'Option B',
                        'option_3': 'Option C',
                        'option_4': 'Option D'
                    })

                    csv = df_display.to_csv(index=False).encode('utf-8')

                    # Download button in a separate container
                    with st.container():
                        st.subheader("💾 Download Results")
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name="generated_mcqs.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                        # Show data preview
                        with st.expander("📊 View as Table"):
                            st.dataframe(df_display)

                else:
                    st.error(f"❌ Error from backend: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("🔌 **Connection Error:** Could not connect to the backend server.")
                st.info("Please ensure the FastAPI server is running on http://127.0.0.1:8000")
                st.code("uvicorn api_server:app --reload --port 8000")

            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
    elif generate_btn:
        st.warning("Please upload a PDF file and enter answers before generating MCQs.")
