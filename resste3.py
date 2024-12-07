import streamlit as st
from langflow.load import run_flow_from_json
import pdfplumber

# Function to extract all pages from a PDF file as images
def extract_all_pages_as_images(file_upload):
    """Extract all pages from a PDF file as images."""
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        for page in pdf.pages:
            pdf_pages.append(page.to_image().original)
    return pdf_pages

# Helper Function to Run the Flow
def run_flow(flow_name, tweaks, input_value):
    try:
        result = run_flow_from_json(
            flow=flow_name,
            session_id="",
            fallback_to_env_vars=True,
            tweaks=tweaks,
            input_value=input_value
        )
        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            return first_result.outputs[0].results['message'].data['text']
        else:
            return "No results returned or unexpected format."
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit App Configuration
st.set_page_config(layout="wide")

# Sidebar: Feedback Section
with st.sidebar:
    st.header("Feedback")
    user_feedback = st.text_area("Provide your feedback:", 
                                  placeholder="Share your thoughts or suggestions...")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

# Main Layout: Columns
col1, col2 = st.columns([1, 2])  # Left: PDF Preview | Right: Actions & Results

# Job Description Input in Right Column
with col2:
    st.title("ATS Resume Checker")
    job_description = st.text_area("Enter the Job Description:", 
                                    placeholder="Paste the job description here...")
    st.divider()  # Visual separator

# File Upload: Resume (PDF)
uploaded_resume = st.file_uploader("Upload Your Resume (PDF):", type=["pdf"])

if uploaded_resume:
    # PDF Preview: Extract pages as images and display
    with col1:
        st.header("PDF Preview")
        try:
            pdf_pages = extract_all_pages_as_images(uploaded_resume)
            zoom_level = st.slider("Zoom Level", min_value=100, max_value=1000, 
                                   value=700, step=50)
            for page_image in pdf_pages:
                st.image(page_image, width=zoom_level)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

# Common Tweaks for Flows
COMMON_TWEAKS = {
    "Chroma": {
        "allow_duplicates": False,
        "collection_name": "resume1",
        "persist_directory": "E:\\lang12\\chroma",
        "number_of_results": 10,
        "search_type": "Similarity"
    },
    "OllamaModel": {
        "base_url": "http://localhost:11434",
        "model_name": "llama3:latest",
        "temperature": 0.2,
        "stream": False
    },
    "OllamaEmbeddings": {
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text:latest",
        "temperature": 0.1
    },
    "SplitText": {
        "chunk_overlap": 200,
        "chunk_size": 1000,
        "separator": "\n"
    }
}

# Specific Tweaks
TWEAKS_PERCENTAGE_MATCH = {
    **COMMON_TWEAKS,
    "Prompt": {
        "template": "{context}\n\n---\n\nCompare the job description with the resume and provide a match percentage.\n"
                    "Percentage Match-\nKeywords Missing-\nFinal Thoughts-\nQuestion: {question}\nAnswer:"
    }
}

TWEAKS_RESUME_ANALYSIS = {
    **COMMON_TWEAKS,
    "Prompt": {
        "template": "{context}\n---\n Compare the job description with the resume and provide a detailed analysis, including:\n - Percentage Match:\n - Keywords Missing:\n - Final Thoughts:\n Question: {question} \n Answer:"
    }
}

# Buttons for Actions
with col2:
    if st.button("Tell About the Resume"):
        if uploaded_resume:
            try:
                resume_content = uploaded_resume.read().decode("utf-8", errors="ignore")
                output = run_flow("Resume Analysis.json", TWEAKS_RESUME_ANALYSIS, resume_content)
                st.subheader("Resume Analysis Output:")
                st.write(output)
            except Exception as e:
                st.error(f"Error analyzing resume: {e}")
        else:
            st.error("Please upload a resume to analyze.")

    if st.button("Percentage Match"):
        if uploaded_resume and job_description:
            try:
                resume_content = uploaded_resume.read().decode("utf-8", errors="ignore")
                input_data = f"Job Description:\n{job_description}\n\nResume Content:\n{resume_content}"
                output = run_flow("Percentage Match.json", TWEAKS_PERCENTAGE_MATCH, input_data)
                st.subheader("Percentage Match Output:")
                st.write(output)
            except Exception as e:
                st.error(f"Error calculating percentage match: {e}")
        else:
            st.error("Please upload a resume and enter a job description.")
