import streamlit as st
import requests
import json

# Configuration
API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(page_title="RAG LLM App", page_icon="📚", layout="wide")

def list_documents():
    """Fetch the list of indexed documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        if response.status_code == 200:
            return response.json().get("documents", [])
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
    return []

def delete_document(doc_id):
    """Delete a document from the index."""
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{doc_id}")
        if response.status_code == 200:
            st.success("Document deleted successfully.")
            st.rerun()
        else:
            st.error(f"Failed to delete document: {response.text}")
    except Exception as e:
        st.error(f"Error deleting document: {e}")

# Main Layout
st.title("📚 RAG-Based LLM Application")
st.markdown("Upload your documents and ask questions grounded in their content.")

# Sidebar for Upload and Document Management
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file (PDF, TXT, MD)", type=["pdf", "txt", "md"])
    
    if st.button("Upload and Index"):
        if uploaded_file is not None:
            with st.spinner("Processing and indexing document..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                try:
                    res = requests.post(f"{API_BASE_URL}/upload", files=files)
                    if res.status_code == 201:
                        st.success(f"Successfully indexed: {uploaded_file.name}")
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
        else:
            st.warning("Please select a file first.")
            
    st.divider()
    
    st.header("Indexed Documents")
    docs = list_documents()
    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {doc['document_name']} ({doc['chunk_count']} chunks)")
            with col2:
                if st.button("🗑️", key=f"del_{doc['document_id']}", help="Delete Document"):
                    delete_document(doc['document_id'])
    else:
        st.info("No documents indexed yet.")

# Main area for Q&A
st.header("Ask a Question")

prompt_type = st.radio(
    "Select Prompt Type:",
    ("strict", "basic", "summarize"),
    horizontal=True,
    help="Strict: Only uses context. Basic: General RAG. Summarize: Summarize chunks."
)

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "question": question,
                    "prompt_type": prompt_type
                }
                res = requests.post(f"{API_BASE_URL}/ask", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    
                    st.subheader("Answer")
                    st.write(answer)
                    
                    if sources:
                        with st.expander("View Source Chunks"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1} ({source['document_name']}) - Score: {source['score']:.4f}**")
                                st.text(source['text'])
                                st.divider()
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
