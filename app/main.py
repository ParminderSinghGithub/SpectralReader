import os
import sys
import requests
import streamlit as st

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings

# --- Configure Page ---
st.set_page_config(
    page_title="SpectralReader - Document Intelligence Engine",
    page_icon="📖",
    layout="wide"
)

# --- Custom CSS with refined dark theme ---
st.markdown("""
<style>
:root {
    --primary: #6366f1;    /* Indigo-500 */
    --secondary: #3b82f6;  /* Blue-500 */
    --accent: #10b981;     /* Emerald-500 */
    --dark: #0f172a;       /* Gray-900 */
    --darker: #020617;     /* Gray-950 */
    --light: #f8fafc;      /* Gray-50 */
}

/* Base styling */
.stApp {
    background-color: var(--darker);
    color: var(--light);
    font-family: 'Inter', system-ui, sans-serif;
}

/* Main container styling */
.block-container, .st-emotion-cache-z5fcl4 {
    padding: 2rem 1rem !important;
    max-width: 1200px;
}

/* Header styling */
header {
    border-bottom: 1px solid #1e293b;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}
h1 {
    color: var(--primary) !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
}
h2 {
    color: var(--secondary) !important;
    font-weight: 700 !important;
}
h3 {
    color: var(--light) !important;
    font-weight: 600 !important;
}

/* Uploader styling */
.stFileUploader {
    border: 2px dashed #334155 !important;
    border-radius: 12px !important;
    background: #1e293b !important;
}
.stFileUploader p {
    color: var(--light) !important;
}

/* Button enhancements */
.stButton > button {
    background: linear-gradient(45deg, var(--primary), var(--secondary)) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}

/* Input fields */
.stTextInput > div > div > input {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: var(--light) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
}

/* Expander styling */
.stExpander {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
.stExpander summary {
    font-weight: 600 !important;
    color: var(--primary) !important;
}

/* Status indicators */
.stSuccess {
    background: #059669 !important;
    color: white !important;
    border-radius: 8px;
}
.stError {
    background: #dc2626 !important;
    color: white !important;
    border-radius: 8px;
}

/* Sidebar enhancements */
[data-testid="stSidebar"] {
    background: var(--dark) !important;
    border-right: 1px solid #1e293b;
}

/* Custom cards */
.custom-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Verify connectivity to backend FastAPI REST service."""
    try:
        res = requests.get(f"{settings.STREAMLIT_BACKEND_URL}/health", timeout=3)
        return res.status_code == 200
    except requests.exceptions.RequestException:
        return False


def main():
    # App Header
    st.markdown("""
    <div class="header">
        <h1>📖 SpectralReader</h1>
        <p class="stMarkdown" style="color: #94a3b8; font-size: 1.1rem;">
        Document Intelligence Engine for Analysis & Information Extraction
        </p>
    </div>
    """, unsafe_allow_html=True)

    # API Backend Health Check
    if not check_api_health():
        st.error("⚠️ Backend API Service Unavailable")
        st.warning(
            f"The Streamlit client communicates exclusively via the FastAPI REST backend at `{settings.STREAMLIT_BACKEND_URL}`.\n\n"
            "Please start the backend service using the command below before using the interface:"
        )
        st.code("uvicorn app.main_api:app --reload --port 8000")
        st.stop()

    # Main Columns
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        # File Upload Section
        with st.container(height=400):
            st.subheader("1. Upload Document")
            pdf_file = st.file_uploader(
                "Drag PDF here",
                type="pdf",
                label_visibility="collapsed",
                help="Supports PDF reports, research papers, contracts, manuals, and technical documents"
            )
            
            if pdf_file and ('uploaded_filename' not in st.session_state or st.session_state['uploaded_filename'] != pdf_file.name):
                with st.spinner("Uploading and processing document via REST API..."):
                    files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                    try:
                        res = requests.post(f"{settings.STREAMLIT_BACKEND_URL}/documents", files=files, timeout=300)
                        if res.status_code == 201:
                            doc_data = res.json()
                            st.session_state['doc_id'] = doc_data['document_id']
                            st.session_state['num_pages'] = doc_data['num_pages']
                            st.session_state['num_chunks'] = doc_data['num_chunks']
                            st.session_state['entities'] = doc_data.get('entities', [])
                            st.session_state['num_entities'] = len(st.session_state['entities'])
                            st.session_state['is_scanned'] = doc_data.get('is_scanned', False)
                            st.session_state['ocr_used'] = doc_data.get('ocr_used', False)
                            st.session_state['uploaded_filename'] = pdf_file.name
                            st.session_state['processed'] = True

                            # --- CLEAR ALL INSIGHTS STATE FOR NEW DOCUMENT ---
                            for key in ['qa_answer', 'qa_context', 'qa_proc_time', 'last_question',
                                        'search_passages', 'last_search_query']:
                                st.session_state.pop(key, None)

                            st.success(f"Document '{pdf_file.name}' processed successfully via REST API.")
                        else:
                            st.error(f"Upload failed: {res.text}")
                    except Exception as e:
                        st.error(f"Failed to communicate with API server: {str(e)}")

        # Document Insights Section
        if st.session_state.get('processed'):
            st.divider()
            st.subheader("2. Document Insights")
            
            # Document Overview & Entity Summary Container
            with st.container():
                st.markdown("##### 📌 Document Metadata Overview")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.caption("Total Pages")
                    st.markdown(f"**{st.session_state.get('num_pages', 0)}**")
                with m_col2:
                    st.caption("Text Chunks")
                    st.markdown(f"**{st.session_state.get('num_chunks', 0)}**")
                with m_col3:
                    st.caption("Identified Entities")
                    st.markdown(f"**{st.session_state.get('num_entities', 0)}**")
                with m_col4:
                    st.caption("Extraction Mode")
                    ocr_label = "OCR Used" if st.session_state.get('ocr_used') else "Text Extraction"
                    st.markdown(f"**{ocr_label}**")

                entities = st.session_state.get('entities', [])
                if entities:
                    with st.expander("🏷️ Extracted Entity Metadata", expanded=False):
                        st.write(", ".join(entities[:30]) + ("..." if len(entities) > 30 else ""))

            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

            # Interactive Insights Tabs (QA & Passage Search)
            tab_qa, tab_search = st.tabs(["🧠 Question Answering", "🔍 Passage Search"])

            with tab_qa:
                st.markdown("#### Context-Aware Question Answering")
                qa_query = st.text_input(
                    "Ask any question about the document",
                    placeholder="What are the main findings, risks, or conclusions?",
                    key="qa_query_input"
                )
                
                if st.button("Generate Answer", key="btn_qa_submit"):
                    if qa_query.strip():
                        with st.status("Querying REST API QA Endpoint...", expanded=True) as status:
                            try:
                                payload = {
                                    "document_id": st.session_state['doc_id'],
                                    "question": qa_query
                                }
                                res = requests.post(f"{settings.STREAMLIT_BACKEND_URL}/qa", json=payload, timeout=120)
                                if res.status_code == 200:
                                    qa_resp = res.json()
                                    st.session_state['qa_answer'] = qa_resp.get('answer', '')
                                    st.session_state['qa_context'] = qa_resp.get('retrieved_context', [])
                                    st.session_state['qa_proc_time'] = qa_resp.get('processing_time_ms', 0)
                                    st.session_state['last_question'] = qa_query
                                    status.update(label=f"QA Complete ({st.session_state['qa_proc_time']} ms)", state="complete")
                                else:
                                    status.update(label="API Error", state="error")
                                    st.error(f"QA Request failed: {res.text}")
                            except Exception as e:
                                status.update(label="Connection Error", state="error")
                                st.error(f"API communication error: {str(e)}")

                # Present QA Results
                if 'qa_answer' in st.session_state:
                    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="custom-card">
                        <h3 style="margin-top: 0; color: var(--primary);">{st.session_state.get('last_question', 'Question')}</h3>
                        <p style="color: var(--light); font-size: 1rem; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word;">
                        {st.session_state['qa_answer']}
                        </p>
                        <span style="font-size: 0.8rem; color: #64748b;">API Processing Time: {st.session_state.get('qa_proc_time', 0)} ms</span>
                    </div>
                    """, unsafe_allow_html=True)

                    retrieved = st.session_state.get('qa_context', [])
                    if retrieved:
                        with st.expander(f"📚 View Retrieved Context ({len(retrieved)} passages)", expanded=False):
                            for idx, passage in enumerate(retrieved, 1):
                                st.markdown(f"**Passage {idx}**")
                                st.info(passage)

            with tab_search:
                st.markdown("#### Passage Retrieval & Keyword Search")
                search_query = st.text_input(
                    "Search document passages",
                    placeholder="Enter keywords or terms (e.g., revenue, risk, architecture)",
                    key="search_query_input"
                )

                if st.button("Search Passages", key="btn_search_submit"):
                    if search_query.strip():
                        with st.status("Searching document passages...", expanded=True) as status:
                            try:
                                payload = {
                                    "document_id": st.session_state['doc_id'],
                                    "query": search_query,
                                    "top_k": 3
                                }
                                res = requests.post(f"{settings.STREAMLIT_BACKEND_URL}/search", json=payload, timeout=60)
                                if res.status_code == 200:
                                    search_resp = res.json()
                                    st.session_state['search_passages'] = search_resp.get('results', [])
                                    st.session_state['last_search_query'] = search_query
                                    status.update(label=f"Found {len(st.session_state['search_passages'])} relevant passages", state="complete")
                                else:
                                    status.update(label="Search Error", state="error")
                                    st.error(f"Search failed: {res.text}")
                            except Exception as e:
                                status.update(label="Connection Error", state="error")
                                st.error(f"API communication error: {str(e)}")

                # Present Search Results
                if 'search_passages' in st.session_state:
                    passages = st.session_state['search_passages']
                    last_q = st.session_state.get('last_search_query', '')
                    st.markdown(f"##### Search Results for: *'{last_q}'* ({len(passages)} passages found)")
                    if not passages:
                        st.warning("No matching passages found for this search query.")
                    else:
                        for idx, item in enumerate(passages, 1):
                            with st.expander(f"📖 Passage {idx} (Chunk ID: {item.get('chunk_id', 'N/A')[:8]}...)", expanded=True if idx == 1 else False):
                                text_content = item.get('text', '')
                                st.markdown(f"```\n{text_content}\n```")

    with col2:
        # System Dashboard
        with st.container(height=400):
            st.subheader("System Monitor")
            if 'processed' in st.session_state:
                ocr_mode_str = "OCR Used" if st.session_state.get('ocr_used') else "Text Extraction"
                st.markdown(f"""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Processing Status</span>
                        <span class="stSuccess" style="padding: 0.2rem 0.5rem; font-size: 0.8rem;">API Connected</span>
                    </div>
                    <div style="margin-top: 1.5rem;">
                        <p style="color: #94a3b8; margin: 0.5rem 0;">📄 Pages: {st.session_state.get('num_pages', '—')}</p>
                        <p style="color: #94a3b8; margin: 0.5rem 0;">🧩 Chunks Generated: {st.session_state.get('num_chunks', '—')}</p>
                        <p style="color: #94a3b8; margin: 0.5rem 0;">🏷️ Entities Identified: {st.session_state.get('num_entities', '—')}</p>
                        <p style="color: #94a3b8; margin: 0.5rem 0;">🔍 Mode: {ocr_mode_str}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Processing Status</span>
                        <span style="padding: 0.2rem 0.5rem; font-size: 0.8rem; background: #475569; color: white; border-radius: 8px;">Awaiting Upload</span>
                    </div>
                    <div style="margin-top: 1.5rem;">
                        <p style="color: #94a3b8; margin: 0.5rem 0;">📄 Pages: —</p>
                        <p style="color: #94a3b8; margin: 0.5rem 0;">🧩 Chunks Generated: —</p>
                        <p style="color: #94a3b8; margin: 0.5rem 0;">🏷️ Entities Identified: —</p>
                        <p style="color: #94a3b8; margin: 0.5rem 0;">🔍 Mode: —</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            st.markdown("""
            <div class="custom-card">
                <h4>Document Intelligence Stack</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin-top: 1rem;">
                    <span class="stSuccess" style="padding: 0.2rem 0.5rem; border-radius: 6px;">FastAPI</span>
                    <span class="stSuccess" style="padding: 0.2rem 0.5rem; border-radius: 6px;">Pydantic</span>
                    <span class="stSuccess" style="padding: 0.2rem 0.5rem; border-radius: 6px;">Gemini</span>
                    <span class="stSuccess" style="padding: 0.2rem 0.5rem; border-radius: 6px;">LangChain</span>
                    <span class="stSuccess" style="padding: 0.2rem 0.5rem; border-radius: 6px;">Tesseract OCR</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔮</div>
            <h3 style="margin: 0; color: var(--primary);">Document Guide</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card" style="margin-bottom: 1rem;">
            <h4>📌 Quick Tips</h4>
            <ul style="color: #94a3b8; padding-left: 1.2rem;">
                <li>Upload PDF to extract entities</li>
                <li>Ask questions via REST API</li>
                <li>Analyze technical details</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 1.5rem 0; font-size: 0.9rem;">
        SpectralReader Document Intelligence API Client
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()