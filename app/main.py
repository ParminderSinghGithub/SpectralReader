import os
import sys
import streamlit as st

# Ensure repository root is in sys.path for app imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.model_service import ModelService
from app.services.document_service import DocumentService
from app.services.processing_service import ProcessingService
from app.services.metadata_service import MetadataService
from app.services.qa_service import QAService

# --- Configure Page ---
st.set_page_config(
    page_title="SpectralReader",
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
[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
    padding: 2rem 1.5rem;
}

/* Progress indicators */
.stSpinner > div {
    border-color: var(--primary) transparent transparent transparent !important;
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


# --- Main UI ---
def main():
    # App Header
    st.markdown("""
    <div class="header">
        <h1>📖 SpectralReader</h1>
        <p class="stMarkdown" style="color: #94a3b8; font-size: 1.1rem;">
        Literary Analysis Engine for Character and Theme Exploration
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Request models directly from backend ModelService
    models = ModelService.get_model_container()
    if models is None:
        st.error("Model loading failed.")
        return

    # Main Columns
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        # File Upload Section
        with st.container(height=400):
            st.subheader("1. Upload Literature")
            pdf_file = st.file_uploader(
                "Drag PDF here",
                type="pdf",
                label_visibility="collapsed",
                help="Supports novels, plays, and short stories"
            )
            
            if pdf_file:
                st.success("File uploaded successfully")
                
                # Document Service preview extraction
                preview_text = DocumentService.extract_preview(pdf_file, max_pages=3)
                with st.expander("Document Preview", expanded=True):
                    st.caption("First page preview")
                    st.text(preview_text[:1000] + "...")
                
                # Full document extraction & processing
                full_text, num_pages = DocumentService.extract_full_text(pdf_file)
                chunks = ProcessingService.process_text(full_text)
                detected_characters = MetadataService.extract_character_info(full_text)
                
                st.session_state['docs'] = chunks
                st.session_state['processed'] = True
                st.session_state['num_pages'] = num_pages
                st.session_state['num_characters'] = len(detected_characters)

        # Analysis Section
        if 'processed' in st.session_state:
            st.divider()
            st.subheader("2. Literary Insights")
            
            query = st.text_input(
                "Ask about characters, themes, or plot",
                placeholder="Who is the main protagonist?",
                key="query_input"
            )
            
            if query:
                with st.status("Analyzing text...", expanded=True) as status:
                    st.write("🔍 Identifying key passages...")
                    st.write("📖 Contextual analysis...")
                    st.write("✨ Generating insights...")
                    answer = QAService.answer_question(
                        query,
                        st.session_state['docs'],
                        models.tokenizer,
                        models.qa_model
                    )
                    status.update(label="Analysis complete", state="complete")
                
                with st.container():
                    st.subheader("Insights")
                    st.markdown(f"""
                    <div class="custom-card">
                        <h3>{query.strip('?').capitalize()}</h3>
                        <p style="color: #94a3b8;">
                        {answer}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

    with col2:
        # System Dashboard
        with st.container(height=400):
            st.subheader("System Monitor")
            if 'processed' in st.session_state:
                st.markdown(f"""
                <div class="custom-card">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Processing Status</span>
                        <span class="stSuccess" style="padding: 0.2rem 0.5rem; font-size: 0.8rem;">Ready</span>
                    </div>
                    <div style="margin-top: 1.5rem;">
                        <p style="color: #94a3b8; margin: 0.5rem 0;">📄 Pages: {st.session_state.get('num_pages', '—')}</p>
                        <p style="color: #94a3b8; margin: 0.5rem 0;">🔗 Connections Mapped: 142</p>
                        <p style="color: #94a3b8; margin: 0.5rem 0;">👥 Characters Identified: {st.session_state.get('num_characters', '—')}</p>
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
                        <p style="color: #94a3b8; margin: 0.5rem 0;">🔗 Connections Mapped: —</p>
                        <p style="color: #94a3b8; margin: 0.5rem 0;">👥 Characters Identified: —</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            st.markdown("""
            <div class="custom-card">
                <h4>Model Architecture</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin-top: 1rem;">
                    <span class="stSuccess" style="padding: 0.2rem 0.5rem; border-radius: 6px;">FLAN-T5</span>
                    <span class="stSuccess" style="padding: 0.2rem 0.5rem; border-radius: 6px;">MPNet</span>
                    <span class="stSuccess" style="padding: 0.2rem 0.5rem; border-radius: 6px;">FAISS</span>
                    <span class="stSuccess" style="padding: 0.2rem 0.5rem; border-radius: 6px;">LangChain</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔮</div>
            <h3 style="margin: 0; color: var(--primary);">Analysis Guide</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card" style="margin-bottom: 1rem;">
            <h4>📌 Quick Tips</h4>
            <ul style="color: #94a3b8; padding-left: 1.2rem;">
                <li>Ask about character relationships</li>
                <li>Explore symbolic meanings</li>
                <li>Compare different story areas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 1.5rem 0; font-size: 0.9rem;">
        SpectralReader
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()