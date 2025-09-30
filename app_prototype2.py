import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import PyPDF2
import io

# Configure page
st.set_page_config(
    page_title="Paper Annotation Tool",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for Prodigy-like styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Prodigy-like styling */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .paper-card {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .action-buttons {
        display: flex;
        gap: 12px;
        justify-content: center;
        margin: 24px 0;
    }
    
    .btn-accept {
        background: #28a745 !important;
        color: white !important;
        border: none !important;
        padding: 12px 32px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        cursor: pointer;
    }
    
    .btn-reject {
        background: #dc3545 !important;
        color: white !important;
        border: none !important;
        padding: 12px 32px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        cursor: pointer;
    }
    
    .btn-ignore {
        background: #6c757d !important;
        color: white !important;
        border: none !important;
        padding: 12px 32px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        cursor: pointer;
    }
    
    .btn-flag {
        background: #ffc107 !important;
        color: #212529 !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
    }
    
    .score-badge {
        background: #007bff;
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: bold;
        font-size: 14px;
        float: right;
    }
    
    .progress-section {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 16px;
        margin: 16px 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 12px;
        background: white;
        border-radius: 6px;
        border: 1px solid #dee2e6;
    }
    
    .stat-number {
        font-size: 24px;
        font-weight: bold;
        color: #007bff;
    }
    
    .stat-label {
        font-size: 12px;
        color: #6c757d;
        text-transform: uppercase;
    }
    
    .keyboard-hint {
        background: #e9ecef;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        margin: 8px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript for keyboard shortcuts
keyboard_shortcuts = """
<script>
document.addEventListener('keydown', function(e) {
    if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        switch(e.key.toLowerCase()) {
            case ' ':
            case 'a':
                e.preventDefault();
                const acceptBtn = document.querySelector('[data-testid*="accept"]');
                if (acceptBtn) acceptBtn.click();
                break;
            case 'r':
            case 'x':
                e.preventDefault();
                const rejectBtn = document.querySelector('[data-testid*="reject"]');
                if (rejectBtn) rejectBtn.click();
                break;
            case 'i':
                e.preventDefault();
                const ignoreBtn = document.querySelector('[data-testid*="ignore"]');
                if (ignoreBtn) ignoreBtn.click();
                break;
            case 'f':
                e.preventDefault();
                const flagBtn = document.querySelector('[data-testid*="flag"]');
                if (flagBtn) flagBtn.click();
                break;
            case 'arrowleft':
                e.preventDefault();
                const prevBtn = document.querySelector('[data-testid*="prev"]');
                if (prevBtn) prevBtn.click();
                break;
            case 'arrowright':
                e.preventDefault();
                const nextBtn = document.querySelector('[data-testid*="next"]');
                if (nextBtn) nextBtn.click();
                break;
        }
    }
});
</script>
"""


def init_session_state():
    """Initialize session state variables"""
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
    if 'session_stats' not in st.session_state:
        st.session_state.session_stats = {
            'start_time': datetime.now(),
            'total_annotations': 0,
            'accept_count': 0,
            'reject_count': 0,
            'ignore_count': 0,
            'flagged_count': 0
        }
    if 'show_instructions' not in st.session_state:
        st.session_state.show_instructions = False
    if 'config' not in st.session_state:
        st.session_state.config = {
            'show_flag': True,
            'auto_advance': True,
            'swipe_enabled': False  # For mobile support
        }


def load_sample_papers():
    """Load sample papers for demonstration"""
    return [
        {
            "id": "paper_001",
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "authors": "Vaswani et al.",
            "venue": "NIPS 2017",
            "score": 0.85,
            "keywords": ["transformer", "attention", "neural networks"]
        },
        {
            "id": "paper_002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
            "authors": "Devlin et al.",
            "venue": "NAACL 2019",
            "score": 0.92,
            "keywords": ["BERT", "bidirectional", "pre-training"]
        },
        {
            "id": "paper_003",
            "title": "Deep Residual Learning for Image Recognition",
            "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
            "authors": "He et al.",
            "venue": "CVPR 2016",
            "score": 0.78,
            "keywords": ["residual networks", "deep learning", "computer vision"]
        }
    ]


def extract_pdf_text(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None


def process_uploaded_files(uploaded_files):
    """Process uploaded files and convert to paper format"""
    papers = []

    for i, file in enumerate(uploaded_files):
        if file.name.endswith('.pdf'):
            # Process PDF
            with st.spinner(f"Processing {file.name}..."):
                full_text = extract_pdf_text(file)
                if full_text:
                    lines = [line.strip()
                             for line in full_text.split('\n') if line.strip()]
                    title = lines[0] if lines else file.name.replace(
                        '.pdf', '')

                    # Use first 300 words as abstract
                    words = full_text.split()
                    abstract = ' '.join(words[:300])
                    if len(words) > 300:
                        abstract += "..."

                    papers.append({
                        "id": f"pdf_{i+1}",
                        "title": title,
                        "abstract": abstract,
                        "full_text": full_text,
                        "authors": "Unknown (from PDF)",
                        "venue": "Unknown",
                        "filename": file.name,
                        "score": 0.5
                    })

        elif file.name.endswith('.csv'):
            # Process CSV
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                papers.append(row.to_dict())

        elif file.name.endswith('.json'):
            # Process JSON
            data = json.load(file)
            if isinstance(data, list):
                papers.extend(data)
            else:
                papers.append(data)

    return papers


def make_annotation(decision, flagged=False, comment=""):
    """Record an annotation decision"""
    current_paper = st.session_state.papers[st.session_state.current_index]

    annotation = {
        "paper_id": current_paper.get('id'),
        "paper_index": st.session_state.current_index,
        "title": current_paper.get('title'),
        "decision": decision,
        "flagged": flagged,
        "comment": comment,
        "model_score": current_paper.get('score'),
        "timestamp": datetime.now().isoformat(),
        "annotator": "user"
    }

    # Update existing annotation or add new one
    existing_idx = None
    for i, ann in enumerate(st.session_state.annotations):
        if ann.get('paper_index') == st.session_state.current_index:
            existing_idx = i
            break

    if existing_idx is not None:
        st.session_state.annotations[existing_idx] = annotation
    else:
        st.session_state.annotations.append(annotation)
        st.session_state.session_stats['total_annotations'] += 1

    # Update stats
    if decision == "accept":
        st.session_state.session_stats['accept_count'] += 1
    elif decision == "reject":
        st.session_state.session_stats['reject_count'] += 1
    elif decision == "ignore":
        st.session_state.session_stats['ignore_count'] += 1

    if flagged:
        st.session_state.session_stats['flagged_count'] += 1


def auto_advance():
    """Move to next paper if auto-advance is enabled"""
    if (st.session_state.config['auto_advance'] and
            st.session_state.current_index < len(st.session_state.papers) - 1):
        st.session_state.current_index += 1
        st.rerun()


def calculate_annotation_speed():
    """Calculate annotations per minute"""
    elapsed = (datetime.now() -
               st.session_state.session_stats['start_time']).total_seconds() / 60
    if elapsed == 0:
        return 0
    return st.session_state.session_stats['total_annotations'] / elapsed


def export_annotations():
    """Export annotations to JSON file"""
    if not st.session_state.annotations:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"annotations_{timestamp}.json"

    Path("annotations").mkdir(exist_ok=True)

    export_data = {
        "annotations": st.session_state.annotations,
        "session_stats": st.session_state.session_stats,
        "export_timestamp": timestamp,
        "total_papers": len(st.session_state.papers)
    }

    with open(f"annotations/{filename}", 'w') as f:
        json.dump(export_data, f, indent=2)

    return filename


def render_sidebar():
    """Render the sidebar with project info and progress"""
    with st.sidebar:
        st.header("Project Info")

        # Dataset info
        if st.session_state.papers:
            st.metric("Total Papers", len(st.session_state.papers))
            st.metric("Current Position",
                      f"{st.session_state.current_index + 1}")

        st.markdown("---")

        # Progress section
        st.header("Progress")

        if st.session_state.annotations:
            stats = st.session_state.session_stats

            # Statistics grid
            st.markdown('<div class="stats-grid">', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-number">{stats['accept_count']}</div>
                    <div class="stat-label">Accept</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-number">{stats['ignore_count']}</div>
                    <div class="stat-label">Ignore</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-number">{stats['reject_count']}</div>
                    <div class="stat-label">Reject</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-number">{stats['flagged_count']}</div>
                    <div class="stat-label">Flagged</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Progress bar
            if st.session_state.papers:
                progress = len(st.session_state.annotations) / \
                    len(st.session_state.papers)
                st.progress(
                    progress, text=f"Progress: {len(st.session_state.annotations)}/{len(st.session_state.papers)}")

            # Speed metric
            speed = calculate_annotation_speed()
            st.metric("Speed (per min)", f"{speed:.1f}")

        st.markdown("---")

        # Controls
        st.header("Controls")

        # Export button
        if st.button("Export Annotations", type="primary"):
            filename = export_annotations()
            if filename:
                st.success(f"Exported: {filename}")
            else:
                st.warning("No annotations to export")

        # Configuration
        with st.expander("Settings"):
            st.session_state.config['show_flag'] = st.checkbox(
                "Show Flag Button",
                value=st.session_state.config['show_flag']
            )
            st.session_state.config['auto_advance'] = st.checkbox(
                "Auto Advance",
                value=st.session_state.config['auto_advance']
            )


def render_instructions():
    """Render annotation instructions"""
    if st.session_state.show_instructions:
        with st.expander("Annotation Instructions", expanded=True):
            st.markdown("""
            ### Paper Review Guidelines
            
            **Your task**: Evaluate each paper for conference acceptance.
            
            **Decisions**:
            - **Accept**: High-quality papers with significant contributions
            - **Reject**: Papers with major issues or insufficient quality  
            - **Ignore**: Skip for now (uncertain cases)
            - **Flag**: Mark for expert review or special attention
            
            **Quality Criteria**:
            - Technical soundness and methodology
            - Novelty and significance of contribution
            - Clarity of presentation and writing
            - Experimental validation and results
            - Relevance to conference scope
            
            **Keyboard Shortcuts**:
            - `Space` or `A`: Accept
            - `R` or `X`: Reject
            - `I`: Ignore  
            - `F`: Flag
            - `←/→`: Navigate papers
            
            **When to Flag**:
            - Potential plagiarism or ethics concerns
            - Papers requiring domain expert review
            - Technical issues preventing evaluation
            - Borderline cases needing discussion
            """)


def render_main_interface():
    """Render the main annotation interface"""
    if not st.session_state.papers:
        st.info("Load papers using the data loading section below to begin annotation.")
        return

    current_paper = st.session_state.papers[st.session_state.current_index]

    # Navigation and score
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])

    with col1:
        if st.button("← Previous", disabled=st.session_state.current_index == 0, key="prev_btn"):
            st.session_state.current_index -= 1
            st.rerun()

    with col2:
        st.markdown(
            f"### Paper {st.session_state.current_index + 1} of {len(st.session_state.papers)}")

    with col3:
        if st.button("Next →", disabled=st.session_state.current_index >= len(st.session_state.papers) - 1, key="next_btn"):
            st.session_state.current_index += 1
            st.rerun()

    with col4:
        score = current_paper.get('score', 0)
        st.markdown(
            f'<div class="score-badge">SCORE: {score:.2f}</div>', unsafe_allow_html=True)

    # Paper content
    st.markdown('<div class="paper-card">', unsafe_allow_html=True)

    st.markdown(f"**{current_paper.get('title', 'Untitled')}**")
    st.markdown(f"*{current_paper.get('authors', 'Unknown authors')}*")

    if current_paper.get('venue'):
        st.markdown(f"**Venue**: {current_paper['venue']}")

    if current_paper.get('filename'):
        st.markdown(f"**Source**: {current_paper['filename']}")

    st.markdown("---")
    st.markdown(current_paper.get('abstract', 'No abstract available'))

    # Show full text for PDFs
    if 'full_text' in current_paper:
        with st.expander("View Full Paper Text"):
            st.text_area(
                "Full text:", current_paper['full_text'], height=300, disabled=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Action buttons
    st.markdown('<div class="action-buttons">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✓ Accept", key="accept_btn", help="Accept paper (Space/A)"):
            make_annotation("accept")
            st.success("Accepted!")
            auto_advance()

    with col2:
        if st.button("✗ Reject", key="reject_btn", help="Reject paper (R/X)"):
            make_annotation("reject")
            st.error("Rejected!")
            auto_advance()

    with col3:
        if st.button("○ Ignore", key="ignore_btn", help="Skip paper (I)"):
            make_annotation("ignore")
            st.warning("Ignored!")
            auto_advance()

    with col4:
        if st.session_state.config['show_flag']:
            if st.button("🚩 Flag", key="flag_btn", help="Flag for review (F)"):
                make_annotation("flag", flagged=True)
                st.info("Flagged!")

    st.markdown('</div>', unsafe_allow_html=True)

    # Keyboard shortcuts hint
    st.markdown('<div class="keyboard-hint">Use keyboard: Space=Accept, R=Reject, I=Ignore, F=Flag, ←/→=Navigate</div>', unsafe_allow_html=True)

    # Show existing annotation
    existing_annotation = None
    for ann in st.session_state.annotations:
        if ann.get('paper_index') == st.session_state.current_index:
            existing_annotation = ann
            break

    if existing_annotation:
        st.info(
            f"Previously annotated: {existing_annotation['decision'].upper()}")
        if existing_annotation.get('flagged'):
            st.warning("This paper was flagged for review")


def render_data_loading():
    """Render data loading interface"""
    st.header("Data Loading")

    tab1, tab2 = st.tabs(["Sample Data", "Upload Files"])

    with tab1:
        st.markdown("Load sample papers for testing the annotation interface.")
        if st.button("Load Sample Papers", type="primary"):
            st.session_state.papers = load_sample_papers()
            st.session_state.current_index = 0
            st.success(f"Loaded {len(st.session_state.papers)} sample papers!")
            st.rerun()

    with tab2:
        st.markdown("Upload your own papers (PDF, CSV, or JSON files).")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'csv', 'json'],
            accept_multiple_files=True,
            help="Upload PDF papers, CSV data files, or JSON datasets"
        )

        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files:")
            for file in uploaded_files:
                file_type = "📄 PDF" if file.name.endswith('.pdf') else "📊 Data"
                st.write(f"- {file_type} {file.name}")

            if st.button("Process Files", type="primary"):
                with st.spinner("Processing files..."):
                    papers = process_uploaded_files(uploaded_files)
                    if papers:
                        st.session_state.papers = papers
                        st.session_state.current_index = 0
                        st.success(f"Loaded {len(papers)} papers!")
                        st.rerun()
                    else:
                        st.error(
                            "No papers could be loaded from the uploaded files.")


def main():
    """Main application function"""
    init_session_state()

    # Add keyboard shortcuts
    st.components.v1.html(keyboard_shortcuts, height=0)

    # Header
    st.title("Paper Annotation Tool")
    st.markdown(
        "*Prodigy-inspired interface for efficient scientific paper review*")

    # Instructions toggle
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📋 Instructions"):
            st.session_state.show_instructions = not st.session_state.show_instructions

    render_instructions()

    # Main layout
    render_sidebar()

    if st.session_state.papers:
        render_main_interface()
    else:
        render_data_loading()


if __name__ == "__main__":
    main()
