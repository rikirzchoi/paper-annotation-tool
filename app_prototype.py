import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import base64

# Configure page
st.set_page_config(
    page_title="Scientific Paper Annotation Tool",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for better styling and keyboard shortcuts
st.markdown("""
<style>
    .annotation-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .accept-btn {
        background-color: #28a745 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
    }
    .reject-btn {
        background-color: #dc3545 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
    }
    .ignore-btn {
        background-color: #6c757d !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
    }
    .flag-btn {
        background-color: #ffc107 !important;
        color: black !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
    }
    .score-display {
        font-size: 24px;
        font-weight: bold;
        text-align: right;
        color: #007bff;
    }
    .progress-container {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .keyboard-hint {
        background: #e9ecef;
        padding: 8px;
        border-radius: 4px;
        font-size: 12px;
        margin: 5px 0;
    }
    .annotation-stats {
        display: flex;
        justify-content: space-between;
        background: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript for keyboard shortcuts
keyboard_js = """
<script>
document.addEventListener('keydown', function(e) {
    // Prevent default browser shortcuts
    if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        switch(e.key) {
            case ' ':
            case 'a':
                e.preventDefault();
                // Trigger accept button
                document.querySelector('[data-testid="accept-btn"]')?.click();
                break;
            case 'r':
            case 'x':
                e.preventDefault();
                // Trigger reject button
                document.querySelector('[data-testid="reject-btn"]')?.click();
                break;
            case 'i':
                e.preventDefault();
                // Trigger ignore button
                document.querySelector('[data-testid="ignore-btn"]')?.click();
                break;
            case 'f':
                e.preventDefault();
                // Trigger flag button
                document.querySelector('[data-testid="flag-btn"]')?.click();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                document.querySelector('[data-testid="prev-btn"]')?.click();
                break;
            case 'ArrowRight':
                e.preventDefault();
                document.querySelector('[data-testid="next-btn"]')?.click();
                break;
        }
    }
});
</script>
"""

# Initialize session state


def initialize_session_state():
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'papers_data' not in st.session_state:
        st.session_state.papers_data = []
    if 'session_stats' not in st.session_state:
        st.session_state.session_stats = {
            'session_start': datetime.now(),
            'annotations_this_session': 0,
            'accept_count': 0,
            'reject_count': 0,
            'ignore_count': 0,
            'flagged_count': 0
        }
    if 'show_instructions' not in st.session_state:
        st.session_state.show_instructions = False
    if 'annotation_mode' not in st.session_state:
        st.session_state.annotation_mode = "detailed"  # or "rapid"


def load_sample_data():
    """Load sample scientific papers for annotation"""
    sample_papers = [
        {
            "id": "paper_001",
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "authors": "Vaswani et al.",
            "venue": "NIPS 2017",
            "keywords": ["transformer", "attention", "neural networks"],
            "score": 0.85  # Simulated ML model confidence
        },
        {
            "id": "paper_002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
            "authors": "Devlin et al.",
            "venue": "NAACL 2019",
            "keywords": ["BERT", "bidirectional", "pre-training", "transformers"],
            "score": 0.92
        },
        {
            "id": "paper_003",
            "title": "Deep Residual Learning for Image Recognition",
            "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
            "authors": "He et al.",
            "venue": "CVPR 2016",
            "keywords": ["residual networks", "deep learning", "computer vision"],
            "score": 0.78
        },
        {
            "id": "paper_004",
            "title": "Generative Adversarial Networks",
            "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
            "authors": "Goodfellow et al.",
            "venue": "NIPS 2014",
            "keywords": ["GANs", "generative models", "adversarial training"],
            "score": 0.95
        },
        {
            "id": "paper_005",
            "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
            "abstract": "Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional statistical machine translation, the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance.",
            "authors": "Bahdanau et al.",
            "venue": "ICLR 2015",
            "keywords": ["neural machine translation", "attention", "sequence-to-sequence"],
            "score": 0.67
        }
    ]
    return sample_papers


def save_annotations():
    """Save current annotations to file"""
    if st.session_state.annotations:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"annotations_{timestamp}.json"

        # Create annotations directory if it doesn't exist
        Path("annotations").mkdir(exist_ok=True)

        # Include session statistics
        export_data = {
            "annotations": st.session_state.annotations,
            "session_stats": st.session_state.session_stats,
            "export_timestamp": timestamp,
            "total_papers": len(st.session_state.papers_data),
            "annotation_mode": st.session_state.annotation_mode
        }

        with open(f"annotations/{filename}", 'w') as f:
            json.dump(export_data, f, indent=2)

        return filename
    return None


def load_custom_data(uploaded_file):
    """Load custom dataset from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            return df.to_dict('records')
        elif uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            return data if isinstance(data, list) else [data]
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def calculate_annotation_speed():
    """Calculate annotations per minute"""
    if st.session_state.session_stats['annotations_this_session'] == 0:
        return 0

    time_elapsed = (datetime.now(
    ) - st.session_state.session_stats['session_start']).total_seconds() / 60
    if time_elapsed == 0:
        return 0
    return st.session_state.session_stats['annotations_this_session'] / time_elapsed


def show_annotation_instructions():
    """Display annotation instructions"""
    with st.expander("📋 Annotation Instructions", expanded=st.session_state.show_instructions):
        st.markdown("""
        ### How to Annotate Scientific Papers
        
        **Your Task:** Review each paper and decide whether it should be accepted for the conference.
        
        **Guidelines:**
        - **Accept**: High-quality papers with novel contributions, clear methodology, and significant impact
        - **Weak Accept**: Good papers with minor issues but overall acceptable
        - **Weak Reject**: Papers with significant issues but some merit
        - **Reject**: Poor quality papers, lack of novelty, or major methodological flaws
        
        **Quality Criteria:**
        - Technical soundness and rigor
        - Novelty and significance of contribution
        - Clarity of presentation
        - Experimental validation (if applicable)
        - Relevance to conference scope
        
        **Keyboard Shortcuts:**
        - `Space` or `A`: Accept
        - `R` or `X`: Reject  
        - `I`: Ignore (skip for now)
        - `F`: Flag for review
        - `←/→`: Navigate between papers
        
        **When to Flag:**
        - Suspicious content or potential plagiarism
        - Papers requiring expert review
        - Unclear or ambiguous cases
        - Technical issues preventing proper evaluation
        """)


def record_annotation(decision, flagged=False):
    """Record an annotation and update statistics"""
    current_paper = st.session_state.papers_data[st.session_state.current_index]

    annotation = {
        "paper_id": current_paper.get('id', f"paper_{st.session_state.current_index}"),
        "paper_index": st.session_state.current_index,
        "title": current_paper.get('title'),
        "decision": decision,
        "flagged": flagged,
        "model_score": current_paper.get('score', None),
        "timestamp": datetime.now().isoformat(),
        "annotator": "current_user",
        "annotation_mode": st.session_state.annotation_mode
    }

    # Update or add annotation
    existing_index = None
    for i, ann in enumerate(st.session_state.annotations):
        if ann.get('paper_index') == st.session_state.current_index:
            existing_index = i
            break

    if existing_index is not None:
        st.session_state.annotations[existing_index] = annotation
    else:
        st.session_state.annotations.append(annotation)
        st.session_state.session_stats['annotations_this_session'] += 1

    # Update decision statistics
    if decision == "accept":
        st.session_state.session_stats['accept_count'] += 1
    elif decision == "reject":
        st.session_state.session_stats['reject_count'] += 1
    elif decision == "ignore":
        st.session_state.session_stats['ignore_count'] += 1

    if flagged:
        st.session_state.session_stats['flagged_count'] += 1


def main():
    initialize_session_state()

    # Add keyboard shortcuts
    st.components.v1.html(keyboard_js, height=0)

    st.title("📄 Scientific Paper Annotation Tool")
    st.markdown("*Prodigy-inspired interface for efficient paper review*")

    # Top toolbar
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        mode = st.selectbox(
            "Annotation Mode:",
            ["rapid", "detailed"],
            index=0 if st.session_state.annotation_mode == "rapid" else 1,
            key="mode_selector"
        )
        st.session_state.annotation_mode = mode

    with col2:
        if st.button("📋 Instructions"):
            st.session_state.show_instructions = not st.session_state.show_instructions

    with col3:
        if st.button("📊 Statistics"):
            show_stats = True

    with col4:
        if st.button("💾 Export All"):
            filename = save_annotations()
            if filename:
                st.success(f"Exported: {filename}")

    # Show instructions if enabled
    show_annotation_instructions()

    # Sidebar for controls and progress
    with st.sidebar:
        st.header("🎛️ Controls")

        # Data loading section
        st.subheader("Data Loading")
        data_option = st.radio(
            "Choose data source:",
            ["Sample Data", "Upload Custom Data"]
        )

        if data_option == "Sample Data":
            if st.button("Load Sample Papers"):
                st.session_state.papers_data = load_sample_data()
                st.session_state.data_loaded = True
                st.session_state.current_index = 0
                st.success("Sample data loaded!")

        else:
            uploaded_file = st.file_uploader(
                "Upload your dataset",
                type=['csv', 'json'],
                help="CSV should have columns: title, abstract, authors, etc."
            )

            if uploaded_file is not None:
                if st.button("Load Custom Data"):
                    custom_data = load_custom_data(uploaded_file)
                    if custom_data:
                        st.session_state.papers_data = custom_data
                        st.session_state.data_loaded = True
                        st.session_state.current_index = 0
                        st.success("Custom data loaded!")

        # Progress section
        if st.session_state.data_loaded:
            st.markdown("---")
            st.subheader("📊 Progress")

            total_papers = len(st.session_state.papers_data)
            current_pos = st.session_state.current_index + 1
            annotated_count = len(st.session_state.annotations)

            # Progress metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current", f"{current_pos}/{total_papers}")
                st.metric(
                    "Accept", st.session_state.session_stats['accept_count'])
                st.metric(
                    "Flagged", st.session_state.session_stats['flagged_count'])

            with col2:
                st.metric("Annotated", annotated_count)
                st.metric(
                    "Reject", st.session_state.session_stats['reject_count'])
                st.metric("Speed/min", f"{calculate_annotation_speed():.1f}")

            # Progress bars
            if total_papers > 0:
                overall_progress = current_pos / total_papers
                annotation_progress = annotated_count / total_papers

                st.write("**Overall Progress**")
                st.progress(overall_progress)

                st.write("**Annotation Progress**")
                st.progress(annotation_progress)

        # Keyboard shortcuts reminder
        st.markdown("---")
        st.subheader("⌨️ Shortcuts")
        st.markdown("""
        - `Space/A`: Accept
        - `R/X`: Reject
        - `I`: Ignore
        - `F`: Flag
        - `←/→`: Navigate
        """)

    # Main content area
    if not st.session_state.data_loaded:
        st.info("👈 Please load data from the sidebar to begin annotation")
        st.markdown("""
        ### Getting Started
        1. Choose to load sample data or upload your own dataset
        2. Select annotation mode (rapid for quick decisions, detailed for comprehensive review)
        3. Use keyboard shortcuts for efficient annotation
        4. Flag papers that need special attention
        5. Export your annotations when complete
        """)
        return

    # Check if we have data
    if not st.session_state.papers_data:
        st.error("No papers loaded. Please load data first.")
        return

    current_paper = st.session_state.papers_data[st.session_state.current_index]

    # Navigation and score display
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])

    with col1:
        if st.button("⬅️ Previous", disabled=st.session_state.current_index == 0, key="prev_btn"):
            st.session_state.current_index -= 1
            st.rerun()

    with col2:
        st.markdown(
            f"<h3 style='text-align: center'>Paper {st.session_state.current_index + 1} of {len(st.session_state.papers_data)}</h3>", unsafe_allow_html=True)

    with col3:
        if st.button("Next ➡️", disabled=st.session_state.current_index >= len(st.session_state.papers_data) - 1, key="next_btn"):
            st.session_state.current_index += 1
            st.rerun()

    with col4:
        # Model confidence score (Prodigy-style)
        score = current_paper.get('score', 0)
        st.markdown(
            f"<div class='score-display'>SCORE: {score:.2f}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Paper display in card format
    with st.container():
        st.markdown("<div class='annotation-card'>", unsafe_allow_html=True)

        # Paper details
        st.subheader("📑 Paper Details")
        st.markdown(f"**Title:** {current_paper.get('title', 'N/A')}")
        st.markdown(f"**Authors:** {current_paper.get('authors', 'N/A')}")
        if 'venue' in current_paper:
            st.markdown(f"**Venue:** {current_paper.get('venue')}")
        if 'keywords' in current_paper:
            keywords = current_paper.get('keywords', [])
            if keywords:
                st.markdown(f"**Keywords:** {', '.join(keywords)}")

        st.markdown("**Abstract:**")
        st.markdown(
            f"*{current_paper.get('abstract', 'No abstract available')}*")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Rapid annotation mode (Prodigy-style)
    if st.session_state.annotation_mode == "rapid":
        st.subheader("🚀 Rapid Annotation")

        # Large action buttons
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("✅ ACCEPT", key="accept_btn", help="Accept this paper (Space/A)"):
                record_annotation("accept")
                st.success("Accepted!")
                if st.session_state.current_index < len(st.session_state.papers_data) - 1:
                    st.session_state.current_index += 1
                    st.rerun()

        with col2:
            if st.button("❌ REJECT", key="reject_btn", help="Reject this paper (R/X)"):
                record_annotation("reject")
                st.error("Rejected!")
                if st.session_state.current_index < len(st.session_state.papers_data) - 1:
                    st.session_state.current_index += 1
                    st.rerun()

        with col3:
            if st.button("⭕ IGNORE", key="ignore_btn", help="Skip for now (I)"):
                record_annotation("ignore")
                st.warning("Ignored!")
                if st.session_state.current_index < len(st.session_state.papers_data) - 1:
                    st.session_state.current_index += 1
                    st.rerun()

        with col4:
            if st.button("🚩 FLAG", key="flag_btn", help="Flag for review (F)"):
                record_annotation("flag", flagged=True)
                st.info("Flagged for review!")

        with col5:
            # Comments for flagged items
            if st.button("💬 Comment"):
                st.session_state.show_comment_box = not getattr(
                    st.session_state, 'show_comment_box', False)

        # Quick comment box
        if getattr(st.session_state, 'show_comment_box', False):
            comment = st.text_area(
                "Quick comment:", placeholder="Why did you flag this paper?", height=80)
            if st.button("Save Comment"):
                # Add comment to last annotation
                if st.session_state.annotations:
                    st.session_state.annotations[-1]['comment'] = comment
                    st.success("Comment saved!")
                    st.session_state.show_comment_box = False
                    st.rerun()

    # Detailed annotation mode
    else:
        st.subheader("🔍 Detailed Annotation")

        col1, col2 = st.columns(2)

        with col1:
            quality_score = st.selectbox(
                "Paper Quality",
                options=[1, 2, 3, 4, 5],
                index=2,
                help="Rate the overall quality (1=Poor, 5=Excellent)"
            )

            relevance = st.selectbox(
                "Relevance to Conference",
                options=["High", "Medium", "Low"],
                index=1
            )

            novelty = st.selectbox(
                "Novelty",
                options=["Highly Novel", "Somewhat Novel", "Not Novel"],
                index=1
            )

        with col2:
            categories = st.multiselect(
                "Categories",
                options=["Machine Learning", "Natural Language Processing", "Computer Vision",
                         "Deep Learning", "Reinforcement Learning", "Theory", "Applications"],
                help="Select relevant categories"
            )

            decision = st.selectbox(
                "Final Decision",
                options=["Accept", "Weak Accept", "Weak Reject", "Reject"],
                index=1
            )

            confidence = st.slider(
                "Confidence",
                min_value=1,
                max_value=5,
                value=3,
                help="How confident are you? (1=Low, 5=High)"
            )

        # Comments and flagging
        col1, col2 = st.columns([3, 1])

        with col1:
            comments = st.text_area(
                "Comments",
                placeholder="Detailed feedback about this paper...",
                height=100
            )

        with col2:
            flagged = st.checkbox("🚩 Flag for Review",
                                  help="Mark for special attention")
            if flagged:
                flag_reason = st.selectbox(
                    "Flag Reason:",
                    ["Quality Concern", "Potential Plagiarism",
                        "Needs Expert Review", "Technical Issues", "Other"]
                )

        # Save detailed annotation
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Detailed Annotation", type="primary"):
                detailed_annotation = {
                    "paper_id": current_paper.get('id', f"paper_{st.session_state.current_index}"),
                    "paper_index": st.session_state.current_index,
                    "title": current_paper.get('title'),
                    "quality_score": quality_score,
                    "relevance": relevance,
                    "novelty": novelty,
                    "categories": categories,
                    "decision": decision,
                    "confidence": confidence,
                    "comments": comments,
                    "flagged": flagged,
                    "flag_reason": flag_reason if flagged else None,
                    "model_score": current_paper.get('score'),
                    "timestamp": datetime.now().isoformat(),
                    "annotator": "current_user",
                    "annotation_mode": "detailed"
                }

                # Update or add annotation
                existing_index = None
                for i, ann in enumerate(st.session_state.annotations):
                    if ann.get('paper_index') == st.session_state.current_index:
                        existing_index = i
                        break

                if existing_index is not None:
                    st.session_state.annotations[existing_index] = detailed_annotation
                    st.success("Annotation updated!")
                else:
                    st.session_state.annotations.append(detailed_annotation)
                    st.session_state.session_stats['annotations_this_session'] += 1
                    st.success("Annotation saved!")

        with col2:
            if st.button("💾 Save & Next"):
                # Save and move to next (similar logic as above)
                # ... (same annotation saving logic)
                if st.session_state.current_index < len(st.session_state.papers_data) - 1:
                    st.session_state.current_index += 1
                    st.rerun()

    # Show existing annotation if available
    existing_annotation = None
    for ann in st.session_state.annotations:
        if ann.get('paper_index') == st.session_state.current_index:
            existing_annotation = ann
            break

    if existing_annotation:
        st.markdown("---")
        st.info("📝 This paper has been previously annotated")
        with st.expander("View Previous Annotation"):
            st.json(existing_annotation)


if __name__ == "__main__":
    main()
