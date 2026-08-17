#!/usr/bin/env python3
"""Quant-GEMM Kernel Bench - Streamlit Web UI"""
import streamlit as st
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Quant-GEMM Kernel Bench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #0d6efd;
        margin-bottom: 0.5rem;
    }
    .kernel-name {
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #495057;
        border-bottom: 2px solid #dee2e6;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: #e7f1ff;
        color: #0d6efd;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .tag-verified {
        background-color: rgba(25, 135, 84, 0.15);
        color: #198754;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0d6efd;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
    }
    .code-block {
        background-color: #f6f8fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1rem;
        font-family: 'SF Mono', Monaco, monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }
    .field-table {
        width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Get definitions directory
DEFINITIONS_DIR = Path(__file__).parent.parent / "definitions"

def load_kernel_definitions():
    """Load all kernel definition JSON files."""
    kernels = {}

    # Scan all subdirectories
    for category_dir in DEFINITIONS_DIR.iterdir():
        if category_dir.is_dir():
            for json_file in category_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        kernels[data.get('name', json_file.stem)] = {
                            'data': data,
                            'category': category_dir.name,
                            'path': str(json_file)
                        }
                except Exception as e:
                    st.error(f"Error loading {json_file}: {e}")

    return kernels

def render_tags(tags):
    """Render tags as HTML."""
    html = ""
    for tag in tags:
        css_class = "tag tag-verified" if "verified" in tag else "tag"
        html += f'<span class="{css_class}">{tag}</span>'
    return html

def render_axes(axes):
    """Render axes parameters."""
    cols = st.columns(len(axes))
    for i, (name, axis) in enumerate(axes.items()):
        with cols[i]:
            value = axis.get('value', 'variable')
            axis_type = axis.get('type', 'unknown')
            desc = axis.get('description', '')

            st.metric(
                label=name,
                value=value if value != 'variable' else 'var',
                delta=axis_type,
                delta_color="off"
            )
            if desc:
                st.caption(desc)

def render_tensors(tensors, title):
    """Render input/output tensors as a table."""
    if not tensors:
        return

    st.markdown(f"**{title}**")

    data = []
    for name, tensor in tensors.items():
        shape = tensor.get('shape', [])
        if isinstance(shape, list):
            shape_str = f"[{', '.join(str(s) for s in shape)}]"
        else:
            shape_str = str(shape)

        data.append({
            "Name": f"`{name}`",
            "Shape": f"`{shape_str}`",
            "Type": f"`{tensor.get('dtype', 'unknown')}`",
            "Description": tensor.get('description', '')
        })

    st.table(data)

def render_types(types):
    """Render type definitions."""
    for type_name, type_def in types.items():
        with st.expander(f"📦 {type_name} ({type_def.get('size', '?')} bytes)", expanded=False):
            # Fields
            if 'fields' in type_def:
                st.markdown("**Fields:**")
                fields_data = []
                for field in type_def['fields']:
                    fields_data.append({
                        "Name": f"`{field.get('name', '')}`",
                        "Type": f"`{field.get('dtype', '')}`",
                        "Size": f"{field.get('size', '?')} bytes",
                        "Description": field.get('description', '')
                    })
                st.table(fields_data)

            # Metadata
            if 'quantization' in type_def:
                st.markdown(f"**Quantization:** `{type_def['quantization']}`")
            if 'formula' in type_def:
                st.markdown(f"**Formula:** `{type_def['formula']}`")

def render_constraints(constraints):
    """Render constraints."""
    for constraint in constraints:
        st.code(constraint, language=None)

def render_formula(formula):
    """Render formula."""
    for key, value in formula.items():
        label = key.replace('_', ' ').title()
        st.markdown(f"**{label}:**")
        st.code(value, language=None)

def render_reference(reference):
    """Render reference implementation."""
    st.code(reference, language="python")

def main():
    # Sidebar
    st.sidebar.markdown("## ⚡ Quant-GEMM Bench")
    st.sidebar.markdown("---")

    # Load kernels
    kernels = load_kernel_definitions()

    if not kernels:
        st.error("No kernel definitions found!")
        return

    # Group by category
    categories = {}
    for name, info in kernels.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)

    # Sidebar navigation
    st.sidebar.markdown("### Kernel Definitions")

    selected_kernel = None
    for category, kernel_names in sorted(categories.items()):
        st.sidebar.markdown(f"**{category}**")
        for name in sorted(kernel_names):
            if st.sidebar.button(name, key=f"btn_{name}", use_container_width=True):
                st.session_state.selected_kernel = name

    # Get selected kernel from session state
    if 'selected_kernel' not in st.session_state:
        st.session_state.selected_kernel = list(kernels.keys())[0]

    selected_kernel = st.session_state.selected_kernel

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    Quantized GEMM kernel specification viewer.

    Similar to [FlashInfer Bench](https://bench.flashinfer.ai)
    """)

    # Main content
    if selected_kernel and selected_kernel in kernels:
        data = kernels[selected_kernel]['data']
        category = kernels[selected_kernel]['category']

        # Header
        st.markdown(f'<p class="kernel-name">{data.get("name", selected_kernel)}</p>', unsafe_allow_html=True)

        # Description
        if 'description' in data:
            st.markdown(data['description'])

        # Tags
        if 'tags' in data:
            st.markdown(render_tags(data['tags']), unsafe_allow_html=True)

        st.markdown("---")

        # Create tabs
        tabs = st.tabs(["📊 Specification", "📝 Types", "💻 Reference Code", "📄 Raw JSON"])

        # Tab 1: Specification
        with tabs[0]:
            # Axes
            if 'axes' in data:
                st.markdown("### Axes Parameters")
                render_axes(data['axes'])
                st.markdown("")

            # Inputs & Outputs
            col1, col2 = st.columns(2)
            with col1:
                if 'inputs' in data:
                    st.markdown("### Inputs")
                    render_tensors(data['inputs'], "")

            with col2:
                if 'outputs' in data:
                    st.markdown("### Outputs")
                    render_tensors(data['outputs'], "")

            # Constraints
            if 'constraints' in data and data['constraints']:
                st.markdown("### Constraints")
                for c in data['constraints']:
                    st.code(c, language=None)

            # Formula
            if 'formula' in data:
                st.markdown("### Formula")
                render_formula(data['formula'])

        # Tab 2: Types
        with tabs[1]:
            if 'types' in data:
                st.markdown("### Type Definitions")
                render_types(data['types'])
            else:
                st.info("No type definitions in this kernel.")

        # Tab 3: Reference Code
        with tabs[2]:
            if 'reference' in data:
                st.markdown("### Reference Implementation (Python)")
                st.code(data['reference'], language="python")
            else:
                st.info("No reference implementation available.")

        # Tab 4: Raw JSON
        with tabs[3]:
            st.json(data)

if __name__ == "__main__":
    main()
