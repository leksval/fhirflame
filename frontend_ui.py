import gradio as gr
import pandas as pd
import time
import threading
import asyncio
import sys
import os
import datetime
from src.heavy_workload_demo import ModalContainerScalingDemo, RealTimeBatchProcessor

# Import dashboard functions from app.py to ensure proper integration
sys.path.append(os.path.dirname(__file__))
# Use dynamic import to avoid circular dependency issues
dashboard_state = None
add_file_to_dashboard = None
get_dashboard_status = None
get_processing_queue = None
get_dashboard_metrics = None
get_jobs_history = None

def _ensure_app_imports():
    """Dynamically import app functions to avoid circular dependencies"""
    global dashboard_state, add_file_to_dashboard, get_dashboard_status
    global get_processing_queue, get_dashboard_metrics, get_jobs_history
    
    if dashboard_state is None:
        try:
            from app import (
                dashboard_state as _dashboard_state,
                add_file_to_dashboard as _add_file_to_dashboard,
                get_dashboard_status as _get_dashboard_status,
                get_processing_queue as _get_processing_queue,
                get_dashboard_metrics as _get_dashboard_metrics,
                get_jobs_history as _get_jobs_history
            )
            dashboard_state = _dashboard_state
            add_file_to_dashboard = _add_file_to_dashboard
            get_dashboard_status = _get_dashboard_status
            get_processing_queue = _get_processing_queue
            get_dashboard_metrics = _get_dashboard_metrics
            get_jobs_history = _get_jobs_history
        except ImportError as e:
            print(f"Warning: Could not import dashboard functions: {e}")
            # Set fallback functions that return empty data
            dashboard_state = {"active_tasks": 0, "total_files": 0}
            add_file_to_dashboard = lambda *args, **kwargs: None
            get_dashboard_status = lambda: "📊 Dashboard not available"
            get_processing_queue = lambda: [["Status", "Not Available"]]
            get_dashboard_metrics = lambda: [["Metric", "Not Available"]]
            get_jobs_history = lambda: []

# Initialize demo components
heavy_workload_demo = ModalContainerScalingDemo()
batch_processor = RealTimeBatchProcessor()

# Global reference to dashboard function (set by create_medical_ui)
_add_file_to_dashboard = None

def is_modal_available():
    """Check if Modal environment is available"""
    try:
        import modal
        return True
    except ImportError:
        return False

def get_environment_name():
    """Get current deployment environment name"""
    if is_modal_available():
        return "Modal Cloud"
    else:
        return "Local/HuggingFace"

def create_text_processing_tab(process_text_only, cancel_current_task, get_dashboard_status,
                              dashboard_state, get_dashboard_metrics):
    """Create the text processing tab"""
    
    with gr.Tab("📝 Text Processing"):
        gr.Markdown("### Medical Text Analysis")
        gr.Markdown("Process medical text directly with entity extraction and FHIR generation")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Medical Text Input")
                text_input = gr.Textbox(
                    label="Medical Text",
                    placeholder="Enter medical text here...",
                    lines=8
                )
                
                enable_fhir_text = gr.Checkbox(
                    label="Generate FHIR Resources",
                    value=False
                )
                
                with gr.Row():
                    process_text_btn = gr.Button("🔍 Process Text", variant="primary")
                    cancel_text_btn = gr.Button("❌ Cancel", variant="secondary", visible=False)
            
            with gr.Column():
                gr.Markdown("### Results")
                text_status = gr.HTML(value="🔄 Ready to process")
                
                with gr.Accordion("🔍 Entities", open=True):
                    extracted_entities = gr.JSON(label="Entities")
                
                with gr.Accordion("🏥 FHIR", open=True):
                    fhir_resources = gr.JSON(label="FHIR Data")
                    
        return {
            "text_input": text_input,
            "enable_fhir_text": enable_fhir_text,
            "process_text_btn": process_text_btn,
            "cancel_text_btn": cancel_text_btn,
            "text_status": text_status,
            "extracted_entities": extracted_entities,
            "fhir_resources": fhir_resources
        }

def create_document_upload_tab(process_file_only, cancel_current_task, get_dashboard_status,
                              dashboard_state, get_dashboard_metrics):
    """Create the document upload tab"""
    
    with gr.Tab("📄 Document Upload"):
        gr.Markdown("### Document Processing")
        gr.Markdown("Upload and process medical documents with comprehensive analysis")
        gr.Markdown("**Supported formats:** PDF, DOCX, DOC, TXT, JPG, JPEG, PNG, GIF, BMP, WEBP, TIFF")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Document Upload")
                file_input = gr.File(
                    label="Upload Medical Document",
                    file_types=[".pdf", ".docx", ".doc", ".txt", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"]
                )
                
                enable_mistral_ocr = gr.Checkbox(
                    label="🔍 Enable Mistral OCR (Advanced OCR for Images/PDFs)",
                    value=True,
                    info="Uses Mistral API for enhanced OCR processing of images and scanned documents"
                )
                
                enable_fhir_file = gr.Checkbox(
                    label="Generate FHIR Resources",
                    value=False
                )
                
                with gr.Row():
                    process_file_btn = gr.Button("📄 Process File", variant="primary")
                    cancel_file_btn = gr.Button("❌ Cancel", variant="secondary", visible=False)
            
            with gr.Column():
                gr.Markdown("### Results")
                file_status = gr.HTML(value="Ready to process documents")
                
                with gr.Accordion("🔍 Entities", open=True):
                    file_entities = gr.JSON(label="Entities")
                
                with gr.Accordion("🏥 FHIR", open=True):
                    file_fhir = gr.JSON(label="FHIR Data")
                    
        return {
            "file_input": file_input,
            "enable_mistral_ocr": enable_mistral_ocr,
            "enable_fhir_file": enable_fhir_file,
            "process_file_btn": process_file_btn,
            "cancel_file_btn": cancel_file_btn,
            "file_status": file_status,
            "file_entities": file_entities,
            "file_fhir": file_fhir
        }

def create_dicom_processing_tab(process_dicom_only, cancel_current_task, get_dashboard_status,
                               dashboard_state, get_dashboard_metrics):
    """Create the DICOM processing tab"""
    
    with gr.Tab("🏥 DICOM Processing"):
        gr.Markdown("### Medical Imaging Analysis")
        gr.Markdown("Process DICOM files for medical imaging analysis and metadata extraction")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### DICOM Upload")
                dicom_input = gr.File(
                    label="Upload DICOM File",
                    file_types=[".dcm", ".dicom"]
                )
                
                with gr.Row():
                    process_dicom_btn = gr.Button("🏥 Process DICOM", variant="primary")
                    cancel_dicom_btn = gr.Button("❌ Cancel", variant="secondary", visible=False)
            
            with gr.Column():
                gr.Markdown("### Results")
                dicom_status = gr.HTML(value="Ready to process DICOM files")
                
                with gr.Accordion("📊 DICOM Analysis", open=False):
                    dicom_analysis = gr.JSON(label="DICOM Metadata & Analysis")
                
                with gr.Accordion("🏥 FHIR Imaging", open=True):
                    dicom_fhir = gr.JSON(label="FHIR ImagingStudy")
                    
        return {
            "dicom_input": dicom_input,
            "process_dicom_btn": process_dicom_btn,
            "cancel_dicom_btn": cancel_dicom_btn,
            "dicom_status": dicom_status,
            "dicom_analysis": dicom_analysis,
            "dicom_fhir": dicom_fhir
        }

def create_heavy_workload_tab():
    """Create the heavy workload demo tab"""
    
    with gr.Tab("🚀 Heavy Workload Demo"):
        if is_modal_available():
            # Demo title
            gr.Markdown("## 🚀 FhirFlame Modal Container Auto-Scaling Demo")
            gr.Markdown(f"**Environment:** {get_environment_name()}")
            gr.Markdown("This demo showcases automatic horizontal scaling of containers based on workload.")
            
            # Demo controls
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Demo Controls")
                    
                    container_table = gr.Dataframe(
                        headers=["Container ID", "Region", "Status", "Requests/sec", "Queue", "Processed", "Entities", "FHIR", "Uptime"],
                        datatype=["str", "str", "str", "str", "number", "number", "number", "number", "str"],
                        label="📊 Active Containers",
                        interactive=False
                    )
                    
                    with gr.Row():
                        start_demo_btn = gr.Button("🚀 Start Modal Container Scaling", variant="primary")
                        stop_demo_btn = gr.Button("⏹️ Stop Demo", variant="secondary", visible=False)
                        refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                    
                with gr.Column():
                    gr.Markdown("### Scaling Metrics")
                    
                    scaling_metrics = gr.Dataframe(
                        headers=["Metric", "Value"],
                        label="📈 Scaling Status",
                        interactive=False
                    )
                    
                    workload_chart = gr.Plot(label="📊 Workload & Scaling Chart")
            
            # Event handlers with button state management
            def start_demo_with_state():
                result = start_heavy_workload()
                return result + (gr.update(visible=True),)  # Show stop button
            
            def stop_demo_with_state():
                result = stop_heavy_workload()
                return result + (gr.update(visible=False),)  # Hide stop button
            
            start_demo_btn.click(
                fn=start_demo_with_state,
                outputs=[container_table, scaling_metrics, workload_chart, stop_demo_btn]
            )
            
            stop_demo_btn.click(
                fn=stop_demo_with_state,
                outputs=[container_table, scaling_metrics, workload_chart, stop_demo_btn]
            )
            
            refresh_btn.click(
                fn=refresh_demo_data,
                outputs=[container_table, scaling_metrics, workload_chart]
            )
            
        else:
            gr.Markdown("## ⚠️ Modal Environment Not Available")
            gr.Markdown("This demo requires Modal cloud environment to showcase container scaling.")
            gr.Markdown("Currently running in: **Local/HuggingFace Environment**")
            
            # Show static placeholder
            placeholder_data = [
                ["container-1", "us-east", "Simulated", "45", 12, 234, 1890, 45, "2h 34m"],
                ["container-2", "us-west", "Simulated", "67", 8, 456, 3245, 89, "1h 12m"],
                ["container-3", "eu-west", "Simulated", "23", 3, 123, 987, 23, "45m"]
            ]
            
            gr.Dataframe(
                value=placeholder_data,
                headers=["Container ID", "Region", "Status", "Requests/sec", "Queue", "Processed", "Entities", "FHIR", "Uptime"],
                label="📊 Demo Container Data (Simulated)",
                interactive=False
            )

def create_system_stats_tab(get_simple_agent_status):
    """Create the system stats tab"""
    
    with gr.Tab("📊 System Dashboard"):
        gr.Markdown("## System Status & Metrics")
        gr.Markdown("*Updates when tasks complete or fail*")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🖥️ System Status")
                
                agent_status_display = gr.HTML(
                    value=get_simple_agent_status()
                )
                
                with gr.Row():
                    refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                
                last_updated_display = gr.HTML(
                    value="<p><small>Last updated: Never</small></p>"
                )
            
            with gr.Column():
                gr.Markdown("### 📁 File Processing Dashboard")
                
                processing_status = gr.HTML(
                    value="<p>📊 No files processed yet</p>"
                )
                
                metrics_display = gr.DataFrame(
                    value=[["Total Files", 0], ["Success Rate", "0%"], ["Last Update", "None"]],
                    headers=["Metric", "Value"],
                    label="📊Metrics",
                    interactive=False
                )
                
                # Add processed jobs history
                gr.Markdown("### 📋 Recent Processing Jobs")
                jobs_history_display = gr.DataFrame(
                    value=[],
                    headers=["Job Name", "Category", "Status", "Processing Time"],
                    label="⚙️Processing Jobs History",
                    interactive=False,
                    column_widths=["50%", "20%", "15%", "15%"]
                )
                
                # Add database management section
                gr.Markdown("### 🗂️ Database Management")
                with gr.Row():
                    clear_db_btn = gr.Button("🗑️ Clear Database", variant="secondary", size="sm")
                    clear_status = gr.Markdown("", visible=False)
                
                def clear_database():
                    try:
                        # Import database functions
                        from database import clear_all_jobs
                        clear_all_jobs()
                        return gr.update(value="✅ Database cleared successfully!", visible=True)
                    except Exception as e:
                        return gr.update(value=f"❌ Error clearing database: {e}", visible=True)
                
                clear_db_btn.click(
                    fn=clear_database,
                    outputs=clear_status
                )
                
    return {
        "agent_status_display": agent_status_display,
        "refresh_status_btn": refresh_status_btn,
        "last_updated_display": last_updated_display,
        "processing_status": processing_status,
        "metrics_display": metrics_display,
        "files_history": jobs_history_display
    }

def create_medical_ui(process_text_only, process_file_only, process_dicom_only,
                     cancel_current_task, get_dashboard_status, dashboard_state,
                     get_dashboard_metrics, get_simple_agent_status,
                     get_enhanced_codellama, add_file_to_dashboard):
    """Create the main medical interface with all tabs"""
    global _add_file_to_dashboard
    _add_file_to_dashboard = add_file_to_dashboard
    
    # Clean, organized CSS for FhirFlame branding
    logo_css = """
    <style>
    /* ====== LOGO STYLING ====== */
    .fhirflame-logo-zero-padding img {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
        padding: 0 !important;
        margin: 0 !important;
        display: block !important;
    }
    
    .fhirflame-subtitle {
        color: var(--body-text-color-subdued, #474747);
        font-size: 16px;
        font-weight: normal;
        line-height: 1.5;
        text-align: left;
        max-width: 800px;
        margin: 0;
        padding: 0;
        display: block;
    }
    
    .fhirflame-mvp-text {
        color: var(--body-text-color) !important;
        opacity: 0.7 !important;
        font-weight: 500 !important;
    }
    
    /* ====== BRAND COLORS ====== */
    /* Primary buttons - red */
    button[data-variant="primary"],
    .gr-button[data-variant="primary"],
    .gr-button-primary,
    .primary {
        background: #B71C1C !important;
        border-color: #B71C1C !important;
    }
    
    button[data-variant="primary"]:hover,
    .gr-button[data-variant="primary"]:hover,
    .gr-button-primary:hover {
        background: #9B1B1B !important;
        border-color: #9B1B1B !important;
    }
    
    /* Selected tabs - red with BLACK underlines */
    .gr-tab-nav button.selected,
    button[role="tab"][aria-selected="true"],
    .gr-tabs button.selected,
    .gr-tabs .gr-tab-nav button[aria-selected="true"] {
        background: #B71C1C !important;
        border-color: #B71C1C !important;
        color: white !important;
        border-bottom: 3px solid #000000 !important;
    }
    
    /* Tab underlines and borders - BLACK */
    .gr-tab-nav button.selected::after,
    .gr-tab-nav button:focus::after,
    .gr-tab-nav button:active::after,
    button[role="tab"][aria-selected="true"]::after,
    .gr-tabs button.selected::after,
    .gr-tabs button:hover::after,
    .gr-tabs button:focus::after,
    .gr-tabs button:active::after {
        background: #000000 !important;
        border-color: #000000 !important;
        border-bottom-color: #000000 !important;
    }
    
    /* Tab containers and nav */
    .gr-tab-nav,
    .gr-tabs {
        border-bottom: 1px solid #000000 !important;
    }
    
    /* Checkboxes - red */
    input[type="checkbox"]:checked,
    .gr-checkbox input:checked {
        background-color: #B71C1C !important;
        border-color: #B71C1C !important;
        accent-color: #B71C1C !important;
    }
    
    /* Progress bars - red */
    .progress-bar,
    .gr-progress,
    [role="progressbar"] {
        background-color: #B71C1C !important;
    }
    
    /* Links - red */
    a {
        color: #B71C1C !important;
    }
    
    a:hover {
        color: #9B1B1B !important;
    }
    
    /* ====== SLIDERS - BLACK ULTRA AGGRESSIVE ====== */
    input[type="range"],
    .gr-slider input[type="range"],
    .gradio-container input[type="range"],
    div input[type="range"],
    span input[type="range"],
    * input[type="range"] {
        accent-color: #000000 !important;
        background: transparent !important;
    }
    
    input[type="range"]::-webkit-slider-thumb,
    .gr-slider input[type="range"]::-webkit-slider-thumb,
    .gradio-container input[type="range"]::-webkit-slider-thumb {
        background: #000000 !important;
        border-color: #000000 !important;
        color: #000000 !important;
    }
    
    input[type="range"]::-moz-range-thumb,
    .gr-slider input[type="range"]::-moz-range-thumb,
    .gradio-container input[type="range"]::-moz-range-thumb {
        background: #000000 !important;
        border-color: #000000 !important;
        color: #000000 !important;
    }
    
    input[type="range"]::-webkit-slider-runnable-track,
    input[type="range"]::-moz-range-track {
        background: linear-gradient(to right, #000000 0%, #000000 var(--value, 50%), #e0e0e0 var(--value, 50%), #e0e0e0 100%) !important;
    }
    
    /* Force all slider containers to use black */
    .gr-block input[type="range"],
    .gr-form input[type="range"],
    div[data-testid*="slider"] input[type="range"],
    div[data-testid*="range"] input[type="range"] {
        accent-color: #000000 !important;
    }
    
    /* ====== PREVENT BLACK BACKGROUNDS ON TEXT ====== */
    label,
    .gr-label,
    .gr-markdown,
    .gr-text,
    span,
    div:not(.gr-button):not([role="button"]) {
        background: transparent !important;
    }
    
    /* ====== THEME ADAPTATION ====== */
    .gr-form,
    .gr-block,
    .gradio-container {
        background: var(--background-fill-primary) !important;
        color: var(--body-text-color) !important;
    }
    
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3, .gr-markdown h4, .gr-markdown h5, .gr-markdown h6 {
        color: var(--body-text-color) !important;
    }
    
    .gr-markdown p, .gr-markdown span, .gr-markdown div {
        color: var(--body-text-color-subdued) !important;
    }
    
    /* ====== OVERRIDE ORANGE - NUCLEAR OPTION ====== */
    /* Override CSS variables */
    :root {
        --slider-color: #000000 !important;
        --accent-color: #000000 !important;
        --primary-hue: 0 !important;
        --primary-sat: 100% !important;
        --primary-lit: 27% !important;
        --color-orange: #000000 !important;
        --primary-500: #B71C1C !important;
        --primary-600: #B71C1C !important;
    }
    
    /* Target ALL orange styles - BLACK in light mode, RED in dark mode */
    *[style*="rgb(255, 165, 0)"],
    *[style*="rgb(255,165,0)"],
    *[style*="#ff8c00"],
    *[style*="#ffa500"],
    *[style*="orange"],
    *[style*="hsl(39"],
    *[style*="hsl(38"],
    *[style*="hsl(40"],
    *[class*="orange"],
    .orange,
    [data-color="orange"] {
        background-color: #000000 !important;
        color: #000000 !important;
        border-color: #000000 !important;
        accent-color: #000000 !important;
    }
    
    /* Dark mode: Orange elements should be RED */
    @media (prefers-color-scheme: dark) {
        *[style*="rgb(255, 165, 0)"],
        *[style*="rgb(255,165,0)"],
        *[style*="#ff8c00"],
        *[style*="#ffa500"],
        *[style*="orange"],
        *[style*="hsl(39"],
        *[style*="hsl(38"],
        *[style*="hsl(40"],
        *[class*="orange"],
        .orange,
        [data-color="orange"] {
            background-color: #B71C1C !important;
            color: #B71C1C !important;
            border-color: #B71C1C !important;
            accent-color: #B71C1C !important;
        }
    }
    
    /* Also handle Gradio's dark theme class */
    .dark *[style*="rgb(255, 165, 0)"],
    .dark *[style*="rgb(255,165,0)"],
    .dark *[style*="#ff8c00"],
    .dark *[style*="#ffa500"],
    .dark *[style*="orange"],
    .dark *[style*="hsl(39"],
    .dark *[style*="hsl(38"],
    .dark *[style*="hsl(40"],
    .dark *[class*="orange"],
    .dark .orange,
    .dark [data-color="orange"] {
        background-color: #B71C1C !important;
        color: #B71C1C !important;
        border-color: #B71C1C !important;
        accent-color: #B71C1C !important;
    }
    
    /* Slider-specific orange override */
    *[style*="rgb(255, 165, 0)"] input[type="range"],
    *[style*="orange"] input[type="range"],
    input[type="range"][style*="orange"],
    input[type="range"][style*="rgb(255, 165, 0)"] {
        accent-color: #000000 !important;
    }
    
    /* Dark mode: Slider-specific orange override */
    @media (prefers-color-scheme: dark) {
        *[style*="rgb(255, 165, 0)"] input[type="range"],
        *[style*="orange"] input[type="range"],
        input[type="range"][style*="orange"],
        input[type="range"][style*="rgb(255, 165, 0)"] {
            accent-color: #B71C1C !important;
        }
    }
    
    /* Also handle Gradio's dark theme class for sliders */
    .dark *[style*="rgb(255, 165, 0)"] input[type="range"],
    .dark *[style*="orange"] input[type="range"],
    .dark input[type="range"][style*="orange"],
    .dark input[type="range"][style*="rgb(255, 165, 0)"] {
        accent-color: #B71C1C !important;
    }
    
    /* Orange elements to red for buttons only */
    button[style*="orange"],
    .gr-button[style*="orange"],
    button[style*="rgb(255, 165, 0)"],
    .gr-button[style*="rgb(255, 165, 0)"] {
        background-color: #B71C1C !important;
        border-color: #B71C1C !important;
    }
    
    /* Force black on ALL accent colors */
    * {
        accent-color: #000000 !important;
    }
    
    /* But allow red for buttons */
    button, .gr-button, [role="button"] {
        accent-color: #B71C1C !important;
    }
    
    /* Fix Gradio settings modal alignment issues */
    .gradio-container .settings-panel,
    .gradio-container .modal,
    .gradio-container .sidebar {
        position: fixed !important;
        top: 0 !important;
        left: auto !important;
        right: 0 !important;
        z-index: 9999 !important;
        background: white !important;
        border: 1px solid #ccc !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
padding: 20px !important;
    width: 400px !important;
    max-height: 90vh !important;
    overflow-y: auto !important;
    font-family: Arial, sans-serif !important;
    border-radius: 8px !important;
    }
    
    </style>
    """
    
    with gr.Blocks(title="FhirFlame: Real-Time Medical AI Processing & FHIR Generation", css=logo_css) as demo:
        
        # FhirFlame Official Logo Header - Using exact-sized SVG (450×150px)
        gr.Image(
            value="fhirflame_logo_450x150.svg",
            type="filepath",
            height="105px",
            width="315px",
            show_label=False,
            show_download_button=False,
            show_fullscreen_button=False,
            show_share_button=False,
            container=False,
            interactive=False,
            elem_classes=["fhirflame-logo-zero-padding"]
        )
                
        # Subtitle below logo
        gr.HTML(f"""
        <div class="fhirflame-subtitle">
            <strong>Medical AI System Demonstration</strong><br>
            <strong>Dockerized Healthcare AI Platform: Local/Cloud/Hybrid Deployment + Agent/MCP Server + FHIR R4/R5 + DICOM Processing + CodeLlama Integration</strong><br>
            <span class="fhirflame-mvp-text">🚧 MVP/Prototype | Hackathon Submission</span>
        </div>
        """)
        
        # Main tab container - all tabs at the same level
        with gr.Tabs():
            
            # Create all main tabs
            text_components = create_text_processing_tab(
                process_text_only, cancel_current_task, get_dashboard_status,
                dashboard_state, get_dashboard_metrics
            )
            
            file_components = create_document_upload_tab(
                process_file_only, cancel_current_task, get_dashboard_status,
                dashboard_state, get_dashboard_metrics
            )
            
            dicom_components = create_dicom_processing_tab(
                process_dicom_only, cancel_current_task, get_dashboard_status,
                dashboard_state, get_dashboard_metrics
            )
            
            # Heavy Workload Demo Tab
            create_heavy_workload_tab()
            
            # Batch Processing Demo Tab - Need to create dashboard components first
            with gr.Tab("🔄 Batch Processing Demo"):
                # Dashboard function is already set globally in create_medical_ui
                
                gr.Markdown("## 🔄 Real-Time Medical Batch Processing")
                gr.Markdown("Demonstrates live batch processing of sample medical documents with real-time progress tracking (no OCR required)")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Batch Configuration")
                        
                        batch_size = gr.Slider(
                            minimum=5,
                            maximum=50,
                            step=5,
                            value=10,
                            label="Batch Size"
                        )
                        
                        processing_type = gr.Radio(
                            choices=["Clinical Notes Sample", "Lab Reports Sample", "Discharge Summaries Sample"],
                            value="Clinical Notes Sample",
                            label="Sample File Category"
                        )
                        
                        enable_live_updates = gr.Checkbox(
                            value=True,
                            label="Live Progress Updates"
                        )
                        
                        with gr.Row():
                            start_demo_btn = gr.Button("🚀 Start Live Processing", variant="primary")
                            stop_demo_btn = gr.Button("⏹️ Stop Processing", visible=False)
                    
                    with gr.Column():
                        gr.Markdown("### Live Progress")
                        batch_status = gr.Markdown("🔄 Ready to start batch processing")
                        
                        processing_log = gr.Textbox(
                            label="Processing Log",
                            lines=8,
                            interactive=False
                        )
                        
                        results_summary = gr.JSON(
                            label="Results Summary",
                            value=create_empty_results_summary()
                        )
                
                # Timer for real-time updates
                status_timer = gr.Timer(value=1.0, active=False)
                
                # Connect event handlers with button state management
                def start_processing_with_timer(batch_size, processing_type, enable_live_updates):
                    result = start_live_processing(batch_size, processing_type, enable_live_updates)
                    # Get dashboard updates
                    
                    # Activate timer for real-time updates
                    return result + (gr.update(visible=True), gr.Timer(active=True),
                                   get_dashboard_status() if get_dashboard_status else "<p>Dashboard not available</p>",
                                   
                                   get_dashboard_metrics() if get_dashboard_metrics else [])
                
                def stop_processing_with_timer():
                    result = stop_processing()
                    # Get dashboard updates
                    
                    # Deactivate timer when processing stops
                    return result + (gr.update(visible=False), gr.Timer(active=False),
                                   get_dashboard_status() if get_dashboard_status else "<p>Dashboard not available</p>",
                                   
                                   get_dashboard_metrics() if get_dashboard_metrics else [])
            
            # System Dashboard Tab - at the far right (after Batch Processing)
            stats_components = create_system_stats_tab(get_simple_agent_status)
            
            # Get processing queue and metrics from stats for batch processing integration
            processing_status = stats_components["processing_status"]
            metrics_display = stats_components["metrics_display"]
            
            # Connect batch processing timer and buttons
            files_history_component = stats_components["files_history"]
            status_timer.tick(
                fn=update_batch_status_realtime,
                outputs=[batch_status, processing_log, results_summary,
                        processing_status, metrics_display,
                        files_history_component]
            )
            
            start_demo_btn.click(
                fn=start_processing_with_timer,
                inputs=[batch_size, processing_type, enable_live_updates],
                outputs=[batch_status, processing_log, results_summary, stop_demo_btn, status_timer,
                        processing_status, metrics_display]
            )
            
            stop_demo_btn.click(
                fn=stop_processing_with_timer,
                outputs=[batch_status, processing_log, stop_demo_btn, status_timer,
                        processing_status, metrics_display]
            )
        
        # Enhanced event handlers with button state management
        def process_text_with_state(text_input, enable_fhir):
            # Ensure dashboard functions are available
            _ensure_app_imports()
            # Get core processing results (3 values)
            status, entities, fhir_resources = process_text_only(text_input, enable_fhir)
            # Return 7 values expected by Gradio outputs
            return (
                status, entities, fhir_resources,           # Core results (3)
                get_dashboard_status(),                     # Dashboard status (1)
                get_dashboard_metrics(),                    # Dashboard metrics (1)
                get_jobs_history(),                         # Jobs history (1)
                gr.update(visible=True)                     # Cancel button state (1)
            )

        def process_file_with_state(file_input, enable_mistral_ocr, enable_fhir):
            # Ensure dashboard functions are available
            _ensure_app_imports()
            # Get core processing results (3 values) - pass mistral_ocr parameter
            status, entities, fhir_resources = process_file_only(file_input, enable_mistral_ocr, enable_fhir)
            # Return 7 values expected by Gradio outputs
            return (
                status, entities, fhir_resources,           # Core results (3)
                get_dashboard_status(),                     # Dashboard status (1)
                get_dashboard_metrics(),                    # Dashboard metrics (1)
                get_jobs_history(),                         # Jobs history (1)
                gr.update(visible=True)                     # Cancel button state (1)
            )

        def process_dicom_with_state(dicom_input):
            # Ensure dashboard functions are available
            _ensure_app_imports()
            # Get core processing results (3 values)
            status, analysis, fhir_imaging = process_dicom_only(dicom_input)
            # Return 8 values expected by Gradio outputs
            return (
                status, analysis, fhir_imaging,             # Core results (3)
                get_dashboard_status(),                     # Dashboard status (1)
                
                get_dashboard_metrics(),                    # Dashboard metrics (1)
                get_jobs_history(),                         # Jobs history (1)
                gr.update(visible=True)                     # Cancel button state (1)
            )

        text_components["process_text_btn"].click(
            fn=process_text_with_state,
            inputs=[text_components["text_input"], text_components["enable_fhir_text"]],
            outputs=[text_components["text_status"], text_components["extracted_entities"],
                    text_components["fhir_resources"], processing_status,
                    metrics_display, files_history_component, text_components["cancel_text_btn"]]
        )
        
        file_components["process_file_btn"].click(
            fn=process_file_with_state,
            inputs=[file_components["file_input"], file_components["enable_mistral_ocr"], file_components["enable_fhir_file"]],
            outputs=[file_components["file_status"], file_components["file_entities"],
                    file_components["file_fhir"], processing_status,
                    metrics_display, files_history_component, file_components["cancel_file_btn"]]
        )
        
        dicom_components["process_dicom_btn"].click(
            fn=process_dicom_with_state,
            inputs=[dicom_components["dicom_input"]],
            outputs=[dicom_components["dicom_status"], dicom_components["dicom_analysis"],
                    dicom_components["dicom_fhir"], processing_status,
                    metrics_display, files_history_component, dicom_components["cancel_dicom_btn"]]
        )

        # Cancel button event handlers - properly interrupt processing and reset state
        def cancel_text_task():
            # Force stop current processing and reset state
            status = cancel_current_task("text_task")
            # Return ready state and clear results
            ready_status = "🔄 Processing cancelled. Ready for next text analysis."
            return ready_status, {}, {}, get_dashboard_status(), get_dashboard_metrics(), get_jobs_history(), gr.update(visible=False)

        def cancel_file_task():
            # Force stop current processing and reset state
            status = cancel_current_task("file_task")
            # Return ready state and clear results
            ready_status = "🔄 Processing cancelled. Ready for next document upload."
            return ready_status, {}, {}, get_dashboard_status(), get_dashboard_metrics(), get_jobs_history(), gr.update(visible=False)

        def cancel_dicom_task():
            # Force stop current processing and reset state
            status = cancel_current_task("dicom_task")
            # Return ready state and clear results
            ready_status = "🔄 Processing cancelled. Ready for next DICOM analysis."
            return ready_status, {}, {}, get_dashboard_status(), get_dashboard_metrics(), get_jobs_history(), gr.update(visible=False)
        
        text_components["cancel_text_btn"].click(
            fn=cancel_text_task,
            outputs=[text_components["text_status"], text_components["extracted_entities"],
                    text_components["fhir_resources"], processing_status,
                    metrics_display, files_history_component, text_components["cancel_text_btn"]]
        )
        
        file_components["cancel_file_btn"].click(
            fn=cancel_file_task,
            outputs=[file_components["file_status"], file_components["file_entities"],
                    file_components["file_fhir"], processing_status,
                    metrics_display, files_history_component, file_components["cancel_file_btn"]]
        )
        
        dicom_components["cancel_dicom_btn"].click(
            fn=cancel_dicom_task,
            outputs=[dicom_components["dicom_status"], dicom_components["dicom_analysis"],
                    dicom_components["dicom_fhir"], processing_status,
                    metrics_display, files_history_component, dicom_components["cancel_dicom_btn"]]
        )
        
        # Add refresh status button click handler
        def refresh_agent_status():
            """Refresh the agent status display"""
            import time
            status_html = get_simple_agent_status()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            last_updated_html = f"<p><small>Last updated: {timestamp}</small></p>"
            return status_html, last_updated_html
        
        stats_components["refresh_status_btn"].click(
            fn=refresh_agent_status,
            outputs=[stats_components["agent_status_display"], stats_components["last_updated_display"]]
        )
    
    return demo

# Helper functions for demos
def start_heavy_workload():
    """Start the heavy workload demo with real Modal container scaling"""
    import asyncio
    
    try:
        # Start the Modal container scaling demo
        result = asyncio.run(heavy_workload_demo.start_modal_scaling_demo())
        
        # Get initial container data
        containers = heavy_workload_demo.get_container_details()
        
        # Get scaling metrics
        stats = heavy_workload_demo.get_demo_statistics()
        metrics_data = [
            ["Demo Status", stats['demo_status']],
            ["Active Containers", stats['active_containers']],
            ["Requests/sec", stats['requests_per_second']],
            ["Total Processed", stats['total_requests_processed']],
            ["Scaling Strategy", stats['scaling_strategy']],
            ["Cost per Request", stats['cost_per_request']],
            ["Runtime", stats['total_runtime']]
        ]
        
        # Create basic workload chart data (placeholder for now)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1, 2], y=[1, 5, 15], mode='lines+markers', name='Containers'))
        fig.update_layout(title="Container Scaling Over Time", xaxis_title="Time (min)", yaxis_title="Container Count")
        
        return containers, metrics_data, fig
        
    except Exception as e:
        error_data = [["Error", f"Failed to start demo: {str(e)}"]]
        return [], error_data, None

def stop_heavy_workload():
    """Stop the heavy workload demo"""
    try:
        # Stop the Modal container scaling demo
        heavy_workload_demo.stop_demo()
        
        # Get final container data (should be empty or scaled down)
        containers = heavy_workload_demo.get_container_details()
        
        # Get final metrics
        stats = heavy_workload_demo.get_demo_statistics()
        metrics_data = [
            ["Demo Status", "Demo Stopped"],
            ["Active Containers", 0],
            ["Requests/sec", 0],
            ["Total Processed", stats['total_requests_processed']],
            ["Final Runtime", stats['total_runtime']],
            ["Cost per Request", stats['cost_per_request']]
        ]
        
        # Empty chart when stopped
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name='Stopped'))
        fig.update_layout(title="Demo Stopped", xaxis_title="Time", yaxis_title="Containers")
        
        return containers, metrics_data, fig
        
    except Exception as e:
        error_data = [["Error", f"Failed to stop demo: {str(e)}"]]
        return [], error_data, None

def refresh_demo_data():
    """Refresh demo data with current container status"""
    try:
        # Get current container data
        containers = heavy_workload_demo.get_container_details()
        
        # Get current scaling metrics
        stats = heavy_workload_demo.get_demo_statistics()
        metrics_data = [
            ["Demo Status", stats['demo_status']],
            ["Active Containers", stats['active_containers']],
            ["Requests/sec", stats['requests_per_second']],
            ["Total Processed", stats['total_requests_processed']],
            ["Concurrent Requests", stats['concurrent_requests']],
            ["Scaling Strategy", stats['scaling_strategy']],
            ["Cost per Request", stats['cost_per_request']],
            ["Runtime", stats['total_runtime']]
        ]
        
        # Update workload chart with current data
        import plotly.graph_objects as go
        import time
        
        # Simulate time series data for demo
        current_time = time.time()
        times = [(current_time - 60 + i*10) for i in range(7)]  # Last 60 seconds
        container_counts = [1, 2, 5, 8, 12, 15, stats['active_containers']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=container_counts,
            mode='lines+markers',
            name='Container Count',
            line=dict(color='#B71C1C', width=3)
        ))
        fig.update_layout(
            title="Modal Container Auto-Scaling",
            xaxis_title="Time",
            yaxis_title="Active Containers",
            showlegend=True
        )
        
        return containers, metrics_data, fig
        
    except Exception as e:
        error_data = [["Error", f"Failed to refresh: {str(e)}"]]
        return [], error_data, None

def start_live_processing(batch_size, processing_type, enable_live_updates):
    """Start live batch processing with real progress tracking"""
    try:
        # Update main dashboard too
        
        # Map sample file categories to workflow types (no OCR used)
        workflow_map = {
            "Clinical Notes Sample": "clinical_fhir",
            "Lab Reports Sample": "lab_entities",
            "Discharge Summaries Sample": "clinical_fhir"
        }
        
        workflow_type = workflow_map.get(processing_type, "clinical_fhir")
        
        # Start batch processing with real data (no OCR used)
        success = batch_processor.start_processing(
            workflow_type=workflow_type,
            batch_size=batch_size,
            progress_callback=None  # We'll check status periodically
        )
        
        if success:
            # Update main dashboard to show batch processing activity
            dashboard_state["active_tasks"] += 1
            dashboard_state["last_update"] = f"Batch processing started: {batch_size} sample documents"
            
            status = f"🔄 **Processing Started**\nBatch Size: {batch_size}\nSample Category: {processing_type}\nWorkflow: {workflow_type}"
            log = f"Started processing {batch_size} {processing_type.lower()} using {workflow_type} workflow (no OCR)\n"
            results = {
                "total_documents": batch_size,
                "processed": 0,
                "entities_extracted": 0,
                "fhir_resources_generated": 0,
                "processing_time": "0s",
                "avg_time_per_doc": "0s"
            }
            return status, log, results
        else:
            return "❌ Failed to start processing - already running", "", {}
            
    except Exception as e:
        return f"❌ Error starting processing: {str(e)}", "", {}

def stop_processing():
    """Stop batch processing"""
    try:
        
        batch_processor.stop_processing()
        
        # Get final status
        final_status = batch_processor.get_status()
        
        # Update main dashboard when stopping
        if dashboard_state["active_tasks"] > 0:
            dashboard_state["active_tasks"] -= 1
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if final_status["status"] == "completed":
            log = f"Processing completed: {final_status['processed']} documents in {final_status['total_time']:.2f}s\n"
            dashboard_state["last_update"] = f"Batch completed: {final_status['processed']} documents at {current_time}"
        else:
            log = "Processing stopped by user\n"
            dashboard_state["last_update"] = f"Batch stopped by user at {current_time}"
            
        return "⏹️ Processing stopped", log
        
    except Exception as e:
        return f"❌ Error stopping processing: {str(e)}", ""

# Global state tracking to prevent UI blinking/flashing
_last_dashboard_state = {}
_last_batch_status = {}
_batch_completion_processed = False  # Track if we've already processed completion

def update_batch_status_realtime():
    """Real-time status updates for batch processing - called by timer"""
    try:
        
        status = batch_processor.get_status()
        
        # Track current state to prevent unnecessary updates and blinking
        global _last_dashboard_state, _last_batch_status, _batch_completion_processed
        
        # If batch is completed and we've already processed it, stop all updates
        if status["status"] == "completed" and _batch_completion_processed:
            return (
                gr.update(),  # batch_status - no update
                gr.update(),  # processing_log - no update
                gr.update(),  # results_summary - no update
                gr.update(),  # processing_status - no update
                gr.update(),  # metrics_display - no update
                gr.update()   # files_history - no update
            )
        current_dashboard_state = {
            'total_files': dashboard_state.get('total_files', 0),
            'successful_files': dashboard_state.get('successful_files', 0),
            'failed_files': dashboard_state.get('failed_files', 0),
            'active_tasks': dashboard_state.get('active_tasks', 0),
            'last_update': dashboard_state.get('last_update', 'Never')
        }
        
        current_batch_state = {
            'status': status.get('status', 'ready'),
            'processed': status.get('processed', 0),
            'total': status.get('total', 0),
            'elapsed_time': status.get('elapsed_time', 0)
        }
        
        # Check if dashboard state has changed
        dashboard_changed = current_dashboard_state != _last_dashboard_state
        batch_changed = current_batch_state != _last_batch_status
        
        # Update tracking state
        _last_dashboard_state = current_dashboard_state.copy()
        _last_batch_status = current_batch_state.copy()
        
        # Mark completion as processed to prevent repeated updates
        if status["status"] == "completed":
            _last_batch_status['completion_processed'] = True
        
        if status["status"] == "ready":
            # Reset completion flag for new batch
            _batch_completion_processed = False
            return (
                "🔄 Ready to start batch processing",
                "",
                create_empty_results_summary(),
                get_dashboard_status() if get_dashboard_status else "<p>Dashboard not available</p>",
                
                get_dashboard_metrics() if get_dashboard_metrics else [],
                get_jobs_history() if get_jobs_history else []
            )
            
        elif status["status"] == "processing":
            # Update main dashboard with current progress
            processed_docs = status['processed']
            total_docs = status['total']
            
            # Add newly completed documents to dashboard in real-time
            results = status.get('results', [])
            if results and _add_file_to_dashboard:
                # Check if there are new completed documents since last update
                completed_count = len([r for r in results if r.get('status') == 'completed'])
                dashboard_processed = dashboard_state.get('batch_processed_count', 0)
                
                # Add new completed documents to dashboard
                if completed_count > dashboard_processed:
                    for i in range(dashboard_processed, completed_count):
                        if i < len(results):
                            result = results[i]
                            sample_category = status.get('current_workflow', 'Sample Document')
                            processing_time = result.get('processing_time', 0)
                            _add_file_to_dashboard(
                                filename=f"Batch Document {i+1}",
                                file_type=f"{sample_category} (Batch)",
                                success=True,
                                processing_time=f"{processing_time:.2f}s",
                                error=None
                            )
                    dashboard_state['batch_processed_count'] = completed_count
            
            # Update dashboard state to show batch processing activity
            dashboard_state["last_update"] = f"Batch processing: {processed_docs}/{total_docs} documents"
            
            # Calculate progress
            progress_percent = (processed_docs / total_docs) * 100
            
            # Create progress bar HTML
            progress_html = f"""
            <div style="margin: 10px 0;">
                <div style="background: #f0f0f0; border-radius: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #4CAF50, #2196F3);
                                height: 20px; width: {progress_percent}%;
                                display: flex; align-items: center; justify-content: center;
                                color: white; font-weight: bold;">
                        {progress_percent:.1f}%
                    </div>
                </div>
            </div>
            """
            
            # Enhanced status text
            current_step_desc = status.get('current_step_description', 'Processing...')
            status_text = f"""
            🔄 **Processing in Progress**
            {progress_html}
            **Document:** {processed_docs}/{total_docs}
            **Current Step:** {current_step_desc}
            **Elapsed:** {status['elapsed_time']:.1f}s
            **Estimated Remaining:** {status['estimated_remaining']:.1f}s
            """
            
            # Build clean processing log - remove duplicates and show only key milestones
            log_entries = []
            processing_log = status.get('processing_log', [])
            
            # Group log entries by document and show only completion status
            doc_status = {}
            for log_entry in processing_log:
                doc_num = log_entry.get('document', 0)
                step = log_entry.get('step', '')
                message = log_entry.get('message', '')
                
                # Only keep completion messages and avoid duplicates
                if 'completed' in step or 'Document' in message and 'completed' in message:
                    doc_status[doc_num] = f"📄 Doc {doc_num}: {message}"
                elif doc_num not in doc_status and ('processing' in step or 'Processing' in message):
                    doc_status[doc_num] = f"📄 Doc {doc_num}: Processing..."
            
            # Show last 6 documents progress
            recent_docs = sorted(doc_status.keys())[-6:]
            for doc_num in recent_docs:
                log_entries.append(doc_status[doc_num])
            
            log_text = "\n".join(log_entries) if log_entries else "Starting batch processing..."
            
            # Calculate metrics from results
            results = status.get('results', [])
            total_entities = sum(len(result.get('entities', [])) for result in results)
            total_fhir = sum(1 for result in results if result.get('fhir_bundle_generated', False))
            
            results_summary = {
                "total_documents": status['total'],
                "processed": status['processed'],
                "entities_extracted": total_entities,
                "fhir_resources_generated": total_fhir,
                "processing_time": f"{status['elapsed_time']:.1f}s",
                "avg_time_per_doc": f"{status['elapsed_time']/status['processed']:.1f}s" if status['processed'] > 0 else "0s",
                "documents_per_second": f"{status['processed']/status['elapsed_time']:.2f}" if status['elapsed_time'] > 0 else "0"
            }
            
            # Return with dashboard updates
            return (status_text, log_text, results_summary,
                   get_dashboard_status() if get_dashboard_status else "<p>Dashboard not available</p>",
                   
                   get_dashboard_metrics() if get_dashboard_metrics else [],
                   get_jobs_history() if get_jobs_history else [])
            
        elif status["status"] == "completed":
            # Mark completion as processed to stop future updates
            _batch_completion_processed = True
            
            # Processing completed - add all processed documents to main dashboard
            results = status.get('results', [])
            total_entities = sum(len(result.get('entities', [])) for result in results)
            total_fhir = sum(1 for result in results if result.get('fhir_bundle_generated', False))
            
            # Add each processed document to the main dashboard
            import datetime
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Ensure we have the add_file_to_dashboard function
            try:
                from app import add_file_to_dashboard
                for i, result in enumerate(results):
                    doc_id = result.get('document_id', f'batch_doc_{i+1}')
                    entities_count = len(result.get('entities', []))
                    processing_time = result.get('processing_time', 0)
                    fhir_generated = result.get('fhir_bundle_generated', False)
                    
                    # Add to dashboard as individual file - this will update all counters automatically
                    sample_category = status.get('processing_type', 'Batch Demo Document')
                    add_file_to_dashboard(
                        filename=f"Batch Document {i+1}",
                        file_type=f"{sample_category}",
                        success=True,
                        processing_time=f"{processing_time:.2f}s",
                        error=None,
                        entities_found=entities_count
                    )
            except Exception as e:
                print(f"Error adding batch files to dashboard: {e}")
            
            # Update final dashboard state
            if dashboard_state["active_tasks"] > 0:
                dashboard_state["active_tasks"] -= 1
            dashboard_state["last_update"] = f"Batch completed: {status['processed']} documents at {current_time}"
            
            completion_text = f"""
            ✅ **Processing Completed Successfully!**
            
            📊 **Final Results:**
            - **Documents Processed:** {status['processed']}/{status['total']}
            - **Total Processing Time:** {status['total_time']:.2f}s
            - **Average Time per Document:** {status['total_time']/status['processed']:.2f}s
            - **Documents per Second:** {status['processed']/status['total_time']:.2f}
            - **Total Entities Extracted:** {total_entities}
            - **FHIR Resources Generated:** {total_fhir}
            
            🎉 **All documents added to File Processing Dashboard!**
            """
            
            final_results = {
                "total_documents": status['total'],
                "processed": status['processed'],
                "entities_extracted": total_entities,
                "fhir_resources_generated": total_fhir,
                "processing_time": f"{status['total_time']:.1f}s",
                "avg_time_per_doc": f"{status['total_time']/status['processed']:.1f}s",
                "documents_per_second": f"{status['processed']/status['total_time']:.2f}"
            }
            
            # Return with dashboard updates
            return (completion_text, "🎉 All documents processed successfully!", final_results,
                   get_dashboard_status() if get_dashboard_status else "<p>Dashboard not available</p>",
                   
                   get_dashboard_metrics() if get_dashboard_metrics else [],
                   get_jobs_history() if get_jobs_history else [])
            
        else:  # cancelled or error
            return (f"⚠️ Processing {status['status']}", status.get('message', ''), create_empty_results_summary(),
                   get_dashboard_status() if get_dashboard_status else "<p>Dashboard not available</p>",
                   
                   get_dashboard_metrics() if get_dashboard_metrics else [],
                   get_jobs_history() if get_jobs_history else [])
            
    except Exception as e:
        return (f"❌ Status update error: {str(e)}", "", create_empty_results_summary(),
               get_dashboard_status() if get_dashboard_status else "<p>Dashboard not available</p>",
               
               get_dashboard_metrics() if get_dashboard_metrics else [],
               get_jobs_history() if get_jobs_history else [])

def create_empty_results_summary():
    """Create empty results summary"""
    return {
        "total_documents": 0,
        "processed": 0,
        "entities_extracted": 0,
        "fhir_resources_generated": 0,
        "processing_time": "0s",
        "avg_time_per_doc": "0s"
    }

def get_batch_processing_status():
    """Get current batch processing status with detailed step-by-step feedback"""
    try:
        status = batch_processor.get_status()
        
        if status["status"] == "ready":
            return "🔄 Ready to start batch processing", "", {
                "total_documents": 0,
                "processed": 0,
                "entities_extracted": 0,
                "fhir_resources_generated": 0,
                "processing_time": "0s",
                "avg_time_per_doc": "0s"
            }
            
        elif status["status"] == "processing":
            # Enhanced progress text with current step information
            current_step_desc = status.get('current_step_description', 'Processing...')
            progress_text = f"🔄 **Processing in Progress**\nProgress: {status['progress']:.1f}%\nDocument: {status['processed']}/{status['total']}\nCurrent Step: {current_step_desc}\nElapsed: {status['elapsed_time']:.1f}s\nEstimated remaining: {status['estimated_remaining']:.1f}s"
            
            # Build clean log with recent processing steps - avoid duplicates
            log_entries = []
            processing_log = status.get('processing_log', [])
            
            # Group by document to avoid duplicates
            doc_status = {}
            for log_entry in processing_log:
                doc_num = log_entry.get('document', 0)
                step = log_entry.get('step', '')
                message = log_entry.get('message', '')
                
                # Only keep meaningful completion messages
                if 'completed' in step or ('completed' in message and 'entities' in message):
                    doc_status[doc_num] = f"Doc {doc_num}: Completed"
                elif doc_num not in doc_status:
                    doc_status[doc_num] = f"Doc {doc_num}: Processing..."
            
            # Show last 5 documents
            recent_docs = sorted(doc_status.keys())[-5:]
            for doc_num in recent_docs:
                log_entries.append(doc_status[doc_num])
            
            log_text = "\n".join(log_entries) + "\n"
            
            # Calculate entities and FHIR from results so far
            results = status.get('results', [])
            total_entities = sum(len(result.get('entities', [])) for result in results)
            total_fhir = sum(1 for result in results if result.get('fhir_bundle_generated', False))
            
            results_summary = {
                "total_documents": status['total'],
                "processed": status['processed'],
                "entities_extracted": total_entities,
                "fhir_resources_generated": total_fhir,
                "processing_time": f"{status['elapsed_time']:.1f}s",
                "avg_time_per_doc": f"{status['elapsed_time']/status['processed']:.1f}s" if status['processed'] > 0 else "0s"
            }
            
            return progress_text, log_text, results_summary
            
        elif status["status"] == "cancelled":
            cancelled_text = f"⏹️ **Processing Cancelled**\nProcessed: {status['processed']}/{status['total']} ({status['progress']:.1f}%)\nElapsed time: {status['elapsed_time']:.1f}s"
            
            # Calculate partial results
            results = status.get('results', [])
            total_entities = sum(len(result.get('entities', [])) for result in results)
            total_fhir = sum(1 for result in results if result.get('fhir_bundle_generated', False))
            
            partial_results = {
                "total_documents": status['total'],
                "processed": status['processed'],
                "entities_extracted": total_entities,
                "fhir_resources_generated": total_fhir,
                "processing_time": f"{status['elapsed_time']:.1f}s",
                "avg_time_per_doc": f"{status['elapsed_time']/status['processed']:.1f}s" if status['processed'] > 0 else "0s"
            }
            
            log_cancelled = f"Processing cancelled by user after {status['elapsed_time']:.1f}s\nPartial results: {status['processed']} documents processed\nExtracted {total_entities} medical entities\nGenerated {total_fhir} FHIR resources\n"
            
            return cancelled_text, log_cancelled, partial_results
            
        elif status["status"] == "completed":
            completed_text = f"✅ **Processing Complete!**\nTotal processed: {status['processed']}/{status['total']}\nTotal time: {status['total_time']:.2f}s"
            
            # Calculate final metrics
            results = status.get('results', [])
            total_entities = sum(len(result.get('entities', [])) for result in results)
            total_fhir = sum(1 for result in results if result.get('fhir_bundle_generated', False))
            
            final_results = {
                "total_documents": status['total'],
                "processed": status['processed'],
                "entities_extracted": total_entities,
                "fhir_resources_generated": total_fhir,
                "processing_time": f"{status['total_time']:.2f}s",
                "avg_time_per_doc": f"{status['total_time']/status['processed']:.2f}s" if status['processed'] > 0 else "0s"
            }
            
            log_final = f"✅ Batch processing completed successfully!\nProcessed {status['processed']} documents in {status['total_time']:.2f}s\nExtracted {total_entities} medical entities\nGenerated {total_fhir} FHIR resources\nAverage processing time: {status['total_time']/status['processed']:.2f}s per document\n"
            
            return completed_text, log_final, final_results
            
    except Exception as e:
        return f"❌ Error getting status: {str(e)}", "", {}
