#!/usr/bin/env python3
"""
Modal Functions for FhirFlame - L4 GPU Only + MCP Integration
Aligned with Modal documentation and integrated with FhirFlame MCP Server
"""
import modal
import json
import time
import os
import sys
from typing import Dict, Any, Optional

# Add src to path for monitoring
sys.path.append('/app/src')
try:
    from monitoring import monitor
except ImportError:
    # Fallback for Modal environment
    class DummyMonitor:
        def log_modal_function_call(self, *args, **kwargs): pass
        def log_modal_scaling_event(self, *args, **kwargs): pass
        def log_error_event(self, *args, **kwargs): pass
        def log_medical_entity_extraction(self, *args, **kwargs): pass
        def log_medical_processing(self, *args, **kwargs): pass
    monitor = DummyMonitor()

def calculate_real_modal_cost(processing_time: float, gpu_type: str = "L4") -> float:
    """Calculate real Modal cost for L4 GPU processing"""
    # L4 GPU pricing from environment
    l4_hourly_rate = float(os.getenv("MODAL_L4_HOURLY_RATE", "0.73"))
    platform_fee = float(os.getenv("MODAL_PLATFORM_FEE", "15")) / 100
    
    hours_used = processing_time / 3600
    total_cost = l4_hourly_rate * hours_used * (1 + platform_fee)
    
    return round(total_cost, 6)

# Create Modal App following official documentation
app = modal.App("fhirflame-medical-ai-fresh")

# Define optimized image for medical AI processing with optional cache busting
cache_bust_commands = []
if os.getenv("MODAL_NO_CACHE", "false").lower() == "true":
    # Add cache busting command with timestamp
    import time
    cache_bust_commands.append(f"echo 'Cache bust: {int(time.time())}'")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands([
        "pip install --upgrade pip",
        "echo 'Fresh build with fixed Langfuse tracking'",
    ] + cache_bust_commands)
    .pip_install([
        "transformers==4.35.0",
        "torch==2.1.0",
        "fhir-resources==7.1.0",  # Compatible with pydantic 2.x
        "pydantic>=2.7.2",
        "httpx>=0.25.0",
        "regex>=2023.10.3"
    ])
    .run_commands([
        "pip cache purge || echo 'Cache purge not available, continuing...'"
    ])
)

# L4 GPU Function - Main processor for MCP Server integration
@app.function(
    image=image,
    gpu="L4",  # RTX 4090 equivalent - only GPU we use
    timeout=300,
    scaledown_window=60,  # Updated parameter name for Modal 1.0
    min_containers=0,
    max_containers=15,
    memory=8192,
    cpu=4.0,
    secrets=[modal.Secret.from_name("fhirflame-env")]
)
def process_medical_document(
    document_content: str,
    document_type: str = "clinical_note",
    processing_mode: str = "comprehensive",
    include_validation: bool = True
) -> Dict[str, Any]:
    """
    Process medical documents using L4 GPU
    Returns structured medical data with cost tracking
    """
    start_time = time.time()
    
    try:
        monitor.log_modal_function_call(
            function_name="process_medical_document",
            gpu_type="L4",
            document_type=document_type,
            processing_mode=processing_mode
        )
        
        # Initialize transformers pipeline
        from transformers import pipeline
        import torch
        
        # Check GPU availability
        device = 0 if torch.cuda.is_available() else -1
        monitor.log_modal_scaling_event("GPU_DETECTED", {"cuda_available": torch.cuda.is_available()})
        
        # Medical NER pipeline
        ner_pipeline = pipeline(
            "ner",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
            device=device
        )
        
        # Extract medical entities
        entities = ner_pipeline(document_content)
        
        # Process entities into structured format
        processed_entities = {}
        for entity in entities:
            entity_type = entity['entity_group']
            if entity_type not in processed_entities:
                processed_entities[entity_type] = []
            
            processed_entities[entity_type].append({
                'text': entity['word'],
                'confidence': float(entity['score']),
                'start': int(entity['start']),
                'end': int(entity['end'])
            })
        
        # Calculate processing metrics
        processing_time = time.time() - start_time
        cost = calculate_real_modal_cost(processing_time, "L4")
        
        monitor.log_medical_entity_extraction(
            entities_found=len(entities),
            processing_time=processing_time,
            cost=cost
        )
        
        # Basic medical document structure (without FHIR for now)
        result = {
            "document_type": document_type,
            "processing_mode": processing_mode,
            "entities": processed_entities,
            "processing_metadata": {
                "processing_time_seconds": processing_time,
                "estimated_cost_usd": cost,
                "gpu_type": "L4",
                "entities_extracted": len(entities),
                "timestamp": time.time()
            },
            "medical_insights": {
                "entity_types_found": list(processed_entities.keys()),
                "total_entities": len(entities),
                "confidence_avg": sum(e['score'] for e in entities) / len(entities) if entities else 0
            }
        }
        
        monitor.log_medical_processing(
            success=True,
            processing_time=processing_time,
            cost=cost,
            entities_count=len(entities)
        )
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        cost = calculate_real_modal_cost(processing_time, "L4")
        
        monitor.log_error_event(
            error_type=type(e).__name__,
            error_message=str(e),
            processing_time=processing_time,
            cost=cost
        )
        
        return {
            "error": True,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "processing_metadata": {
                "processing_time_seconds": processing_time,
                "estimated_cost_usd": cost,
                "gpu_type": "L4",
                "timestamp": time.time()
            }
        }

# MCP Integration Endpoint
@app.function(
    image=image,
    gpu="L4",
    timeout=300,
    scaledown_window=60,
    min_containers=0,
    max_containers=10,
    memory=8192,
    cpu=4.0,
    secrets=[modal.Secret.from_name("fhirflame-env")]
)
def mcp_medical_processing_endpoint(
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    MCP-compatible endpoint for medical document processing
    Used by FhirFlame MCP Server
    """
    start_time = time.time()
    
    try:
        # Extract request parameters
        document_content = request_data.get("document_content", "")
        document_type = request_data.get("document_type", "clinical_note")
        processing_mode = request_data.get("processing_mode", "comprehensive")
        
        if not document_content:
            return {
                "success": False,
                "error": "No document content provided",
                "mcp_response": {
                    "status": "error",
                    "message": "Document content is required"
                }
            }
        
        # Process document
        result = process_medical_document.local(
            document_content=document_content,
            document_type=document_type,
            processing_mode=processing_mode
        )
        
        # Format for MCP response
        mcp_response = {
            "success": not result.get("error", False),
            "data": result,
            "mcp_metadata": {
                "endpoint": "mcp-medical-processing",
                "version": "1.0",
                "timestamp": time.time()
            }
        }
        
        return mcp_response
        
    except Exception as e:
        processing_time = time.time() - start_time
        cost = calculate_real_modal_cost(processing_time, "L4")
        
        return {
            "success": False,
            "error": str(e),
            "mcp_response": {
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "cost": cost,
                "processing_time": processing_time
            }
        }

# Health check endpoint
@app.function(
    image=image,
    timeout=30,
    scaledown_window=30,
    min_containers=1,  # Keep one warm for health checks
    max_containers=3,
    memory=1024,
    cpu=1.0
)
def health_check() -> Dict[str, Any]:
    """Health check endpoint for Modal functions"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "app": "fhirflame-medical-ai-fresh",
        "functions": ["process_medical_document", "mcp_medical_processing_endpoint"],
        "gpu_support": "L4"
    }

if __name__ == "__main__":
    print("FhirFlame Modal Functions - L4 GPU Medical Processing")
    print("Available functions:")
    print("- process_medical_document: Main medical document processor")
    print("- mcp_medical_processing_endpoint: MCP-compatible endpoint")
    print("- health_check: System health monitoring")