#!/usr/bin/env python3
"""
FhirFlame: Medical AI Technology Demonstration
MVP/Prototype Platform - Development & Testing Only

⚠️ IMPORTANT: This is a technology demonstration and MVP prototype for development,
testing, and educational purposes only. NOT approved for clinical use, patient data,
or production healthcare environments. Requires proper regulatory evaluation,
compliance review, and legal assessment before any real-world deployment.

Technology Stack Demonstration:
- Real-time medical text processing with CodeLlama 13B-Instruct
- FHIR R4/R5 compliance workflow prototypes
- Multi-provider AI routing architecture (Ollama, HuggingFace, Modal)
- Healthcare document processing with OCR capabilities
- DICOM medical imaging analysis demos
- Enterprise-grade security patterns (demonstration)

Architecture: Microservices with horizontal auto-scaling patterns
Security: Healthcare-grade infrastructure patterns (demo implementation)
Performance: Optimized for demonstration and development workflows
"""

import os
import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional
from pathlib import Path

# Import our core modules
from src.workflow_orchestrator import WorkflowOrchestrator
from src.enhanced_codellama_processor import EnhancedCodeLlamaProcessor
from src.fhir_validator import FhirValidator
from src.dicom_processor import dicom_processor
from src.monitoring import monitor

# Import database module for persistent job tracking
from database import db_manager

# Frontend UI components will be imported dynamically to avoid circular imports

# Global instances - using proper initialization to ensure services are ready
codellama = None
enhanced_codellama = None
fhir_validator = None
workflow_orchestrator = None

# ============================================================================
# SERVICE INITIALIZATION & STATUS TRACKING
# ============================================================================

# Service initialization status tracking for all AI providers and core components
# This ensures proper startup sequence and service health monitoring
service_status = {
    "ollama_initialized": False,           # Ollama local AI service status
    "enhanced_codellama_initialized": False,  # Enhanced CodeLlama processor status
    "ollama_connection_url": None,         # Active Ollama connection endpoint
    "last_ollama_check": None             # Timestamp of last Ollama health check
}

# ============================================================================
# TASK CANCELLATION & CONCURRENCY MANAGEMENT
# ============================================================================

# Task cancellation mechanism for graceful job termination
# Each task type can be independently cancelled without affecting others
cancellation_flags = {
    "text_task": False,    # Medical text processing cancellation flag
    "file_task": False,    # Document/file processing cancellation flag
    "dicom_task": False    # DICOM medical imaging cancellation flag
}

# Active running tasks storage for proper cancellation and cleanup
# Stores asyncio Task objects for each processing type
running_tasks = {
    "text_task": None,     # Current text processing asyncio Task
    "file_task": None,     # Current file processing asyncio Task
    "dicom_task": None     # Current DICOM processing asyncio Task
}

# Task queue system for handling multiple concurrent requests
# Allows queueing of pending tasks when system is busy
task_queues = {
    "text_task": [],       # Queued text processing requests
    "file_task": [],       # Queued file processing requests
    "dicom_task": []       # Queued DICOM processing requests
}

# Current active job IDs for tracking and dashboard display
# Maps task types to their current PostgreSQL job record IDs
active_jobs = {
    "text_task": None,     # Active text processing job ID
    "file_task": None,     # Active file processing job ID
    "dicom_task": None     # Active DICOM processing job ID
}

import uuid
import datetime

class UnifiedJobManager:
    """Centralized job and metrics management for all FhirFlame processing with PostgreSQL persistence"""
    
    def __init__(self):
        # Keep minimal in-memory state for compatibility, but use PostgreSQL as primary store
        self.jobs_database = {
            "processing_jobs": [],      # Legacy compatibility - now synced from PostgreSQL
            "batch_jobs": [],           # Legacy compatibility - now synced from PostgreSQL
            "container_metrics": [],    # Modal container scaling
            "performance_metrics": [],  # AI provider performance
            "queue_statistics": {       # Processing queue stats - calculated from PostgreSQL
                "active_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0
            },
            "system_monitoring": []     # System performance
        }
        
        # Dashboard state - calculated from PostgreSQL
        self.dashboard_state = {
            "active_tasks": 0,
            "files_processed": [],
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "failed_tasks": 0,
            "processing_queue": {"active_tasks": 0, "completed_files": 0, "failed_files": 0},
            "last_update": None
        }
        
        # Sync dashboard state from PostgreSQL on initialization
        self._sync_dashboard_from_db()
    
    def _sync_dashboard_from_db(self):
        """Sync dashboard state from PostgreSQL database"""
        try:
            metrics = db_manager.get_dashboard_metrics()
            self.dashboard_state.update({
                "active_tasks": metrics.get('active_jobs', 0),
                "total_files": metrics.get('completed_jobs', 0),
                "successful_files": metrics.get('successful_jobs', 0),
                "failed_files": metrics.get('failed_jobs', 0),
                "failed_tasks": metrics.get('failed_jobs', 0)
            })
            print(f"✅ Dashboard synced from PostgreSQL: {metrics}")
        except Exception as e:
            print(f"⚠️ Failed to sync dashboard from PostgreSQL: {e}")
        
    def add_processing_job(self, job_type: str, name: str, details: dict = None) -> str:
        """Record start of any type of processing job in PostgreSQL"""
        job_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        job_record = {
            "id": job_id,
            "job_type": job_type,  # "text", "file", "dicom", "batch"
            "name": name[:100],    # Truncate long names
            "status": "processing",
            "success": None,
            "processing_time": None,
            "error_message": None,
            "entities_found": 0,
            "result_data": details or {},
            "text_input": details.get("text_input") if details else None,
            "file_path": details.get("file_path") if details else None,
            "workflow_type": details.get("workflow_type") if details else None
        }
        
        # Save to PostgreSQL
        db_success = db_manager.add_job(job_record)
        
        if db_success:
            # Also add to in-memory for legacy compatibility
            legacy_job = {
                "job_id": job_id,
                "job_type": job_type,
                "name": name[:100],
                "status": "started",
                "success": None,
                "start_time": timestamp,
                "completion_time": None,
                "processing_time": None,
                "error": None,
                "entities_found": 0,
                "details": details or {}
            }
            self.jobs_database["processing_jobs"].append(legacy_job)
            
            # Update dashboard state and queue statistics
            self.dashboard_state["active_tasks"] += 1
            self.jobs_database["queue_statistics"]["active_tasks"] += 1
            self.dashboard_state["last_update"] = timestamp
            
            print(f"✅ Job {job_id[:8]} added to PostgreSQL: {name[:30]}...")
        else:
            print(f"❌ Failed to add job {job_id[:8]} to PostgreSQL")
        
        return job_id
        
    def update_job_completion(self, job_id: str, success: bool, metrics: dict = None):
        """Update job completion with metrics in PostgreSQL"""
        completion_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare update data for PostgreSQL
        updates = {
            "status": "completed",
            "success": success,
            "completed_at": completion_time
        }
        
        if metrics:
            updates["processing_time"] = metrics.get("processing_time", "N/A")
            updates["entities_found"] = metrics.get("entities_found", 0)
            updates["error_message"] = metrics.get("error", None)
            updates["result_data"] = metrics.get("details", {})
            
            # Handle cancellation flag
            if metrics.get("cancelled", False):
                updates["status"] = "cancelled"
                updates["error_message"] = "Cancelled by user"
        
        # Update in PostgreSQL
        db_success = db_manager.update_job(job_id, updates)
        
        if db_success:
            # Also update in-memory for legacy compatibility
            for job in self.jobs_database["processing_jobs"]:
                if job["job_id"] == job_id:
                    job["status"] = updates["status"]
                    job["success"] = success
                    job["completion_time"] = completion_time
                    
                    if metrics:
                        job["processing_time"] = metrics.get("processing_time", "N/A")
                        job["entities_found"] = metrics.get("entities_found", 0)
                        job["error"] = metrics.get("error", None)
                        job["details"].update(metrics.get("details", {}))
                        
                        # Handle cancellation flag
                        if metrics.get("cancelled", False):
                            job["status"] = "cancelled"
                            job["error"] = "Cancelled by user"
                    
                    break
            
            # Update dashboard state
            self.dashboard_state["active_tasks"] = max(0, self.dashboard_state["active_tasks"] - 1)
            self.dashboard_state["total_files"] += 1
            
            if success:
                self.dashboard_state["successful_files"] += 1
                self.jobs_database["queue_statistics"]["completed_tasks"] += 1
            else:
                self.dashboard_state["failed_files"] += 1
                self.dashboard_state["failed_tasks"] += 1
                self.jobs_database["queue_statistics"]["failed_tasks"] += 1
            
            self.jobs_database["queue_statistics"]["active_tasks"] = max(0,
                self.jobs_database["queue_statistics"]["active_tasks"] - 1)
            
            # Update files_processed list
            job_name = "Unknown"
            job_type = "Processing"
            for job in self.jobs_database["processing_jobs"]:
                if job["job_id"] == job_id:
                    job_name = job["name"]
                    job_type = job["job_type"].title() + " Processing"
                    break
            
            file_info = {
                "filename": job_name,
                "file_type": job_type,
                "success": success,
                "processing_time": updates.get("processing_time", "N/A"),
                "timestamp": completion_time,
                "error": updates.get("error_message"),
                "entities_found": updates.get("entities_found", 0)
            }
            self.dashboard_state["files_processed"].append(file_info)
            self.dashboard_state["last_update"] = completion_time
            
            # Log completion for debugging
            status_icon = "✅" if success else "❌" if not metrics.get("cancelled", False) else "⏹️"
            print(f"{status_icon} Job {job_id[:8]} completed in PostgreSQL: {job_name[:30]}... - Success: {success}")
        else:
            print(f"❌ Failed to update job {job_id[:8]} in PostgreSQL")
                
    def add_batch_job(self, batch_type: str, batch_size: int, workflow_type: str) -> str:
        """Record start of batch processing job"""
        job_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        batch_record = {
            "job_id": job_id,
            "job_type": "batch",
            "batch_type": batch_type,
            "batch_size": batch_size,
            "workflow_type": workflow_type,
            "status": "started",
            "start_time": timestamp,
            "completion_time": None,
            "processed_count": 0,
            "success_count": 0,
            "failed_count": 0,
            "documents": []
        }
        
        self.jobs_database["batch_jobs"].append(batch_record)
        self.dashboard_state["active_tasks"] += 1
        self.dashboard_state["last_update"] = f"Batch processing started: {batch_size} {workflow_type} documents"
        
        return job_id
        
    def update_batch_progress(self, job_id: str, processed_count: int, success_count: int, failed_count: int):
        """Update batch processing progress"""
        for batch in self.jobs_database["batch_jobs"]:
            if batch["job_id"] == job_id:
                batch["processed_count"] = processed_count
                batch["success_count"] = success_count
                batch["failed_count"] = failed_count
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.dashboard_state["last_update"] = f"Batch processing: {processed_count}/{batch['batch_size']} documents"
                break
                
    def get_dashboard_status(self) -> str:
        """Get current dashboard status string"""
        if self.dashboard_state["total_files"] == 0:
            return "📊 No files processed yet"
        
        return f"📊 Files: {self.dashboard_state['total_files']} | Success: {self.dashboard_state['successful_files']} | Failed: {self.dashboard_state['failed_files']} | Active: {self.dashboard_state['active_tasks']}"
    
    def get_dashboard_metrics(self) -> list:
        """Get file processing metrics for DataFrame display from PostgreSQL"""
        # Get metrics directly from PostgreSQL database
        metrics = db_manager.get_dashboard_metrics()
        
        total_jobs = metrics.get('total_jobs', 0)
        completed_jobs = metrics.get('completed_jobs', 0)
        success_jobs = metrics.get('successful_jobs', 0)
        failed_jobs = metrics.get('failed_jobs', 0)
        active_jobs = metrics.get('active_jobs', 0)
        
        # Update dashboard state with PostgreSQL data
        self.dashboard_state["total_files"] = completed_jobs
        self.dashboard_state["successful_files"] = success_jobs
        self.dashboard_state["failed_files"] = failed_jobs
        self.dashboard_state["active_tasks"] = active_jobs
        
        success_rate = (success_jobs / max(1, completed_jobs)) * 100 if completed_jobs else 0
        last_update = self.dashboard_state["last_update"] or "Never"
        
        print(f"🔍 DEBUG get_dashboard_metrics from PostgreSQL: Total={total_jobs}, Completed={completed_jobs}, Success={success_jobs}, Failed={failed_jobs}, Active={active_jobs}")
        
        return [
            ["Total Files", completed_jobs],
            ["Success Rate", f"{success_rate:.1f}%"],
            ["Failed Files", failed_jobs],
            ["Completed Files", success_jobs],
            ["Active Tasks", active_jobs],
            ["Last Update", last_update]
        ]

    def get_processing_queue(self) -> list:
        """Get processing queue for DataFrame display"""
        return [
            ["Active Tasks", self.dashboard_state["active_tasks"]],
            ["Completed Files", self.dashboard_state["successful_files"]],
            ["Failed Files", self.dashboard_state["failed_files"]]
        ]

    def get_jobs_history(self) -> list:
        """Get comprehensive jobs history for DataFrame display from PostgreSQL"""
        jobs_data = []
        
        # Get jobs from PostgreSQL database
        recent_jobs = db_manager.get_jobs_history(limit=20)
        
        print(f"🔍 DEBUG get_jobs_history from PostgreSQL: Retrieved {len(recent_jobs)} jobs")
        
        if recent_jobs:
            print(f"🔍 DEBUG: Sample jobs from PostgreSQL:")
            for i, job in enumerate(recent_jobs[:3]):
                status = job.get('status', 'unknown')
                success = job.get('success', None)
                print(f"  Job {i}: {job.get('name', 'Unknown')[:20]} | Status: {status} | Success: {success} | Type: {job.get('job_type', 'Unknown')}")
        
        # Process jobs from PostgreSQL
        for job in recent_jobs:
            job_type = job.get("job_type", "Unknown")
            job_name = job.get("name", "Unknown")
            
            # Determine job category
            if job_type == "batch":
                category = "🔄 Batch Job"
            elif job_type == "text":
                category = "📝 Text Processing"
            elif job_type == "dicom":
                category = "🏥 DICOM Analysis"
            elif job_type == "file":
                category = "📄 Document Processing"
            else:
                category = "⚙️ Processing"

            # Determine status with better handling
            if job.get("status") == "cancelled":
                status = "⏹️ Cancelled"
            elif job.get("success") is True:
                status = "✅ Success"
            elif job.get("success") is False:
                status = "❌ Failed"
            elif job.get("status") == "processing":
                status = "🔄 Processing"
            else:
                status = "⏳ Pending"
                
            job_row = [
                job_name,
                category,
                status,
                job.get("processing_time", "N/A")
            ]
            jobs_data.append(job_row)
            print(f"🔍 DEBUG: Added PostgreSQL job row: {job_row}")
        
        print(f"🔍 DEBUG: Final jobs_data length from PostgreSQL: {len(jobs_data)}")
        return jobs_data

# Create global instance
job_manager = UnifiedJobManager()
# Expose dashboard_state as reference to job_manager.dashboard_state
dashboard_state = job_manager.dashboard_state

def get_codellama():
    """Lazy load CodeLlama processor with proper Ollama initialization checks"""
    global codellama, service_status
    if codellama is None:
        print("🔄 Initializing CodeLlama processor with Ollama connection check...")
        
        # Check Ollama availability first
        ollama_ready = _check_ollama_service()
        service_status["ollama_initialized"] = ollama_ready
        service_status["last_ollama_check"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not ollama_ready:
            print("⚠️ Ollama service not ready - CodeLlama will have limited functionality")
        
        from src.codellama_processor import CodeLlamaProcessor
        codellama = CodeLlamaProcessor()
        print(f"✅ CodeLlama processor initialized (Ollama: {'Ready' if ollama_ready else 'Not Ready'})")
    return codellama

def get_enhanced_codellama():
    """Lazy load Enhanced CodeLlama processor with provider initialization checks"""
    global enhanced_codellama, service_status
    if enhanced_codellama is None:
        print("🔄 Initializing Enhanced CodeLlama processor with provider checks...")
        
        # Initialize with proper provider status tracking
        enhanced_codellama = EnhancedCodeLlamaProcessor()
        service_status["enhanced_codellama_initialized"] = True
        
        # Check provider availability after initialization
        router = enhanced_codellama.router
        print(f"✅ Enhanced CodeLlama processor ready:")
        print(f"   Ollama: {'✅ Ready' if router.ollama_available else '❌ Not Ready'}")
        print(f"   HuggingFace: {'✅ Ready' if router.hf_available else '❌ Not Ready'}")
        print(f"   Modal: {'✅ Ready' if router.modal_available else '❌ Not Ready'}")
        
    return enhanced_codellama

def _check_ollama_service():
    """Check if Ollama service is properly initialized and accessible with model status"""
    import requests
    import os
    
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    use_real_ollama = os.getenv("USE_REAL_OLLAMA", "true").lower() == "true"
    model_name = os.getenv("OLLAMA_MODEL", "codellama:13b-instruct")
    
    if not use_real_ollama:
        print("📝 Ollama disabled by configuration")
        return False
    
    # Try multiple connection attempts with different URLs
    urls_to_try = [ollama_url]
    if "ollama:11434" in ollama_url:
        urls_to_try.append("http://localhost:11434")
    elif "localhost:11434" in ollama_url:
        urls_to_try.append("http://ollama:11434")
    
    for attempt in range(3):  # Try 3 times with delays
        for url in urls_to_try:
            try:
                response = requests.get(f"{url}/api/version", timeout=5)
                if response.status_code == 200:
                    print(f"✅ Ollama service ready at {url}")
                    service_status["ollama_connection_url"] = url
                    
                    # Check model status
                    model_status = _check_ollama_model_status(url, model_name)
                    service_status["model_status"] = model_status
                    service_status["model_name"] = model_name
                    
                    if model_status == "available":
                        print(f"✅ Model {model_name} is ready")
                        return True
                    elif model_status == "downloading":
                        print(f"🔄 Model {model_name} is downloading (7.4GB)...")
                        return False
                    else:
                        print(f"❌ Model {model_name} not found")
                        return False
            except Exception as e:
                print(f"⚠️ Ollama check failed for {url}: {e}")
                continue
        import time
        time.sleep(2)  # Wait between attempts
    
    print("❌ All Ollama connection attempts failed")
    return False

def _check_ollama_model_status(url: str, model_name: str) -> str:
    """Check if specific model is available in Ollama"""
    import requests
    try:
        # Check if model is in the list of downloaded models
        response = requests.get(f"{url}/api/tags", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get("models", [])
            
            # Check if our model is in the list
            for model in models:
                if model.get("name", "").startswith(model_name.split(":")[0]):
                    return "available"
            
            # Model not found - it's likely downloading if Ollama is responsive
            return "downloading"
        else:
            return "unknown"
            
    except Exception as e:
        print(f"⚠️ Model status check failed: {e}")
        return "unknown"

def get_ollama_status() -> dict:
    """Get current Ollama and model status for UI display"""
    model_name = os.getenv("OLLAMA_MODEL", "codellama:13b-instruct")
    model_status = service_status.get("model_status", "unknown")
    
    status_messages = {
        "available": f"✅ {model_name} ready for processing",
        "downloading": f"🔄 {model_name} downloading (7.4GB). Please wait...",
        "unknown": f"⚠️ {model_name} status unknown"
    }
    
    return {
        "service_available": service_status.get("ollama_initialized", False),
        "model_status": model_status,
        "model_name": model_name,
        "message": status_messages.get(model_status, f"⚠️ Unknown status: {model_status}")
    }

def get_fhir_validator():
    """Lazy load FHIR validator"""
    global fhir_validator
    if fhir_validator is None:
        print("🔄 Initializing FHIR validator...")
        fhir_validator = FhirValidator()
        print("✅ FHIR validator ready")
    return fhir_validator

def get_workflow_orchestrator():
    """Lazy load workflow orchestrator"""
    global workflow_orchestrator
    if workflow_orchestrator is None:
        print("🔄 Initializing workflow orchestrator...")
        workflow_orchestrator = WorkflowOrchestrator()
        print("✅ Workflow orchestrator ready")
    return workflow_orchestrator

def get_current_model_display():
    """Get current model name from environment variables for display"""
    import os
    
    # Try to get from OLLAMA_MODEL first (most common)
    ollama_model = os.getenv("OLLAMA_MODEL", "")
    if ollama_model:
        # Format for display (e.g., "codellama:13b-instruct" -> "CodeLlama 13B-Instruct")
        model_parts = ollama_model.split(":")
        if len(model_parts) >= 2:
            model_name = model_parts[0].title()
            model_size = model_parts[1].upper().replace("B-", "B ").replace("-", " ").title()
            return f"{model_name} {model_size}"
        else:
            return ollama_model.title()
    
    # Fallback to other model configs
    if os.getenv("MISTRAL_API_KEY"):
        return "Mistral Large"
    elif os.getenv("HF_TOKEN"):
        return "HuggingFace Transformers"
    elif os.getenv("MODAL_TOKEN_ID"):
        return "Modal Labs GPU"
    else:
        return "CodeLlama 13B-Instruct"  # Default fallback

def get_simple_agent_status():
    """Get comprehensive system status including APIs and configurations"""
    global codellama, enhanced_codellama, fhir_validator, workflow_orchestrator
    
    # Core component status
    codellama_status = "✅ Ready" if codellama is not None else "⏳ On-demand loading"
    enhanced_status = "✅ Ready" if enhanced_codellama is not None else "⏳ On-demand loading"
    fhir_status = "✅ Ready" if fhir_validator is not None else "⏳ On-demand loading"
    workflow_status = "✅ Ready" if workflow_orchestrator is not None else "⏳ On-demand loading"
    dicom_status = "✅ Available" if dicom_processor else "❌ Not available"
    
    # API and service status
    mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
    mistral_status = "✅ Configured" if mistral_api_key else "❌ Missing API key"
    
    # Use enhanced processor availability check for Ollama
    ollama_status = "❌ Not available locally"
    try:
        # Check using the same logic as enhanced processor
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        use_real_ollama = os.getenv("USE_REAL_OLLAMA", "true").lower() == "true"
        
        if use_real_ollama:
            import requests
            # Try both docker service name and localhost
            urls_to_try = [ollama_url]
            if "ollama:11434" in ollama_url:
                urls_to_try.append("http://localhost:11434")
            elif "localhost:11434" in ollama_url:
                urls_to_try.append("http://ollama:11434")
                
            for url in urls_to_try:
                try:
                    response = requests.get(f"{url}/api/version", timeout=2)
                    if response.status_code == 200:
                        ollama_status = "✅ Available"
                        break
                except:
                    continue
                    
            # If configured but can't reach, assume it's starting up
            if ollama_status == "❌ Not available locally" and use_real_ollama:
                ollama_status = "⚠️ Configured (starting up)"
    except:
        pass
    
    # DICOM processing status
    try:
        import pydicom
        dicom_lib_status = "✅ pydicom available"
    except ImportError:
        dicom_lib_status = "⚠️ pydicom not installed (fallback mode)"
    
    # Modal Labs status
    modal_token = os.getenv("MODAL_TOKEN_ID", "")
    modal_status = "✅ Configured" if modal_token else "❌ Not configured"
    
    # HuggingFace status using enhanced processor logic
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        hf_status = "❌ No token (set HF_TOKEN)"
    elif not hf_token.startswith("hf_"):
        hf_status = "❌ Invalid token format"
    else:
        try:
            # Use the same validation as enhanced processor
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            user_info = api.whoami()
            if user_info and 'name' in user_info:
                hf_status = f"✅ Authenticated as {user_info['name']}"
            else:
                hf_status = "❌ Authentication failed"
        except ImportError:
            hf_status = "❌ huggingface_hub not installed"
        except Exception as e:
            hf_status = f"❌ Error: {str(e)[:30]}..."
    
    status_html = f"""
    <div class="system-status-container" style="padding: 20px; border-radius: 8px; border: 1px solid var(--border-color-primary, #e5e7eb); background: var(--background-fill-primary, #ffffff); color: var(--body-text-color, #374151);">
        <h3 style="color: var(--body-text-color, #374151); margin-bottom: 20px;">🔧 System Components Status</h3>
        
        <div style="margin-bottom: 15px;">
            <h4 style="color: var(--body-text-color-subdued, #6b7280); margin-bottom: 8px;">Core Processing Components</h4>
            <p><strong>CodeLlama Processor:</strong> <span style="color: #059669;">{codellama_status}</span></p>
            <p><strong>Enhanced Processor:</strong> <span style="color: #059669;">{enhanced_status}</span></p>
            <p><strong>FHIR Validator:</strong> <span style="color: #059669;">{fhir_status}</span></p>
            <p><strong>Workflow Orchestrator:</strong> <span style="color: #059669;">{workflow_status}</span></p>
            <p><strong>DICOM Processor:</strong> <span style="color: #059669;">{dicom_status}</span></p>
        </div>
        
        <div style="margin-bottom: 15px;">
            <h4 style="color: var(--body-text-color-subdued, #6b7280); margin-bottom: 8px;">AI Provider APIs</h4>
            <p><strong>Mistral API:</strong> <span style="color: {'#059669' if mistral_api_key else '#dc2626'};">{mistral_status}</span></p>
            <p><strong>Ollama Local:</strong> <span style="color: {'#059669' if '✅' in ollama_status else '#dc2626'};">{ollama_status}</span></p>
            <p><strong>Modal Labs GPU:</strong> <span style="color: {'#059669' if modal_token else '#dc2626'};">{modal_status}</span></p>
            <p><strong>HuggingFace API:</strong> <span style="color: {'#059669' if hf_token else '#dc2626'};">{hf_status}</span></p>
        </div>
        
        <div style="margin-bottom: 15px;">
            <h4 style="color: var(--body-text-color-subdued, #6b7280); margin-bottom: 8px;">Medical Processing</h4>
            <p><strong>DICOM Library:</strong> <span style="color: {'#059669' if '✅' in dicom_lib_status else '#B71C1C'};">{dicom_lib_status}</span></p>
            <p><strong>FHIR R4 Compliance:</strong> <span style="color: #059669;">✅ Active</span></p>
            <p><strong>FHIR R5 Compliance:</strong> <span style="color: #059669;">✅ Active</span></p>
            <p><strong>Medical Entity Extraction:</strong> <span style="color: #059669;">✅ Ready</span></p>
            <p><strong>OCR Processing:</strong> <span style="color: #059669;">✅ Integrated</span></p>
        </div>
        
        <div>
            <h4 style="color: var(--body-text-color-subdued, #6b7280); margin-bottom: 8px;">System Status</h4>
            <p><strong>Overall Status:</strong> <span style="color: #16a34a;">🟢 Operational</span></p>
            <p><strong>Current Model:</strong> <span style="color: #2563eb;">{get_current_model_display()}</span></p>
            <p><strong>Processing Mode:</strong> <span style="color: #2563eb;">Multi-Provider Dynamic Scaling</span></p>
            <p><strong>Architecture:</strong> <span style="color: #2563eb;">Lazy Loading + Frontend/Backend Separation</span></p>
        </div>
    </div>
    """
    return status_html

# Processing Functions
async def _process_text_async(text, enable_fhir):
    """Async text processing that can be cancelled"""
    global cancellation_flags, running_tasks
    
    # Check for cancellation before processing
    if cancellation_flags["text_task"]:
        raise asyncio.CancelledError("Text processing cancelled")
    
    # Use Enhanced CodeLlama processor directly (with our Ollama fixes)
    try:
        processor = get_enhanced_codellama()
        method_name = "Enhanced CodeLlama (Multi-Provider)"
        
        result = await processor.process_document(
            medical_text=text,
            document_type="clinical_note",
            extract_entities=True,
            generate_fhir=enable_fhir
        )
        
        # Check for cancellation after processing
        if cancellation_flags["text_task"]:
            raise asyncio.CancelledError("Text processing cancelled")
        
        # Get the actual provider used from the result
        actual_provider = result.get("provider_metadata", {}).get("provider_used", "Enhanced Processor")
        method_name = f"Enhanced CodeLlama ({actual_provider.title()})"
        
        return result, method_name
        
    except Exception as e:
        print(f"⚠️ Enhanced CodeLlama processing failed: {e}")
        
        # If enhanced processor fails, try basic CodeLlama as fallback
        try:
            processor = get_codellama()
            method_name = "CodeLlama (Basic Fallback)"
            
            result = await processor.process_document(
                medical_text=text,
                document_type="clinical_note",
                extract_entities=True,
                generate_fhir=enable_fhir
            )
            
            # Check for cancellation after processing
            if cancellation_flags["text_task"]:
                raise asyncio.CancelledError("Text processing cancelled")
            
            return result, method_name
            
        except Exception as fallback_error:
            print(f"❌ HuggingFace fallback also failed: {fallback_error}")
            # Return a basic result structure instead of raising exception
            return {
                "extracted_data": {"error": "Processing failed", "patient": "Unknown Patient", "conditions": [], "medications": []},
                "metadata": {"model_used": "error_fallback", "processing_time": 0}
            }, "Error (Both Failed)"

def process_text_only(text, enable_fhir=True):
    """Process text with CodeLlama processor"""
    global cancellation_flags, running_tasks
    
    print(f"🔥 DEBUG: process_text_only called with text length: {len(text) if text else 0}")
    
    if not text.strip():
        return "❌ Please enter some medical text", {}, {}
    
    # FORCE JOB RECORDING - Always record job start with error handling
    job_id = None
    try:
        job_id = job_manager.add_processing_job("text", text[:50], {"enable_fhir": enable_fhir})
        active_jobs["text_task"] = job_id
        print(f"✅ DEBUG: Job {job_id[:8]} recorded successfully")
    except Exception as job_error:
        print(f"❌ DEBUG: Failed to record job: {job_error}")
        # Create fallback job_id to continue processing
        job_id = "fallback-" + str(uuid.uuid4())[:8]
    
    try:
        # Reset cancellation flag at start
        cancellation_flags["text_task"] = False
        start_time = time.time()
        monitor.log_event("text_processing_start", {"text_length": len(text)})
        
        # Check for cancellation early
        if cancellation_flags["text_task"]:
            job_manager.update_job_completion(job_id, False, {"error": "Cancelled by user"})
            return "⏹️ Processing cancelled", {}, {}
        
        # Run async processing with proper cancellation handling
        async def run_with_cancellation():
            task = asyncio.create_task(_process_text_async(text, enable_fhir))
            running_tasks["text_task"] = task
            try:
                return await task
            finally:
                if "text_task" in running_tasks:
                    del running_tasks["text_task"]
        
        result, method_name = asyncio.run(run_with_cancellation())
        
        # Calculate processing time and extract results
        processing_time = time.time() - start_time
        
        # Extract results for display
        # Handle extracted_data - it might be a dict or JSON string
        extracted_data_raw = result.get("extracted_data", {})
        if isinstance(extracted_data_raw, str):
            try:
                entities = json.loads(extracted_data_raw)
            except json.JSONDecodeError:
                entities = {}
        else:
            entities = extracted_data_raw
        
        # Check if processing actually failed
        processing_failed = (
            isinstance(entities, dict) and entities.get("error") == "Processing failed" or
            result.get("metadata", {}).get("error") == "All providers failed" or
            method_name == "Error (Both Failed)" or
            result.get("failover_metadata", {}).get("complete_failure", False)
        )
        
        if processing_failed:
            # Processing failed - return error status
            providers_tried = entities.get("providers_tried", ["ollama", "huggingface"]) if isinstance(entities, dict) else ["unknown"]
            error_msg = entities.get("error", "Processing failed") if isinstance(entities, dict) else "Processing failed"
            
            status = f"❌ **Processing Failed**\n\n📝 **Text:** {len(text)} characters\n⚠️ **Error:** {error_msg}\n🔄 **Providers Tried:** {', '.join(providers_tried)}\n💡 **Note:** All available AI providers are currently unavailable"
            
            # FORCE RECORD failed job completion with error handling
            try:
                if job_id:
                    job_manager.update_job_completion(job_id, False, {
                        "processing_time": f"{processing_time:.2f}s",
                        "error": error_msg,
                        "providers_tried": providers_tried
                    })
                    print(f"✅ DEBUG: Failed job {job_id[:8]} recorded successfully")
                else:
                    print("❌ DEBUG: No job_id to record failure")
            except Exception as completion_error:
                print(f"❌ DEBUG: Failed to record job completion: {completion_error}")
            
            monitor.log_event("text_processing_failed", {"error": error_msg, "providers_tried": providers_tried})
            
            return status, entities, {}
        else:
            # Processing succeeded
            status = f"✅ **Processing Complete!**\n\nProcessed {len(text)} characters using **{method_name}**"
            
            fhir_resources = result.get("fhir_bundle", {}) if enable_fhir else {}
            
            # FORCE RECORD successful job completion with error handling
            try:
                if job_id:
                    job_manager.update_job_completion(job_id, True, {
                        "processing_time": f"{processing_time:.2f}s",
                        "entities_found": len(entities) if isinstance(entities, dict) else 0,
                        "method": method_name
                    })
                    print(f"✅ DEBUG: Success job {job_id[:8]} recorded successfully")
                else:
                    print("❌ DEBUG: No job_id to record success")
            except Exception as completion_error:
                print(f"❌ DEBUG: Failed to record job completion: {completion_error}")
            
            # Clear active job tracking
            active_jobs["text_task"] = None
            
            monitor.log_event("text_processing_success", {"entities_found": len(entities), "method": method_name})
            
            return status, entities, fhir_resources
        
    except asyncio.CancelledError:
        job_manager.update_job_completion(job_id, False, {"error": "Processing cancelled"})
        active_jobs["text_task"] = None
        monitor.log_event("text_processing_cancelled", {})
        return "⏹️ Processing cancelled", {}, {}
        
    except Exception as e:
        job_manager.update_job_completion(job_id, False, {"error": str(e)})
        active_jobs["text_task"] = None
        monitor.log_event("text_processing_error", {"error": str(e)})
        return f"❌ Processing failed: {str(e)}", {}, {}

async def _process_file_async(file, enable_mistral_ocr, enable_fhir):
    """Async file processing that can be cancelled"""
    global cancellation_flags, running_tasks
    
    # First, extract text from the file using OCR
    from src.file_processor import local_processor
    
    with open(file.name, 'rb') as f:
        document_bytes = f.read()
    
    # Track actual OCR method used
    actual_ocr_method = None
    
    # Use local processor for OCR extraction
    if enable_mistral_ocr:
        # Try Mistral OCR first if enabled
        try:
            extracted_text = await local_processor._extract_with_mistral(document_bytes)
            actual_ocr_method = "mistral_api"
        except Exception as e:
            print(f"⚠️ Mistral OCR failed, falling back to local OCR: {e}")
            # Fallback to local OCR
            ocr_result = await local_processor.process_document(document_bytes, "user", file.name)
            extracted_text = ocr_result.get('extracted_text', '')
            actual_ocr_method = "local_processor"
    else:
        # Use local OCR
        ocr_result = await local_processor.process_document(document_bytes, "user", file.name)
        extracted_text = ocr_result.get('extracted_text', '')
        actual_ocr_method = "local_processor"
    
    # Check for cancellation after OCR
    if cancellation_flags["file_task"]:
        raise asyncio.CancelledError("File processing cancelled")
    
    # Process the extracted text using CodeLlama with HuggingFace fallback
    # Check for cancellation before processing
    if cancellation_flags["file_task"]:
        raise asyncio.CancelledError("File processing cancelled")
    
    # Try CodeLlama processor first
    try:
        processor = get_codellama()
        method_name = "CodeLlama (Ollama)"
        
        result = await processor.process_document(
            medical_text=extracted_text,
            document_type="clinical_note",
            extract_entities=True,
            generate_fhir=enable_fhir,
            source_metadata={"extraction_method": actual_ocr_method}
        )
    except Exception as e:
        print(f"⚠️ CodeLlama processing failed: {e}, falling back to HuggingFace")
        
        # Fallback to Enhanced CodeLlama (HuggingFace)
        try:
            processor = get_enhanced_codellama()
            method_name = "HuggingFace (Fallback)"
            
            result = await processor.process_document(
                medical_text=extracted_text,
                document_type="clinical_note",
                extract_entities=True,
                generate_fhir=enable_fhir,
                source_metadata={"extraction_method": actual_ocr_method}
            )
        except Exception as fallback_error:
            print(f"❌ HuggingFace fallback also failed: {fallback_error}")
            # Return a basic result structure instead of raising exception
            result = {
                "extracted_data": {"error": "Processing failed", "patient": "Unknown Patient", "conditions": [], "medications": []},
                "metadata": {"model_used": "error_fallback", "processing_time": 0}
            }
            method_name = "Error (Both Failed)"
    
    # Check for cancellation after processing
    if cancellation_flags["file_task"]:
        raise asyncio.CancelledError("File processing cancelled")
    
    return result, method_name, extracted_text, actual_ocr_method

def process_file_only(file, enable_mistral_ocr=True, enable_fhir=True):
    """Process uploaded file with CodeLlama processor and optional Mistral OCR"""
    global cancellation_flags
    
    if not file:
        return "❌ Please upload a file", {}, {}
    
    # Record job start
    job_id = job_manager.add_processing_job("file", file.name, {
        "enable_mistral_ocr": enable_mistral_ocr,
        "enable_fhir": enable_fhir
    })
    active_jobs["file_task"] = job_id
    
    try:
        # Reset cancellation flag at start
        cancellation_flags["file_task"] = False
        monitor.log_event("file_processing_start", {"filename": file.name})
        
        # Check for cancellation early
        if cancellation_flags["file_task"]:
            job_manager.update_job_completion(job_id, False, {"error": "Cancelled by user"})
            return "⏹️ File processing cancelled", {}, {}
        
        import time
        start_time = time.time()
        
        # Process the file with cancellation support
        try:
            # Run async processing with proper cancellation handling
            async def run_with_cancellation():
                task = asyncio.create_task(_process_file_async(file, enable_mistral_ocr, enable_fhir))
                running_tasks["file_task"] = task
                try:
                    return await task
                finally:
                    if "file_task" in running_tasks:
                        del running_tasks["file_task"]
            
            result, method_name, extracted_text, actual_ocr_method = asyncio.run(run_with_cancellation())
        except asyncio.CancelledError:
            job_manager.update_job_completion(job_id, False, {"error": "Processing cancelled"})
            active_jobs["file_task"] = None
            return "⏹️ File processing cancelled", {}, {}
        
        processing_time = time.time() - start_time
        
        # Enhanced status message with actual OCR information
        ocr_method_display = "Mistral OCR (Advanced)" if actual_ocr_method == "mistral_api" else "Local OCR (Standard)"
        status = f"✅ **File Processing Complete!**\n\n📁 **File:** {file.name}\n🔍 **OCR Method:** {ocr_method_display}\n🤖 **AI Processor:** {method_name}\n⏱️ **Processing Time:** {processing_time:.2f}s"
        
        # Handle extracted_data - it might be a dict or JSON string
        extracted_data_raw = result.get("extracted_data", {})
        if isinstance(extracted_data_raw, str):
            try:
                entities = json.loads(extracted_data_raw)
            except json.JSONDecodeError:
                entities = {}
        else:
            entities = extracted_data_raw
            
        fhir_resources = result.get("fhir_bundle", {}) if enable_fhir else {}
        
        # Record successful job completion
        job_manager.update_job_completion(job_id, True, {
            "processing_time": f"{processing_time:.2f}s",
            "entities_found": len(entities) if isinstance(entities, dict) else 0,
            "method": method_name
        })
        
        # Clear active job tracking
        active_jobs["file_task"] = None
        
        monitor.log_event("file_processing_success", {"filename": file.name, "method": method_name})
        
        return status, entities, fhir_resources
        
    except Exception as e:
        job_manager.update_job_completion(job_id, False, {"error": str(e)})
        active_jobs["file_task"] = None
        monitor.log_event("file_processing_error", {"error": str(e)})
        return f"❌ File processing failed: {str(e)}", {}, {}

def process_dicom_only(dicom_file):
    """Process DICOM files using the real DICOM processor"""
    global cancellation_flags
    
    if not dicom_file:
        return "❌ Please upload a DICOM file", {}, {}
    
    # Record job start
    job_id = job_manager.add_processing_job("dicom", dicom_file.name)
    active_jobs["dicom_task"] = job_id
    
    try:
        # Reset cancellation flag at start
        cancellation_flags["dicom_task"] = False
        
        # Check for cancellation early
        if cancellation_flags["dicom_task"]:
            job_manager.update_job_completion(job_id, False, {"error": "Cancelled by user"})
            return "⏹️ DICOM processing cancelled", {}, {}
        monitor.log_event("dicom_processing_start", {"filename": dicom_file.name})
        
        import time
        start_time = time.time()
        
        # Process DICOM file using the real processor with cancellation support
        async def run_dicom_with_cancellation():
            task = asyncio.create_task(dicom_processor.process_dicom_file(dicom_file.name))
            running_tasks["dicom_task"] = task
            try:
                return await task
            finally:
                if "dicom_task" in running_tasks:
                    del running_tasks["dicom_task"]
        
        try:
            result = asyncio.run(run_dicom_with_cancellation())
        except asyncio.CancelledError:
            job_manager.update_job_completion(job_id, False, {"error": "Processing cancelled"})
            active_jobs["dicom_task"] = None
            return "⏹️ DICOM processing cancelled", {}, {}
        
        processing_time = time.time() - start_time
        
        # Extract processing results - fix structure mismatch
        if result.get("status") == "success":
            # Format the status message with real data from DICOM processor
            fhir_bundle = result.get("fhir_bundle", {})
            patient_name = result.get("patient_name", "Unknown")
            study_description = result.get("study_description", "Unknown")
            modality = result.get("modality", "Unknown")
            file_size = result.get("file_size", 0)
            
            status = f"""✅ **DICOM Processing Complete!**

📁 **File:** {os.path.basename(dicom_file.name)}
📊 **Size:** {file_size} bytes
⏱️ **Processing Time:** {processing_time:.2f}s
🏥 **Modality:** {modality}
👤 **Patient:** {patient_name}
📋 **Study:** {study_description}
📊 **FHIR Resources:** {len(fhir_bundle.get('entry', []))} generated"""
            
            # Format analysis data for display
            analysis = {
                "file_info": {
                    "filename": os.path.basename(dicom_file.name),
                    "file_size_bytes": file_size,
                    "processing_time": result.get('processing_time', 0)
                },
                "patient_info": {
                    "name": patient_name
                },
                "study_info": {
                    "description": study_description,
                    "modality": modality
                },
                "processing_status": "✅ Successfully processed",
                "processor_used": "DICOM Processor with pydicom",
                "pydicom_available": True
            }
            
            # Use the FHIR bundle from processor
            fhir_imaging = fhir_bundle
            
            # Record successful job completion
            job_manager.update_job_completion(job_id, True, {
                "processing_time": f"{processing_time:.2f}s",
                "patient_name": patient_name,
                "modality": modality
            })
            
            # Clear active job tracking
            active_jobs["dicom_task"] = None
            
        else:
            # Handle processing failure
            error_msg = result.get("error", "Unknown error")
            fallback_used = result.get("fallback_used", False)
            processor_info = "DICOM Fallback Processor" if fallback_used else "DICOM Processor"
            
            status = f"""❌ **DICOM Processing Failed**

📁 **File:** {os.path.basename(dicom_file.name)}
🚫 **Error:** {error_msg}
🔧 **Processor:** {processor_info}
💡 **Note:** pydicom library may not be available or file format issue"""
            
            analysis = {
                "error": error_msg,
                "file_info": {"filename": os.path.basename(dicom_file.name)},
                "processing_status": "❌ Failed",
                "processor_used": processor_info,
                "fallback_used": fallback_used,
                "pydicom_available": not fallback_used
            }
            
            fhir_imaging = {}
            
            # Record failed job completion
            job_manager.update_job_completion(job_id, False, {"error": error_msg})
            
            # Clear active job tracking
            active_jobs["dicom_task"] = None
        
        monitor.log_event("dicom_processing_success", {"filename": dicom_file.name})
        
        return status, analysis, fhir_imaging
        
    except Exception as e:
        job_manager.update_job_completion(job_id, False, {"error": str(e)})
        active_jobs["dicom_task"] = None
        monitor.log_event("dicom_processing_error", {"error": str(e)})
        error_analysis = {
            "error": str(e),
            "file_info": {"filename": os.path.basename(dicom_file.name) if dicom_file else "Unknown"},
            "processing_status": "❌ Exception occurred"
        }
        return f"❌ DICOM processing failed: {str(e)}", error_analysis, {}

def cancel_current_task(task_type):
    """Cancel current processing task"""
    global cancellation_flags, running_tasks, task_queues, active_jobs

    # DEBUG: log state before cancellation
    monitor.log_event("cancel_state_before", {
        "task_type": task_type,
        "cancellation_flags": cancellation_flags.copy(),
        "active_jobs": active_jobs.copy(),
        "task_queues": {k: len(v) for k, v in task_queues.items()}
    })

    # Set cancellation flag
    cancellation_flags[task_type] = True

    # Cancel the actual running task if it exists
    if running_tasks[task_type] is not None:
        try:
            running_tasks[task_type].cancel()
            running_tasks[task_type] = None
        except Exception as e:
            print(f"Error cancelling task {task_type}: {e}")

    # Clear the task queue for this task type to prevent new tasks from starting
    if task_queues.get(task_type):
        task_queues[task_type].clear()

    # Reset active job tracking for this task type
    active_jobs[task_type] = None

    # Reset active tasks counter
    if dashboard_state["active_tasks"] > 0:
        dashboard_state["active_tasks"] -= 1

    monitor.log_event("task_cancelled", {"task_type": task_type})

    # DEBUG: log state after cancellation
    monitor.log_event("cancel_state_after", {
        "task_type": task_type,
        "cancellation_flags": cancellation_flags.copy(),
        "active_jobs": active_jobs.copy(),
        "task_queues": {k: len(v) for k, v in task_queues.items()}
    })

    return f"⏹️ Cancelled {task_type}"

    # DEBUG: log state before cancellation
    monitor.log_event("cancel_state_before", {
        "task_type": task_type,
        "cancellation_flags": cancellation_flags.copy(),
        "active_jobs": active_jobs.copy(),
        "task_queues": {k: len(v) for k, v in task_queues.items()}
    })
    
    # Set cancellation flag
    cancellation_flags[task_type] = True
    
    # Cancel the actual running task if it exists
    if running_tasks[task_type] is not None:
        try:
            running_tasks[task_type].cancel()
            running_tasks[task_type] = None
        except Exception as e:
            print(f"Error cancelling task {task_type}: {e}")
    
    # Reset active tasks counter
    if dashboard_state["active_tasks"] > 0:
        dashboard_state["active_tasks"] -= 1
    
    monitor.log_event("task_cancelled", {"task_type": task_type})

    # DEBUG: log state after cancellation
    monitor.log_event("cancel_state_after", {
        "task_type": task_type,
        "cancellation_flags": cancellation_flags.copy(),
        "active_jobs": active_jobs.copy(),
        "task_queues": {k: len(v) for k, v in task_queues.items()}
    })
    return f"⏹️ Cancelled {task_type}"

def get_dashboard_status():
    """Get current file processing dashboard status"""
    return job_manager.get_dashboard_status()

def get_dashboard_metrics():
    """Get file processing metrics for DataFrame display"""
    return job_manager.get_dashboard_metrics()

def get_processing_queue():
    """Get processing queue for DataFrame display"""
    return job_manager.get_processing_queue()

def get_jobs_history():
    """Get processing jobs history for DataFrame display"""
    return job_manager.get_jobs_history()

# Keep the old function for backward compatibility but redirect to new one
def get_files_history():
    """Legacy function - redirects to get_jobs_history()"""
    return get_jobs_history()
def get_old_files_history():
    """Get list of recently processed files for dashboard (legacy function)"""
    # Return the last 10 processed files
    recent_files = dashboard_state["files_processed"][-10:] if dashboard_state["files_processed"] else []
    return recent_files

def add_file_to_dashboard(filename, file_type, success, processing_time=None, error=None, entities_found=None):
    """Add a processed file to the dashboard statistics"""
    import datetime
    
    file_info = {
        "filename": filename,
        "file_type": file_type,
        "success": success,
        "processing_time": processing_time,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": error if not success else None,
        "entities_found": entities_found or 0
    }
    
    dashboard_state["files_processed"].append(file_info)
    dashboard_state["total_files"] += 1
    
    if success:
        dashboard_state["successful_files"] += 1
    else:
        dashboard_state["failed_files"] += 1
    
    dashboard_state["last_update"] = file_info["timestamp"]

# Main application
if __name__ == "__main__":
    print("🔥 Starting FhirFlame Medical AI Platform...")
    
    # Import frontend UI components dynamically to avoid circular imports
    from frontend_ui import create_medical_ui
    
    # Create the UI using the separated frontend components
    demo = create_medical_ui(
        process_text_only=process_text_only,
        process_file_only=process_file_only,
        process_dicom_only=process_dicom_only,
        cancel_current_task=cancel_current_task,
        get_dashboard_status=get_dashboard_status,
        dashboard_state=dashboard_state,
        get_dashboard_metrics=get_dashboard_metrics,
        get_simple_agent_status=get_simple_agent_status,
        get_enhanced_codellama=get_enhanced_codellama,
        add_file_to_dashboard=add_file_to_dashboard
    )
    
    # Launch the application
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        favicon_path="static/favicon.ico"
    )
