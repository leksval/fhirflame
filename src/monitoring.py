"""
FhirFlame Unified Monitoring and Observability
Comprehensive Langfuse integration for medical AI workflows with centralized monitoring
"""

import time
import json
from typing import Dict, Any, Optional, List, Union
from functools import wraps
from contextlib import contextmanager

# Langfuse monitoring with environment configuration
try:
    import os
    import sys
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables
    
    # Comprehensive test environment detection
    is_testing = (
        os.getenv("DISABLE_LANGFUSE") == "true" or
        os.getenv("PYTEST_RUNNING") == "true" or
        os.getenv("PYTEST_CURRENT_TEST") is not None or
        "pytest" in str(sys.argv) or
        "pytest" in os.getenv("_", "") or
        "test" in os.path.basename(os.getenv("_", "")) or
        any("pytest" in arg for arg in sys.argv) or
        any("test" in arg for arg in sys.argv)
    )
    
    if is_testing:
        print("🧪 Test environment detected - disabling Langfuse")
        langfuse = None
        LANGFUSE_AVAILABLE = False
    else:
        try:
            from langfuse import Langfuse
            
            # Check if Langfuse is properly configured
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            
            if not secret_key or not public_key:
                print("⚠️ Langfuse keys not configured - using local monitoring only")
                langfuse = None
                LANGFUSE_AVAILABLE = False
            else:
                # Initialize with environment variables and timeout settings
                try:
                    langfuse = Langfuse(
                        secret_key=secret_key,
                        public_key=public_key,
                        host=host,
                        timeout=2  # Very short timeout for faster failure detection
                    )
                    
                    # Test connection with a simple call
                    try:
                        # Quick health check - if this fails, disable Langfuse
                        # Use the newer Langfuse API for health check
                        if hasattr(langfuse, 'trace'):
                            test_trace = langfuse.trace(name="connection_test")
                            if test_trace:
                                test_trace.update(output={"status": "connection_ok"})
                        else:
                            # Fallback: just test if the client exists
                            _ = str(langfuse)
                        LANGFUSE_AVAILABLE = True
                        print(f"🔍 Langfuse initialized: {host}")
                    except Exception as connection_error:
                        print(f"⚠️ Langfuse connection test failed: {connection_error}")
                        print("🔄 Continuing with local-only monitoring...")
                        langfuse = None
                        LANGFUSE_AVAILABLE = False
                        
                except Exception as init_error:
                    print(f"⚠️ Langfuse client initialization failed: {init_error}")
                    print("🔄 Continuing with local-only monitoring...")
                    langfuse = None
                    LANGFUSE_AVAILABLE = False
        except Exception as langfuse_error:
            print(f"⚠️ Langfuse initialization failed: {langfuse_error}")
            langfuse = None
            LANGFUSE_AVAILABLE = False
        
except ImportError:
    langfuse = None
    LANGFUSE_AVAILABLE = False
    print("⚠️ Langfuse package not available - using local monitoring only")
except Exception as e:
    langfuse = None
    LANGFUSE_AVAILABLE = False
    print(f"⚠️ Langfuse initialization failed: {e}")
    print(f"🔄 Continuing with local-only monitoring...")

# LangChain monitoring
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class FhirFlameMonitor:
    """Comprehensive monitoring for FhirFlame medical AI workflows"""
    
    def __init__(self):
        self.langfuse = langfuse if LANGFUSE_AVAILABLE else None
        self.session_id = f"fhirflame_{int(time.time())}" if self.langfuse else None
        
    def track_operation(self, operation_name: str):
        """Universal decorator to track any operation"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                trace = None
                
                if self.langfuse:
                    try:
                        # Use newer Langfuse API if available
                        if hasattr(self.langfuse, 'trace'):
                            trace = self.langfuse.trace(
                                name=operation_name,
                                session_id=self.session_id
                            )
                        else:
                            trace = None
                    except Exception:
                        trace = None
                
                try:
                    result = await func(*args, **kwargs)
                    processing_time = time.time() - start_time
                    
                    if trace:
                        trace.update(
                            output={"status": "success", "processing_time": processing_time},
                            metadata={"operation": operation_name}
                        )
                    
                    return result
                    
                except Exception as e:
                    if trace:
                        trace.update(
                            output={"status": "error", "error": str(e)},
                            metadata={"processing_time": time.time() - start_time}
                        )
                    raise
                    
            return wrapper
        return decorator
    
    def log_event(self, event_name: str, properties: Dict[str, Any]):
        """Log any event with properties"""

        # LOCAL DEBUG: write log to local file
        try:
            import os
            os.makedirs('/app/logs', exist_ok=True)
            with open('/app/logs/debug_events.log', 'a') as f:
                f.write(f"{time.time()} {event_name} {json.dumps(properties)}\n")
        except Exception:
            pass
        if self.langfuse:
            try:
                # Use newer Langfuse API if available
                if hasattr(self.langfuse, 'event'):
                    self.langfuse.event(
                        name=event_name,
                        properties=properties,
                        session_id=self.session_id
                    )
                elif hasattr(self.langfuse, 'log'):
                    # Fallback to older API
                    self.langfuse.log(
                        level="INFO",
                        message=event_name,
                        extra=properties
                    )
            except Exception:
                # Silently fail for logging to avoid disrupting workflow
                # Disable Langfuse for this session if it keeps failing
                self.langfuse = None
    
    # === AI MODEL PROCESSING MONITORING ===
    
    def log_ollama_api_call(self, model: str, url: str, prompt_length: int, success: bool = True, response_time: float = 0.0, status_code: int = 200, error: str = None):
        """Log Ollama API call details"""
        self.log_event("ollama_api_call", {
            "model": model,
            "url": url,
            "prompt_length": prompt_length,
            "success": success,
            "response_time": response_time,
            "status_code": status_code,
            "error": error,
            "api_type": "ollama_generate"
        })
    
    def log_ai_generation(self, model: str, response_length: int, processing_time: float, entities_found: int, confidence: float, processing_mode: str):
        """Log AI text generation results"""
        self.log_event("ai_generation_complete", {
            "model": model,
            "response_length": response_length,
            "processing_time": processing_time,
            "entities_found": entities_found,
            "confidence_score": confidence,
            "processing_mode": processing_mode,
            "generation_type": "medical_entity_extraction"
        })
    
    def log_ai_parsing(self, success: bool, response_format: str, entities_extracted: int, parsing_time: float, error: str = None):
        """Log AI response parsing results"""
        self.log_event("ai_response_parsing", {
            "parsing_success": success,
            "response_format": response_format,
            "entities_extracted": entities_extracted,
            "parsing_time": parsing_time,
            "error": error,
            "parser_type": "json_medical_extractor"
        })
    
    def log_data_transformation(self, input_format: str, output_format: str, entities_transformed: int, transformation_time: float, complex_nested: bool = False):
        """Log data transformation operations"""
        self.log_event("data_transformation", {
            "input_format": input_format,
            "output_format": output_format,
            "entities_transformed": entities_transformed,
            "transformation_time": transformation_time,
            "complex_nested_input": complex_nested,
            "transformer_type": "ai_to_pydantic"
        })
    
    # === MEDICAL PROCESSING MONITORING ===
    
    def log_medical_processing(self, entities_found: int, confidence: float, processing_time: float, processing_mode: str = "unknown", model_used: str = "codellama:13b-instruct"):
        """Log medical processing results"""
        self.log_event("medical_processing_complete", {
            "entities_found": entities_found,
            "confidence_score": confidence,
            "processing_time": processing_time,
            "processing_mode": processing_mode,
            "model_used": model_used,
            "extraction_type": "clinical_entities"
        })
    
    def log_medical_entity_extraction(self, conditions: int, medications: int, vitals: int, procedures: int, patient_info_found: bool, confidence: float):
        """Log detailed medical entity extraction"""
        self.log_event("medical_entity_extraction", {
            "conditions_found": conditions,
            "medications_found": medications,
            "vitals_found": vitals,
            "procedures_found": procedures,
            "patient_info_extracted": patient_info_found,
            "total_entities": conditions + medications + vitals + procedures,
            "confidence_score": confidence,
            "extraction_category": "clinical_data"
        })
    
    def log_rule_based_processing(self, entities_found: int, conditions: int, medications: int, vitals: int, confidence: float, processing_time: float):
        """Log rule-based processing fallback"""
        self.log_event("rule_based_processing_complete", {
            "total_entities": entities_found,
            "conditions_found": conditions,
            "medications_found": medications,
            "vitals_found": vitals,
            "confidence_score": confidence,
            "processing_time": processing_time,
            "processing_mode": "rule_based_fallback",
            "fallback_triggered": True
        })
    
    # === FHIR VALIDATION MONITORING ===
    
    def log_fhir_validation(self, is_valid: bool, compliance_score: float, validation_level: str, fhir_version: str = "R4", resource_types: List[str] = None):
        """Log FHIR validation results"""
        self.log_event("fhir_validation_complete", {
            "is_valid": is_valid,
            "compliance_score": compliance_score,
            "validation_level": validation_level,
            "fhir_version": fhir_version,
            "resource_types": resource_types or [],
            "validation_type": "bundle_validation"
        })
    
    def log_fhir_structure_validation(self, structure_valid: bool, resource_types: List[str], validation_time: float, errors: List[str] = None):
        """Log FHIR structure validation"""
        self.log_event("fhir_structure_validation", {
            "structure_valid": structure_valid,
            "resource_types_detected": resource_types,
            "validation_time": validation_time,
            "error_count": len(errors) if errors else 0,
            "validation_errors": errors or [],
            "validator_type": "pydantic_fhir"
        })
    
    def log_fhir_terminology_validation(self, terminology_valid: bool, codes_validated: int, loinc_found: bool, snomed_found: bool, validation_time: float):
        """Log FHIR terminology validation"""
        self.log_event("fhir_terminology_validation", {
            "terminology_valid": terminology_valid,
            "codes_validated": codes_validated,
            "loinc_codes_found": loinc_found,
            "snomed_codes_found": snomed_found,
            "validation_time": validation_time,
            "coding_systems": ["LOINC" if loinc_found else "", "SNOMED" if snomed_found else ""],
            "validator_type": "medical_terminology"
        })
    
    def log_hipaa_compliance_check(self, is_compliant: bool, phi_protected: bool, security_met: bool, validation_time: float, errors: List[str] = None):
        """Log HIPAA compliance validation"""
        self.log_event("hipaa_compliance_check", {
            "hipaa_compliant": is_compliant,
            "phi_properly_protected": phi_protected,
            "security_requirements_met": security_met,
            "validation_time": validation_time,
            "compliance_errors": errors or [],
            "compliance_level": "healthcare_grade",
            "validator_type": "hipaa_checker"
        })
    
    def log_fhir_bundle_generation(self, patient_resources: int, condition_resources: int, observation_resources: int, generation_time: float, success: bool):
        """Log FHIR bundle generation"""
        self.log_event("fhir_bundle_generation", {
            "patient_resources": patient_resources,
            "condition_resources": condition_resources,
            "observation_resources": observation_resources,
            "total_resources": patient_resources + condition_resources + observation_resources,
            "generation_time": generation_time,
            "generation_success": success,
            "bundle_type": "document",
            "generator_type": "pydantic_fhir"
        })
    
    # === WORKFLOW MONITORING ===
    
    def log_document_processing_start(self, document_type: str, text_length: int, extract_entities: bool, generate_fhir: bool):
        """Log start of document processing"""
        self.log_event("document_processing_start", {
            "document_type": document_type,
            "text_length": text_length,
            "extract_entities": extract_entities,
            "generate_fhir": generate_fhir,
            "workflow_stage": "initialization"
        })
    
    def log_document_processing_complete(self, success: bool, processing_time: float, entities_found: int, fhir_generated: bool, quality_score: float):
        """Log completion of document processing"""
        self.log_event("document_processing_complete", {
            "processing_success": success,
            "total_processing_time": processing_time,
            "entities_extracted": entities_found,
            "fhir_bundle_generated": fhir_generated,
            "quality_score": quality_score,
            "workflow_stage": "completion"
        })
    
    def log_workflow_summary(self, documents_processed: int, successful_documents: int, total_time: float, average_time: float, monitoring_active: bool):
        """Log overall workflow summary"""
        self.log_event("workflow_summary", {
            "documents_processed": documents_processed,
            "successful_documents": successful_documents,
            "failed_documents": documents_processed - successful_documents,
            "success_rate": successful_documents / documents_processed if documents_processed > 0 else 0,
            "total_processing_time": total_time,
            "average_time_per_document": average_time,
            "monitoring_active": monitoring_active,
            "workflow_type": "real_medical_processing"
        })
    
    def log_mcp_tool(self, tool_name: str, success: bool, processing_time: float, input_size: int = 0, entities_found: int = 0):
        """Log MCP tool execution"""
        self.log_event("mcp_tool_execution", {
            "tool_name": tool_name,
            "success": success,
            "processing_time": processing_time,
            "input_size": input_size,
            "entities_found": entities_found,
            "mcp_protocol_version": "2024-11-05"
        })
        
    def log_mcp_server_start(self, server_name: str, tools_count: int, port: int):
        """Log MCP server startup"""
        self.log_event("mcp_server_startup", {
            "server_name": server_name,
            "tools_available": tools_count,
            "port": port,
            "protocol": "mcp_2024"
        })
        
    def log_mcp_authentication(self, auth_method: str, success: bool, user_id: str = None):
        """Log MCP authentication events"""
        self.log_event("mcp_authentication", {
            "auth_method": auth_method,
            "success": success,
            "user_id": user_id or "anonymous",
            "security_level": "a2a_api"
        })
    
    # === MISTRAL OCR MONITORING ===
    
    def log_mistral_ocr_processing(self, document_size: int, extraction_time: float, success: bool, text_length: int = 0, error: str = None):
        """Log Mistral OCR API processing"""
        self.log_event("mistral_ocr_processing", {
            "document_size_bytes": document_size,
            "extraction_time": extraction_time,
            "success": success,
            "extracted_text_length": text_length,
            "error": error,
            "ocr_provider": "mistral_api"
        })
    
    def log_ocr_workflow_integration(self, ocr_method: str, agent_processing_time: float, total_workflow_time: float, entities_found: int):
        """Log complete OCR → Agent workflow integration"""
        self.log_event("ocr_workflow_integration", {
            "ocr_method": ocr_method,
            "agent_processing_time": agent_processing_time,
            "total_workflow_time": total_workflow_time,
            "entities_extracted": entities_found,
            "workflow_type": "ocr_to_agent_pipeline"
        })
    
    # === A2A API MONITORING ===
    
    def log_a2a_api_request(self, endpoint: str, method: str, auth_method: str, request_size: int, user_id: str = None):
        """Log A2A API request"""
        self.log_event("a2a_api_request", {
            "endpoint": endpoint,
            "method": method,
            "auth_method": auth_method,
            "request_size_bytes": request_size,
            "user_id": user_id or "anonymous",
            "api_version": "v1.0"
        })
    
    def log_a2a_api_response(self, endpoint: str, status_code: int, response_time: float, success: bool, entities_processed: int = 0):
        """Log A2A API response"""
        self.log_event("a2a_api_response", {
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time": response_time,
            "success": success,
            "entities_processed": entities_processed,
            "api_type": "rest_a2a"
        })
    
    def log_a2a_authentication(self, auth_provider: str, success: bool, auth_time: float, user_claims: Dict[str, Any] = None):
        """Log A2A authentication events"""
        self.log_event("a2a_authentication", {
            "auth_provider": auth_provider,
            "success": success,
            "auth_time": auth_time,
            "user_claims": user_claims or {},
            "security_level": "production" if auth_provider == "auth0" else "development"
        })
    
    # === MODAL SCALING MONITORING ===
    
    def log_modal_function_call(self, function_name: str, gpu_type: str, processing_time: float, cost_estimate: float, container_id: str):
        """Log Modal function execution"""
        self.log_event("modal_function_call", {
            "function_name": function_name,
            "gpu_type": gpu_type,
            "processing_time": processing_time,
            "cost_estimate": cost_estimate,
            "container_id": container_id,
            "cloud_provider": "modal_labs"
        })
    
    def log_modal_scaling_event(self, event_type: str, container_count: int, gpu_utilization: str, auto_scaling: bool):
        """Log Modal auto-scaling events"""
        self.log_event("modal_scaling_event", {
            "event_type": event_type,  # scale_up, scale_down, container_start, container_stop
            "container_count": container_count,
            "gpu_utilization": gpu_utilization,
            "auto_scaling_active": auto_scaling,
            "scaling_provider": "modal_l4"
        })
    
    def log_modal_deployment(self, app_name: str, functions_deployed: int, success: bool, deployment_time: float):
        """Log Modal deployment events"""
        self.log_event("modal_deployment", {
            "app_name": app_name,
            "functions_deployed": functions_deployed,
            "deployment_success": success,
            "deployment_time": deployment_time,
            "deployment_target": "modal_serverless"
        })
    
    def log_modal_cost_tracking(self, daily_cost: float, requests_processed: int, cost_per_request: float, gpu_hours: float):
        """Log Modal cost analytics"""
        self.log_event("modal_cost_tracking", {
            "daily_cost": daily_cost,
            "requests_processed": requests_processed,
            "cost_per_request": cost_per_request,
            "gpu_hours_used": gpu_hours,
            "cost_optimization": "l4_gpu_auto_scaling"
        })
    
    # === DOCKER DEPLOYMENT MONITORING ===
    
    def log_docker_deployment(self, compose_file: str, services_started: int, success: bool, startup_time: float):
        """Log Docker Compose deployment"""
        self.log_event("docker_deployment", {
            "compose_file": compose_file,
            "services_started": services_started,
            "deployment_success": success,
            "startup_time": startup_time,
            "deployment_type": "docker_compose"
        })
    
    def log_docker_service_health(self, service_name: str, status: str, response_time: float, healthy: bool):
        """Log Docker service health checks"""
        self.log_event("docker_service_health", {
            "service_name": service_name,
            "status": status,
            "response_time": response_time,
            "healthy": healthy,
            "monitoring_type": "health_check"
        })
    
    # === ERROR AND PERFORMANCE MONITORING ===
    
    def log_error_event(self, error_type: str, error_message: str, stack_trace: str, component: str, severity: str = "error"):
        """Log error events with context"""
        self.log_event("error_event", {
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "component": component,
            "severity": severity,
            "timestamp": time.time()
        })
    
    def log_performance_metrics(self, component: str, cpu_usage: float, memory_usage: float, response_time: float, throughput: float):
        """Log performance metrics"""
        self.log_event("performance_metrics", {
            "component": component,
            "cpu_usage_percent": cpu_usage,
            "memory_usage_mb": memory_usage,
            "response_time": response_time,
            "throughput_requests_per_second": throughput,
            "metrics_type": "system_performance"
        })
    
    # === LANGFUSE TRACE UTILITIES ===
    
    def create_langfuse_trace(self, name: str, input_data: Dict[str, Any] = None, session_id: str = None) -> Any:
        """Create a Langfuse trace if available"""
        if self.langfuse:
            try:
                return self.langfuse.trace(
                    name=name,
                    input=input_data or {},
                    session_id=session_id or self.session_id
                )
            except Exception:
                return None
        return None
    
    def update_langfuse_trace(self, trace: Any, output: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        """Update a Langfuse trace if available"""
        if trace and self.langfuse:
            try:
                trace.update(
                    output=output or {},
                    metadata=metadata or {}
                )
            except Exception:
                pass
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        return {
            "langfuse_enabled": self.langfuse is not None,
            "session_id": self.session_id,
            "langfuse_host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com") if self.langfuse else None,
            "monitoring_active": True,
            "events_logged": True,
            "trace_collection": "enabled" if self.langfuse else "disabled"
        }
    
    @contextmanager
    def trace_operation(self, operation_name: str, input_data: Dict[str, Any] = None):
        """Context manager for tracing operations"""
        trace = None
        if self.langfuse:
            try:
                trace = self.langfuse.trace(
                    name=operation_name,
                    input=input_data or {},
                    session_id=self.session_id
                )
            except Exception:
                # Silently fail trace creation to avoid disrupting workflow
                trace = None
        
        start_time = time.time()
        try:
            yield trace
        except Exception as e:
            if trace:
                try:
                    trace.update(
                        output={"error": str(e), "status": "failed"},
                        metadata={"processing_time": time.time() - start_time}
                    )
                except Exception:
                    # Silently fail trace update
                    pass
            raise
        else:
            if trace:
                try:
                    trace.update(
                        metadata={"processing_time": time.time() - start_time, "status": "completed"}
                    )
                except Exception:
                    # Silently fail trace update
                    pass
    
    @contextmanager
    def trace_ai_processing(self, model: str, text_length: int, temperature: float, max_tokens: int):
        """Context manager specifically for AI processing operations"""
        with self.trace_operation("ai_model_processing", {
            "model": model,
            "input_length": text_length,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "processing_type": "medical_extraction"
        }) as trace:
            yield trace
    
    @contextmanager
    def trace_fhir_validation(self, validation_level: str, resource_count: int):
        """Context manager specifically for FHIR validation operations"""
        with self.trace_operation("fhir_validation_process", {
            "validation_level": validation_level,
            "resource_count": resource_count,
            "fhir_version": "R4",
            "validation_type": "comprehensive"
        }) as trace:
            yield trace
    
    @contextmanager
    def trace_document_workflow(self, document_type: str, text_length: int):
        """Context manager for complete document processing workflow"""
        with self.trace_operation("document_processing_workflow", {
            "document_type": document_type,
            "text_length": text_length,
            "workflow_type": "end_to_end_medical"
        }) as trace:
            yield trace
    
    def get_langchain_callback(self):
        """Get LangChain callback handler for monitoring"""
        if LANGCHAIN_AVAILABLE and self.langfuse:
            try:
                return self.langfuse.get_langchain_callback(session_id=self.session_id)
            except Exception:
                return None
        return None
    
    def process_with_langchain(self, text: str, operation: str = "document_processing"):
        """Process text using LangChain with monitoring"""
        if not LANGCHAIN_AVAILABLE:
            return {"processed_text": text, "chunks": [text]}
        
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " "]
            )
            
            chunks = splitter.split_text(text)
            
            self.log_event("langchain_processing", {
                "operation": operation,
                "chunk_count": len(chunks),
                "total_length": len(text)
            })
            
            return {"processed_text": text, "chunks": chunks}
            
        except Exception as e:
            self.log_event("langchain_error", {"error": str(e), "operation": operation})
            return {"processed_text": text, "chunks": [text], "error": str(e)}

# Global monitor instance
monitor = FhirFlameMonitor()

# Convenience decorators
def track_medical_processing(operation: str):
    """Convenience decorator for medical processing tracking"""
    return monitor.track_operation(f"medical_{operation}")

def track_performance(func):
    """Decorator to track function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        processing_time = time.time() - start_time
        
        monitor.log_event("performance", {
            "function": func.__name__,
            "processing_time": processing_time
        })
        
        return result
    return wrapper

# Make available for import
__all__ = ["FhirFlameMonitor", "monitor", "track_medical_processing", "track_performance"]