"""
FhirFlame Workflow Orchestrator
Model-agnostic orchestrator that respects user preferences for OCR and LLM models
"""

import asyncio
import time
import os
from typing import Dict, Any, Optional, Union
from .file_processor import local_processor
from .codellama_processor import CodeLlamaProcessor
from .monitoring import monitor


class WorkflowOrchestrator:
    """Model-agnostic workflow orchestrator for medical document processing"""
    
    def __init__(self):
        self.local_processor = local_processor
        self.codellama_processor = CodeLlamaProcessor()
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        
        # Available models configuration
        self.available_models = {
            "codellama": {
                "processor": self.codellama_processor,
                "name": "CodeLlama 13B-Instruct",
                "available": True
            },
            "huggingface": {
                "processor": self.codellama_processor,  # Will be enhanced processor in app.py
                "name": "HuggingFace API",
                "available": True
            },
            "nlp_basic": {
                "processor": self.codellama_processor,  # Basic fallback
                "name": "NLP Basic Processing",
                "available": True
            }
            # Future models can be added here
        }
        
        self.available_ocr_methods = {
            "mistral": {
                "name": "Mistral OCR API",
                "available": bool(self.mistral_api_key),
                "requires_api": True
            },
            "local": {
                "name": "Local OCR Processor", 
                "available": True,
                "requires_api": False
            }
        }
        
    @monitor.track_operation("complete_document_workflow")
    async def process_complete_workflow(
        self,
        document_bytes: Optional[bytes] = None,
        medical_text: Optional[str] = None,
        user_id: str = "workflow-user",
        filename: str = "medical_document",
        document_type: str = "clinical_note",
        use_mistral_ocr: bool = None,
        use_advanced_llm: bool = True,
        llm_model: str = "codellama",
        generate_fhir: bool = True
    ) -> Dict[str, Any]:
        """
        Complete workflow: Document → OCR → Entity Extraction → FHIR Generation
        
        Args:
            document_bytes: Document content as bytes
            medical_text: Direct text input (alternative to document_bytes)
            user_id: User identifier for tracking
            filename: Original filename for metadata
            document_type: Type of medical document
            use_mistral_ocr: Whether to use Mistral OCR API vs local OCR
            use_advanced_llm: Whether to use advanced LLM processing
            llm_model: Which LLM model to use (currently supports 'codellama')
            generate_fhir: Whether to generate FHIR bundles
        """
        
        workflow_start = time.time()
        extracted_text = None
        ocr_method_used = None
        llm_processing_result = None
        
        # Stage 1: Text Extraction
        if document_bytes:
            ocr_start_time = time.time()
            
            # Auto-select Mistral if available and not explicitly disabled
            if use_mistral_ocr is None:
                use_mistral_ocr = bool(self.mistral_api_key)
            
            # Choose OCR method based on user preference and availability
            if use_mistral_ocr and self.mistral_api_key:
                
                monitor.log_event("workflow_stage_start", {
                    "stage": "mistral_ocr_extraction",
                    "document_size": len(document_bytes),
                    "filename": filename
                })
                
                # Use Mistral OCR for text extraction
                extracted_text = await self.local_processor._extract_with_mistral(document_bytes)
                ocr_processing_time = time.time() - ocr_start_time
                ocr_method_used = "mistral_api"
                
                
                # Log Mistral OCR processing
                monitor.log_mistral_ocr_processing(
                    document_size=len(document_bytes),
                    extraction_time=ocr_processing_time,
                    success=True,
                    text_length=len(extracted_text)
                )
                    
            else:
                # Use local processor
                result = await self.local_processor.process_document(
                    document_bytes, user_id, filename
                )
                extracted_text = result.get('extracted_text', '')
                ocr_method_used = "local_processor"
                
                
        elif medical_text:
            # Direct text input
            extracted_text = medical_text
            ocr_method_used = "direct_input"
            
            
        else:
            raise ValueError("Either document_bytes or medical_text must be provided")
        
        # Stage 2: Medical Entity Extraction
        if use_advanced_llm and llm_model in self.available_models:
            model_config = self.available_models[llm_model]
            
            if model_config["available"]:
                monitor.log_event("workflow_stage_start", {
                    "stage": "llm_entity_extraction",
                    "model": llm_model,
                    "text_length": len(extracted_text),
                    "ocr_method": ocr_method_used
                })
                
                # Prepare source metadata
                source_metadata = {
                    "extraction_method": ocr_method_used,
                    "original_filename": filename,
                    "document_size": len(document_bytes) if document_bytes else None,
                    "workflow_stage": "post_ocr_extraction" if document_bytes else "direct_text_input",
                    "llm_model": llm_model
                }
                
                # DEBUG: before entity extraction call
                monitor.log_event("entity_extraction_pre_call", {
                    "provider": llm_model,
                    "text_snippet": extracted_text[:100]
                })
                
                
                llm_processing_result = await model_config["processor"].process_document(
                    medical_text=extracted_text,
                    document_type=document_type,
                    extract_entities=True,
                    generate_fhir=generate_fhir,
                    source_metadata=source_metadata
                )
                
                
                # DEBUG: after entity extraction call
                monitor.log_event("entity_extraction_post_call", {
                    "provider": llm_model,
                    "extraction_results": llm_processing_result.get("extraction_results", {}),
                    "fhir_bundle_present": "fhir_bundle" in llm_processing_result
                })
            else:
                # Model not available, use basic processing
                llm_processing_result = {
                    "extracted_data": '{"error": "Advanced LLM not available"}',
                    "extraction_results": {
                        "entities_found": 0,
                        "quality_score": 0.0
                    },
                    "metadata": {
                        "model_used": "none",
                        "processing_time": 0.0
                    }
                }
        else:
            # Basic text processing without advanced LLM
            llm_processing_result = {
                "extracted_data": f'{{"text_length": {len(extracted_text)}, "processing_mode": "basic"}}',
                "extraction_results": {
                    "entities_found": 0,
                    "quality_score": 0.5
                },
                "metadata": {
                    "model_used": "basic_processor",
                    "processing_time": 0.1
                }
            }
        
        # Stage 3: FHIR Validation (if FHIR bundle was generated)
        fhir_validation_result = None
        if generate_fhir and llm_processing_result.get('fhir_bundle'):
            from .fhir_validator import FhirValidator
            validator = FhirValidator()
            
            monitor.log_event("workflow_stage_start", {
                "stage": "fhir_validation",
                "bundle_generated": True
            })
            
            fhir_validation_result = validator.validate_fhir_bundle(llm_processing_result['fhir_bundle'])
            
            monitor.log_event("fhir_validation_complete", {
                "is_valid": fhir_validation_result['is_valid'],
                "compliance_score": fhir_validation_result['compliance_score'],
                "validation_level": fhir_validation_result['validation_level']
            })
        
        # Stage 4: Workflow Results Assembly
        workflow_time = time.time() - workflow_start
        
        # Determine completed stages
        stages_completed = ["text_extraction"]
        if use_advanced_llm:
            stages_completed.append("entity_extraction")
        if generate_fhir:
            stages_completed.append("fhir_generation")
        if fhir_validation_result:
            stages_completed.append("fhir_validation")
 
        integrated_result = {
            "workflow_metadata": {
                "total_processing_time": workflow_time,
                "mistral_ocr_used": ocr_method_used == "mistral_api",
                "ocr_method": ocr_method_used,
                "llm_model": llm_model if use_advanced_llm else "none",
                "advanced_llm_used": use_advanced_llm,
                "fhir_generated": generate_fhir,
                "stages_completed": stages_completed,
                "user_id": user_id,
                "filename": filename,
                "document_type": document_type
            },
            "text_extraction": {
                "extracted_text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                "full_text_length": len(extracted_text),
                "extraction_method": ocr_method_used
            },
            "medical_analysis": {
                "entities_found": llm_processing_result["extraction_results"]["entities_found"],
                "quality_score": llm_processing_result["extraction_results"]["quality_score"],
                "model_used": llm_processing_result["metadata"]["model_used"],
                "extracted_data": llm_processing_result["extracted_data"]
            },
            "fhir_bundle": llm_processing_result.get("fhir_bundle") if generate_fhir else None,
            "fhir_validation": fhir_validation_result,
            "status": "success",
            "processing_mode": "integrated_workflow"
        }
        
        # Log workflow completion
        monitor.log_workflow_summary(
            documents_processed=1,
            successful_documents=1,
            total_time=workflow_time,
            average_time=workflow_time,
            monitoring_active=monitor.langfuse is not None
        )
        
        # Log OCR workflow integration if OCR was used
        if ocr_method_used in ["mistral_api", "local_processor"]:
            monitor.log_ocr_workflow_integration(
                ocr_method=ocr_method_used,
                agent_processing_time=llm_processing_result["metadata"]["processing_time"],
                total_workflow_time=workflow_time,
                entities_found=llm_processing_result["extraction_results"]["entities_found"]
            )
        
        monitor.log_event("complete_workflow_success", {
            "total_time": workflow_time,
            "ocr_method": ocr_method_used,
            "llm_model": llm_model if use_advanced_llm else "none",
            "entities_found": llm_processing_result["extraction_results"]["entities_found"],
            "fhir_generated": generate_fhir and "fhir_bundle" in llm_processing_result,
            "processing_pipeline": f"{ocr_method_used} → {llm_model if use_advanced_llm else 'basic'} → {'fhir' if generate_fhir else 'no-fhir'}"
        })
        
        return integrated_result
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow configuration and available models"""
        monitoring_status = monitor.get_monitoring_status()
        
        return {
            "available_ocr_methods": self.available_ocr_methods,
            "available_llm_models": self.available_models,
            "mistral_api_key_configured": bool(self.mistral_api_key),
            "monitoring_enabled": monitoring_status["langfuse_enabled"],
            "monitoring_status": monitoring_status,
            "default_configuration": {
                "ocr_method": "mistral" if self.mistral_api_key else "local",
                "llm_model": "codellama",
                "generate_fhir": True
            }
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models for UI dropdowns"""
        return {
            "ocr_methods": [
                {"value": "mistral", "label": "Mistral OCR API", "available": bool(self.mistral_api_key)},
                {"value": "local", "label": "Local OCR Processor", "available": True}
            ],
            "llm_models": [
                {"value": "codellama", "label": "CodeLlama 13B-Instruct", "available": True},
                {"value": "basic", "label": "Basic Text Processing", "available": True}
            ]
        }

# Global workflow orchestrator instance
workflow_orchestrator = WorkflowOrchestrator()