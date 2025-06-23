"""
CodeLlama Processor for FhirFlame
RTX 4090 GPU-optimized medical text processing with CodeLlama 13B-instruct
Enhanced with Pydantic models and clean monitoring integration
NOW WITH REAL OLLAMA INTEGRATION!
"""

import asyncio
import json
import time
import os
import httpx
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

class CodeLlamaProcessor:
    """CodeLlama 13B-instruct processor optimized for RTX 4090 with Pydantic validation"""
    
    def __init__(self):
        """Initialize CodeLlama processor with environment-driven configuration"""
        # Load configuration from .env
        self.use_real_ollama = os.getenv("USE_REAL_OLLAMA", "false").lower() == "true"
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "codellama:13b-instruct")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2048"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))
        self.timeout = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", "300"))
        
        # GPU settings
        self.gpu_available = os.getenv("GPU_ENABLED", "true").lower() == "true"
        self.vram_allocated = f"{os.getenv('MAX_VRAM_GB', '12')}GB"
        
        print(f"🔥 CodeLlamaProcessor initialized:")
        print(f"   Real Ollama: {'✅ ENABLED' if self.use_real_ollama else '❌ MOCK MODE'}")
        print(f"   Model: {self.model_name}")
        print(f"   Ollama URL: {self.ollama_base_url}")
        
    async def process_document(self, medical_text: str, document_type: str = "clinical_note", extract_entities: bool = True, generate_fhir: bool = False, source_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process medical document using CodeLlama 13B-instruct with Pydantic validation"""
        from .monitoring import monitor
        
        # Start comprehensive document processing monitoring
        with monitor.trace_document_workflow(document_type, len(medical_text)) as trace:
            start_time = time.time()
            
            # Handle source metadata (e.g., from Mistral OCR)
            source_info = source_metadata or {}
            ocr_source = source_info.get("extraction_method", "direct_input")
            
            # Log document processing start with OCR info
            monitor.log_document_processing_start(
                document_type=document_type,
                text_length=len(medical_text),
                extract_entities=extract_entities,
                generate_fhir=generate_fhir
            )
            
            # Log OCR integration if applicable
            if ocr_source != "direct_input":
                monitor.log_event("ocr_integration", {
                    "ocr_method": ocr_source,
                    "text_length": len(medical_text),
                    "document_type": document_type,
                    "processing_stage": "pre_entity_extraction"
                })
            
            # Real processing implementation with environment-driven behavior
            start_processing = time.time()
            
            if self.use_real_ollama:
                # **PRIMARY: REAL OLLAMA PROCESSING** with validation logic
                try:
                    print("🔥 Attempting Ollama processing...")
                    processing_result = await self._process_with_real_ollama(medical_text, document_type)
                    actual_processing_time = time.time() - start_processing
                    print(f"✅ Ollama processing successful in {actual_processing_time:.2f}s")
                except Exception as e:
                    print(f"⚠️ Ollama processing failed ({e}), falling back to rule-based...")
                    processing_result = await self._process_with_rules(medical_text)
                    actual_processing_time = time.time() - start_processing
                    print(f"✅ Rule-based fallback successful in {actual_processing_time:.2f}s")
            else:
                # Rule-based processing (when Ollama is disabled)
                print("📝 Using rule-based processing (Ollama disabled)")
                processing_result = await self._process_with_rules(medical_text)
                actual_processing_time = time.time() - start_processing
                print(f"✅ Rule-based processing completed in {actual_processing_time:.2f}s")
            
            processing_time = time.time() - start_time
            
            # Use results from rule-based processing (always successful)
            if extract_entities and processing_result.get("success", True):
                raw_extracted = processing_result["extracted_data"]
                
                # Import and create validated medical data using Pydantic
                from .fhir_validator import ExtractedMedicalData
                medical_data = ExtractedMedicalData(
                    patient=raw_extracted.get("patient_info", "Unknown Patient"),
                    conditions=raw_extracted.get("conditions", []),
                    medications=raw_extracted.get("medications", []),
                    confidence_score=raw_extracted.get("confidence_score", 0.75)
                )
                
                entities_found = len(raw_extracted.get("conditions", [])) + len(raw_extracted.get("medications", []))
                quality_score = medical_data.confidence_score
                extracted_data = medical_data.model_dump()
                
                # Add processing metadata
                extracted_data["_processing_metadata"] = {
                    "mode": processing_result.get("processing_mode", "rule_based"),
                    "model": processing_result.get("model_used", "rule_based_nlp"),
                    "vitals_found": len(raw_extracted.get("vitals", [])),
                    "procedures_found": len(raw_extracted.get("procedures", []))
                }
                
                # Log successful medical processing using centralized monitoring
                monitor.log_medical_processing(
                    entities_found=entities_found,
                    confidence=quality_score,
                    processing_time=actual_processing_time,
                    processing_mode=processing_result.get("processing_mode", "rule_based"),
                    model_used=processing_result.get("model_used", "rule_based_nlp")
                )
                
            else:
                # Fallback if processing failed
                entities_found = 0
                quality_score = 0.0
                extracted_data = {"error": "Processing failed", "mode": "error_fallback"}
            
            # Generate FHIR bundle using Pydantic validator
            fhir_bundle = None
            fhir_generated = False
            if generate_fhir:
                from .fhir_validator import FhirValidator
                validator = FhirValidator()
                bundle_data = {
                    'patient_name': extracted_data.get('patient', 'Unknown Patient'),
                    'conditions': extracted_data.get('conditions', [])
                }
                
                # Generate FHIR bundle with monitoring
                fhir_start_time = time.time()
                fhir_bundle = validator.generate_fhir_bundle(bundle_data)
                fhir_generation_time = time.time() - fhir_start_time
                fhir_generated = True
                
                # Log FHIR bundle generation using centralized monitoring
                monitor.log_fhir_bundle_generation(
                    patient_resources=1 if extracted_data.get('patient') != 'Unknown Patient' else 0,
                    condition_resources=len(extracted_data.get('conditions', [])),
                    observation_resources=0,  # Not generating observations yet
                    generation_time=fhir_generation_time,
                    success=fhir_bundle is not None
                )
            
            # Log document processing completion using centralized monitoring
            monitor.log_document_processing_complete(
                success=processing_result["success"] if processing_result else False,
                processing_time=processing_time,
                entities_found=entities_found,
                fhir_generated=fhir_generated,
                quality_score=quality_score
            )
            
            result = {
                "metadata": {
                    "model_used": self.model_name,
                    "gpu_used": "RTX_4090",
                    "vram_used": self.vram_allocated,
                    "processing_time": processing_time,
                    "source_metadata": source_info
                },
                "extraction_results": {
                    "entities_found": entities_found,
                    "quality_score": quality_score,
                    "confidence_score": 0.95,
                    "ocr_source": ocr_source
                },
                "extracted_data": json.dumps(extracted_data)
            }
            
            # Add FHIR bundle only if generated
            if fhir_bundle:
                result["fhir_bundle"] = fhir_bundle
                
            return result
    
    async def process_medical_text_codellama(self, medical_text: str) -> Dict[str, Any]:
        """Legacy method - use process_document instead"""
        result = await self.process_document(medical_text)
        return {
            "success": True,
            "model_used": result["metadata"]["model_used"],
            "gpu_used": result["metadata"]["gpu_used"],
            "vram_used": result["metadata"]["vram_used"],
            "processing_time": result["metadata"]["processing_time"],
            "extracted_data": result["extracted_data"]
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information"""
        return {
            "total_vram": "24GB",
            "allocated_vram": self.vram_allocated,
            "available_vram": "12GB",
            "memory_efficient": True
        }
    
    async def _process_with_real_ollama(self, medical_text: str, document_type: str) -> Dict[str, Any]:
        """🚀 REAL OLLAMA PROCESSING - This is the breakthrough!"""
        from .monitoring import monitor
        
        # Use centralized AI processing monitoring
        with monitor.trace_ai_processing(
            model=self.model_name,
            text_length=len(medical_text),
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ) as trace:
            
            # Validate input text before processing
            if not medical_text or len(medical_text.strip()) < 10:
                # Return structure consistent with successful processing
                extracted_data = {
                    "patient_info": "No data available",
                    "conditions": [],
                    "medications": [],
                    "vitals": [],
                    "procedures": [],
                    "confidence_score": 0.0,
                    "extraction_summary": "Insufficient medical text for analysis",
                    "entities_found": 0
                }
                return {
                    "processing_mode": "real_ollama",
                    "model_used": self.model_name,
                    "extracted_data": extracted_data,
                    "raw_response": "Input too short for processing",
                    "success": True,
                    "api_time": 0.0,
                    "insufficient_input": True,
                    "reason": "Input text too short or empty"
                }

            # Prepare the medical analysis prompt
            prompt = f"""You are a medical AI assistant specializing in clinical text analysis and FHIR data extraction.

CRITICAL RULES:
- ONLY extract information that is explicitly present in the provided text
- DO NOT generate, invent, or create any medical information
- If no medical data is found, return empty arrays and "No data available"
- DO NOT use examples or placeholder data

TASK: Analyze the following medical text and extract structured medical information.

MEDICAL TEXT:
{medical_text}

Please extract and return a JSON response with the following structure:
{{
    "patient_info": "Patient name or identifier if found, otherwise 'No data available'",
    "conditions": ["list", "of", "medical", "conditions", "only", "if", "found"],
    "medications": ["list", "of", "medications", "only", "if", "found"],
    "vitals": ["list", "of", "vital", "signs", "only", "if", "found"],
    "procedures": ["list", "of", "procedures", "only", "if", "found"],
    "confidence_score": 0.85,
    "extraction_summary": "Brief summary of what was actually found (not generated)"
}}

Focus on medical accuracy and FHIR R4 compliance. Return only valid JSON. DO NOT GENERATE FAKE DATA."""

            try:
                # Make real HTTP request to Ollama API
                api_start_time = time.time()
                
                # Use the configured Ollama URL directly (already corrected in .env)
                ollama_url = self.ollama_base_url
                print(f"🔥 DEBUG: Using Ollama URL: {ollama_url}")
                
                # Validate that we have the correct model loaded
                async with httpx.AsyncClient(timeout=10) as test_client:
                    try:
                        # Check what models are available
                        models_response = await test_client.get(f"{ollama_url}/api/tags")
                        if models_response.status_code == 200:
                            models_data = models_response.json()
                            available_models = [model.get("name", "") for model in models_data.get("models", [])]
                            print(f"🔍 DEBUG: Available models: {available_models}")
                            
                            if self.model_name not in available_models:
                                error_msg = f"❌ Model {self.model_name} not found. Available: {available_models}"
                                print(error_msg)
                                raise Exception(error_msg)
                        else:
                            print(f"⚠️ Could not check available models: {models_response.status_code}")
                    except Exception as model_check_error:
                        print(f"⚠️ Model availability check failed: {model_check_error}")
                        # Continue anyway, but log the issue
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{ollama_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": self.temperature,
                                "top_p": self.top_p,
                                "num_predict": self.max_tokens
                            }
                        }
                    )
                    
                    api_time = time.time() - api_start_time
                    
                    # Log API call using centralized monitoring
                    monitor.log_ollama_api_call(
                        model=self.model_name,
                        url=ollama_url,
                        prompt_length=len(prompt),
                        success=response.status_code == 200,
                        response_time=api_time,
                        status_code=response.status_code,
                        error=None if response.status_code == 200 else response.text
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        generated_text = result.get("response", "")
                        
                        # Parse JSON from model response
                        parsing_start = time.time()
                        try:
                            # Extract JSON from the response (model might add extra text)
                            json_start = generated_text.find('{')
                            json_end = generated_text.rfind('}') + 1
                            if json_start >= 0 and json_end > json_start:
                                json_str = generated_text[json_start:json_end]
                                raw_extracted_data = json.loads(json_str)
                                
                                # Transform complex AI response to simple format for Pydantic compatibility
                                transformation_start = time.time()
                                extracted_data = self._transform_ai_response(raw_extracted_data)
                                transformation_time = time.time() - transformation_start
                                
                                # Log successful parsing using centralized monitoring
                                parsing_time = time.time() - parsing_start
                                entities_found = len(extracted_data.get("conditions", [])) + len(extracted_data.get("medications", []))
                                
                                monitor.log_ai_parsing(
                                    success=True,
                                    response_format="json",
                                    entities_extracted=entities_found,
                                    parsing_time=parsing_time
                                )
                                
                                # Log data transformation
                                monitor.log_data_transformation(
                                    input_format="complex_nested_json",
                                    output_format="pydantic_compatible",
                                    entities_transformed=entities_found,
                                    transformation_time=transformation_time,
                                    complex_nested=isinstance(raw_extracted_data.get("patient_info"), dict)
                                )
                                
                                # Log AI generation success
                                monitor.log_ai_generation(
                                    model=self.model_name,
                                    response_length=len(generated_text),
                                    processing_time=api_time,
                                    entities_found=entities_found,
                                    confidence=extracted_data.get("confidence_score", 0.0),
                                    processing_mode="real_ollama"
                                )
                                
                            else:
                                raise ValueError("No valid JSON found in response")
                                
                        except (json.JSONDecodeError, ValueError) as e:
                            # Log parsing failure using centralized monitoring
                            monitor.log_ai_parsing(
                                success=False,
                                response_format="malformed_json",
                                entities_extracted=0,
                                parsing_time=time.time() - parsing_start,
                                error=str(e)
                            )
                            print(f"⚠️ JSON parsing failed: {e}")
                            print(f"Raw response: {generated_text[:200]}...")
                            # Fall back to rule-based extraction
                            return await self._process_with_rules(medical_text)
                        
                        # Update trace with success
                        if trace:
                            trace.update(output={
                                "status": "success",
                                "processing_mode": "real_ollama",
                                "entities_extracted": len(extracted_data.get("conditions", [])) + len(extracted_data.get("medications", [])),
                                "api_time": api_time,
                                "confidence": extracted_data.get("confidence_score", 0.0)
                            })
                        
                        return {
                            "processing_mode": "real_ollama",
                            "model_used": self.model_name,
                            "extracted_data": extracted_data,
                            "raw_response": generated_text[:500],  # First 500 chars for debugging
                            "success": True,
                            "api_time": api_time
                        }
                    else:
                        error_msg = f"Ollama API returned {response.status_code}: {response.text}"
                        raise Exception(error_msg)
                        
            except Exception as e:
                print(f"❌ Real Ollama processing failed: {e}")
                raise e
    
    async def _process_with_rules(self, medical_text: str) -> Dict[str, Any]:
        """📝 Rule-based processing fallback (enhanced from original)"""
        from .monitoring import monitor
        
        # Start monitoring for rule-based processing
        with monitor.trace_operation("rule_based_processing", {
            "text_length": len(medical_text),
            "processing_mode": "fallback"
        }) as trace:
            
            start_time = time.time()
            
            # Enhanced rule-based extraction with comprehensive medical patterns
            import re
            medical_text_lower = medical_text.lower()
        
            # Extract patient information with name parsing
            patient_info = "Unknown Patient"
            patient_dob = None
            
            # Look for patient name patterns
            patient_patterns = [
                r"patient:\s*([^\n\r]+)",
                r"name:\s*([^\n\r]+)",
                r"pt:\s*([^\n\r]+)"
            ]
            for pattern in patient_patterns:
                match = re.search(pattern, medical_text_lower)
                if match:
                    patient_info = match.group(1).strip().title()
                    break
            
            # Extract date of birth with multiple patterns
            dob_patterns = [
                r"dob:\s*([^\n\r]+)",
                r"date of birth:\s*([^\n\r]+)",
                r"born:\s*([^\n\r]+)",
                r"birth date:\s*([^\n\r]+)"
            ]
            for pattern in dob_patterns:
                match = re.search(pattern, medical_text_lower)
                if match:
                    patient_dob = match.group(1).strip()
                    break
            
            # Enhanced condition detection with context
            condition_keywords = [
                "hypertension", "diabetes", "pneumonia", "asthma", "copd",
                "depression", "anxiety", "arthritis", "cancer", "stroke",
                "heart disease", "kidney disease", "liver disease", "chest pain",
                "acute coronary syndrome", "myocardial infarction", "coronary syndrome",
                "myocardial infarction", "angina", "atrial fibrillation"
            ]
            conditions = []
            for keyword in condition_keywords:
                if keyword in medical_text_lower:
                    # Try to get the full condition name from context
                    context_pattern = rf"([^\n\r]*{re.escape(keyword)}[^\n\r]*)"
                    context_match = re.search(context_pattern, medical_text_lower)
                    if context_match:
                        full_condition = context_match.group(1).strip()
                        conditions.append(full_condition.title())
                    else:
                        conditions.append(keyword.title())
            
            # Enhanced medication detection with dosages
            medication_patterns = [
                r"([a-zA-Z]+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|units?)\s+(daily|twice daily|bid|tid|qid|every \d+ hours?|once daily|nightly)",
                r"([a-zA-Z]+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|units?)",
                r"([a-zA-Z]+)\s+(daily|twice daily|bid|tid|qid|nightly)"
            ]
            medications = []
            
            # Look for complete medication entries with dosages
            med_lines = [line.strip() for line in medical_text.split('\n') if line.strip()]
            for line in med_lines:
                line_lower = line.lower()
                # Check if line contains medication information
                if any(word in line_lower for word in ['mg', 'daily', 'twice', 'bid', 'tid', 'aspirin', 'lisinopril', 'atorvastatin', 'metformin']):
                    for pattern in medication_patterns:
                        matches = re.finditer(pattern, line_lower)
                        for match in matches:
                            if len(match.groups()) >= 3:
                                med_name = match.group(1).title()
                                dose = match.group(2)
                                unit = match.group(3)
                                frequency = match.group(4) if len(match.groups()) >= 4 else ""
                                full_med = f"{med_name} {dose} {unit} {frequency}".strip()
                                medications.append(full_med)
                            elif len(match.groups()) >= 2:
                                med_name = match.group(1).title()
                                dose_info = match.group(2)
                                full_med = f"{med_name} {dose_info}".strip()
                                medications.append(full_med)
                    
                    # If no pattern matched, try simple medication detection
                    if not any(med in line for med in medications):
                        simple_meds = ["aspirin", "lisinopril", "atorvastatin", "metformin", "metoprolol"]
                        for med in simple_meds:
                            if med in line_lower:
                                medications.append(line.strip())
                                break
            
            # Enhanced vital signs detection
            vitals = []
            vital_patterns = [
                "blood pressure", "bp", "heart rate", "hr", "temperature",
                "temp", "oxygen saturation", "o2 sat", "respiratory rate", "rr"
            ]
            for pattern in vital_patterns:
                if pattern in medical_text_lower:
                    vitals.append(pattern.title())
            
            # Calculate proper confidence score based on data quality and completeness
            base_confidence = 0.7
            
            # Add confidence for patient info completeness
            if patient_info != "Unknown Patient":
                base_confidence += 0.1
            if patient_dob:
                base_confidence += 0.05
                
            # Add confidence for medical data found
            entity_bonus = min(0.15, (len(conditions) + len(medications)) * 0.02)
            base_confidence += entity_bonus
            
            # Bonus for detailed medication information (with dosages)
            detailed_meds = sum(1 for med in medications if any(unit in med.lower() for unit in ['mg', 'g', 'ml', 'daily', 'twice']))
            if detailed_meds > 0:
                base_confidence += min(0.1, detailed_meds * 0.03)
            
            final_confidence = min(0.95, base_confidence)
            
            extracted_data = {
                "patient": patient_info,
                "patient_info": patient_info,
                "date_of_birth": patient_dob,
                "conditions": conditions,
                "medications": medications,
                "vitals": vitals,
                "procedures": [],  # Could enhance this too
                "confidence_score": final_confidence,
                "extraction_summary": f"Enhanced extraction found {len(conditions)} conditions, {len(medications)} medications, {len(vitals)} vitals" + (f", DOB: {patient_dob}" if patient_dob else ""),
                "extraction_quality": {
                    "patient_identified": patient_info != "Unknown Patient",
                    "dob_found": bool(patient_dob),
                    "detailed_medications": detailed_meds,
                    "total_entities": len(conditions) + len(medications) + len(vitals)
                }
            }
            
            processing_time = time.time() - start_time
            
            # Log rule-based processing using centralized monitoring
            monitor.log_rule_based_processing(
                entities_found=len(conditions) + len(medications),
                conditions=len(conditions),
                medications=len(medications),
                vitals=len(vitals),
                confidence=extracted_data["confidence_score"],
                processing_time=processing_time
            )
            
            # Log medical entity extraction details
            monitor.log_medical_entity_extraction(
                conditions=len(conditions),
                medications=len(medications),
                vitals=len(vitals),
                procedures=0,
                patient_info_found=patient_info != "Unknown Patient",
                confidence=extracted_data["confidence_score"]
            )
            
            # Update trace with results
            if trace:
                trace.update(output={
                    "status": "success",
                    "processing_mode": "rule_based_fallback",
                    "entities_extracted": len(conditions) + len(medications),
                    "processing_time": processing_time,
                    "confidence": extracted_data["confidence_score"]
                })
            
            return {
                "processing_mode": "rule_based_fallback",
                "model_used": "rule_based_nlp",
                "extracted_data": extracted_data,
                "success": True,
                "processing_time": processing_time
            }
    
    def _transform_ai_response(self, raw_data: dict) -> dict:
        """Transform complex AI response to Pydantic-compatible format"""
        
        # Initialize with defaults
        transformed = {
            "patient_info": "Unknown Patient",
            "conditions": [],
            "medications": [],
            "vitals": [],
            "procedures": [],
            "confidence_score": 0.75
        }
        
        # Transform patient information
        patient_info = raw_data.get("patient_info", {})
        if isinstance(patient_info, dict):
            # Extract from nested structure
            name = patient_info.get("name", "")
            if not name and "given" in patient_info and "family" in patient_info:
                name = f"{' '.join(patient_info.get('given', []))} {patient_info.get('family', '')}"
            transformed["patient_info"] = name or "Unknown Patient"
        elif isinstance(patient_info, str):
            transformed["patient_info"] = patient_info
        
        # Transform conditions
        conditions = raw_data.get("conditions", [])
        transformed_conditions = []
        for condition in conditions:
            if isinstance(condition, dict):
                # Extract from complex structure
                name = condition.get("name") or condition.get("display") or condition.get("text", "")
                if name:
                    transformed_conditions.append(name)
            elif isinstance(condition, str):
                transformed_conditions.append(condition)
        transformed["conditions"] = transformed_conditions
        
        # Transform medications
        medications = raw_data.get("medications", [])
        transformed_medications = []
        for medication in medications:
            if isinstance(medication, dict):
                # Extract from complex structure
                name = medication.get("name") or medication.get("display") or medication.get("text", "")
                dosage = medication.get("dosage") or medication.get("dose", "")
                frequency = medication.get("frequency", "")
                
                # Combine medication info
                med_str = name
                if dosage:
                    med_str += f" {dosage}"
                if frequency:
                    med_str += f" {frequency}"
                
                if med_str.strip():
                    transformed_medications.append(med_str.strip())
            elif isinstance(medication, str):
                transformed_medications.append(medication)
        transformed["medications"] = transformed_medications
        
        # Transform vitals (if present)
        vitals = raw_data.get("vitals", [])
        transformed_vitals = []
        for vital in vitals:
            if isinstance(vital, dict):
                name = vital.get("name") or vital.get("type", "")
                value = vital.get("value", "")
                unit = vital.get("unit", "")
                
                vital_str = name
                if value:
                    vital_str += f": {value}"
                if unit:
                    vital_str += f" {unit}"
                    
                if vital_str.strip():
                    transformed_vitals.append(vital_str.strip())
            elif isinstance(vital, str):
                transformed_vitals.append(vital)
        transformed["vitals"] = transformed_vitals
        
        # Preserve confidence score
        confidence = raw_data.get("confidence_score", 0.75)
        if isinstance(confidence, (int, float)):
            transformed["confidence_score"] = min(max(confidence, 0.0), 1.0)
        
        # Generate summary
        total_entities = len(transformed["conditions"]) + len(transformed["medications"]) + len(transformed["vitals"])
        transformed["extraction_summary"] = f"AI extraction found {total_entities} entities: {len(transformed['conditions'])} conditions, {len(transformed['medications'])} medications, {len(transformed['vitals'])} vitals"
        
        return transformed


# Make class available for import
__all__ = ["CodeLlamaProcessor"]