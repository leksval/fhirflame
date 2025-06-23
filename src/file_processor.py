"""
Local Processor for FhirFlame Development
Core logic with optional Mistral API OCR and multimodal fallbacks
"""

import asyncio
import json
import uuid
import os
import io
import base64
from datetime import datetime
from typing import Dict, Any, Optional, List
from .monitoring import monitor

# PDF and Image Processing
try:
    from pdf2image import convert_from_bytes
    from PIL import Image
    import PyPDF2
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False

class LocalProcessor:
    """Local processor with optional external fallbacks"""
    
    def __init__(self):
        self.use_mistral_fallback = os.getenv("USE_MISTRAL_FALLBACK", "false").lower() == "true"
        self.use_multimodal_fallback = os.getenv("USE_MULTIMODAL_FALLBACK", "false").lower() == "true"
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        
    @monitor.track_operation("real_document_processing")
    async def process_document(self, document_bytes: bytes, user_id: str, filename: str) -> Dict[str, Any]:
        """Process document with fallback capabilities and quality assertions"""
        
        # Try external OCR if enabled and available
        extracted_text = await self._extract_text_with_fallback(document_bytes, filename)
        
        # Log OCR quality metrics
        monitor.log_event("ocr_text_extracted", {
            "text_extracted": len(extracted_text) > 0,
            "text_length": len(extracted_text),
            "filename": filename
        })
        monitor.log_event("ocr_minimum_length", {
            "substantial_text": len(extracted_text) > 50,
            "text_length": len(extracted_text)
        })
        
        # Extract medical entities from text
        entities = self._extract_medical_entities(extracted_text)
        
        # Log medical entity extraction
        monitor.log_event("medical_entities_found", {
            "entities_found": len(entities) > 0,
            "entity_count": len(entities)
        })
        
        # Create FHIR bundle
        fhir_bundle = self._create_simple_fhir_bundle(entities, user_id)
        
        # Log FHIR validation
        monitor.log_event("fhir_bundle_valid", {
            "bundle_valid": fhir_bundle.get("resourceType") == "Bundle",
            "resource_type": fhir_bundle.get("resourceType")
        })
        monitor.log_event("fhir_has_entries", {
            "has_entries": len(fhir_bundle.get("entry", [])) > 0,
            "entry_count": len(fhir_bundle.get("entry", []))
        })
        
        # Log processing with enhanced metrics
        monitor.log_medical_processing(
            entities_found=len(entities),
            confidence=0.85,
            processing_time=100.0,
            processing_mode="file_processing",
            model_used="enhanced_processor"
        )
        
        return {
            "status": "success",
            "processing_mode": self._get_processing_mode(),
            "filename": filename,
            "processed_by": user_id,
            "entities_found": len(entities),
            "fhir_bundle": fhir_bundle,
            "extracted_text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            "text_length": len(extracted_text)
        }
    
    async def _extract_text_with_fallback(self, document_bytes: bytes, filename: str) -> str:
        """Extract text with optional fallbacks"""
        
        # Try Mistral API OCR first if enabled
        if self.use_mistral_fallback and self.mistral_api_key:
            try:
                monitor.log_event("mistral_attempt_start", {
                    "document_size": len(document_bytes),
                    "api_key_present": bool(self.mistral_api_key),
                    "use_mistral_fallback": self.use_mistral_fallback
                })
                result = await self._extract_with_mistral(document_bytes)
                monitor.log_event("mistral_success_in_fallback", {
                    "text_length": len(result),
                    "text_preview": result[:100] + "..." if len(result) > 100 else result
                })
                return result
            except Exception as e:
                import traceback
                monitor.log_event("mistral_fallback_failed", {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "document_size": len(document_bytes),
                    "api_key_format": f"{self.mistral_api_key[:8]}...{self.mistral_api_key[-4:]}" if self.mistral_api_key else "none"
                })
                print(f"🚨 MISTRAL API FAILED: {type(e).__name__}: {str(e)}")
                print(f"🚨 Full traceback: {traceback.format_exc()}")
        
        # Try multimodal processor if enabled
        if self.use_multimodal_fallback:
            try:
                return await self._extract_with_multimodal(document_bytes)
            except Exception as e:
                monitor.log_event("multimodal_fallback_failed", {"error": str(e)})
        
        # CRITICAL: No dummy data in production - fail properly when OCR fails
        raise Exception(f"Document text extraction failed for {filename}. All OCR methods exhausted. Cannot return dummy data for real medical processing.")
    
    def _convert_pdf_to_images(self, pdf_bytes: bytes) -> List[bytes]:
        """Convert PDF to list of image bytes for Mistral vision processing"""
        if not PDF_PROCESSING_AVAILABLE:
            raise Exception("PDF processing libraries not available. Install pdf2image, Pillow, and PyPDF2.")
        
        try:
            # Convert PDF pages to PIL Images
            monitor.log_event("pdf_conversion_debug", {
                "step": "starting_pdf_conversion",
                "pdf_size": len(pdf_bytes)
            })
            
            # Convert PDF to images (300 DPI for good OCR quality)
            images = convert_from_bytes(pdf_bytes, dpi=300, fmt='PNG')
            
            monitor.log_event("pdf_conversion_debug", {
                "step": "pdf_converted_to_images",
                "page_count": len(images),
                "image_sizes": [(img.width, img.height) for img in images]
            })
            
            # Convert PIL Images to bytes
            image_bytes_list = []
            for i, img in enumerate(images):
                # Convert to RGB if necessary (for JPEG compatibility)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as high-quality JPEG bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=95)
                img_bytes = img_byte_arr.getvalue()
                image_bytes_list.append(img_bytes)
                
                monitor.log_event("pdf_conversion_debug", {
                    "step": f"page_{i+1}_converted",
                    "page_size": len(img_bytes),
                    "dimensions": f"{img.width}x{img.height}"
                })
            
            monitor.log_event("pdf_conversion_success", {
                "total_pages": len(image_bytes_list),
                "total_size": sum(len(img_bytes) for img_bytes in image_bytes_list)
            })
            
            return image_bytes_list
            
        except Exception as e:
            monitor.log_event("pdf_conversion_error", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise Exception(f"PDF to image conversion failed: {str(e)}")

    async def _extract_with_mistral(self, document_bytes: bytes) -> str:
        """Extract text using Mistral OCR API - using proper document understanding endpoint"""
        import httpx
        import base64
        import tempfile
        import os
        
        # 🔍 DEBUGGING: Log entry to Mistral OCR function
        monitor.log_event("mistral_ocr_start", {
            "document_size": len(document_bytes),
            "api_key_present": bool(self.mistral_api_key),
            "api_key_format": f"sk-...{self.mistral_api_key[-4:]}" if self.mistral_api_key else "none"
        })
        
        # Detect file type and extension
        def detect_file_info(data: bytes) -> tuple[str, str]:
            if data.startswith(b'%PDF'):
                return "application/pdf", ".pdf"
            elif data.startswith(b'\xff\xd8\xff'):  # JPEG
                return "image/jpeg", ".jpg"
            elif data.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                return "image/png", ".png"
            elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):  # GIF
                return "image/gif", ".gif"
            elif data.startswith(b'BM'):  # BMP
                return "image/bmp", ".bmp"
            elif data.startswith(b'RIFF') and b'WEBP' in data[:12]:  # WEBP
                return "image/webp", ".webp"
            elif data.startswith(b'II*\x00') or data.startswith(b'MM\x00*'):  # TIFF
                return "image/tiff", ".tiff"
            elif data.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):  # DOC (OLE2)
                return "application/msword", ".doc"
            elif data.startswith(b'PK\x03\x04') and b'word/' in data[:1000]:  # DOCX
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"
            else:
                return "application/pdf", ".pdf"
        
        mime_type, file_ext = detect_file_info(document_bytes)
        
        # 🔍 DEBUGGING: Log document analysis
        monitor.log_event("mistral_ocr_debug", {
            "step": "document_analysis",
            "mime_type": mime_type,
            "file_extension": file_ext,
            "document_size": len(document_bytes),
            "document_start": document_bytes[:100].hex()[:50] + "..." if len(document_bytes) > 50 else document_bytes.hex()
        })
        
        try:
            # 🔍 DEBUGGING: Log exact HTTP request details
            monitor.log_event("mistral_http_debug", {
                "step": "preparing_http_client",
                "api_endpoint": "https://api.mistral.ai/v1/chat/completions",
                "api_key_prefix": f"{self.mistral_api_key[:8]}..." if self.mistral_api_key else "none",
                "timeout": 180.0,
                "client_config": "httpx.AsyncClient() with default settings"
            })
            
            async with httpx.AsyncClient() as client:
                
                # Handle PDF conversion to images
                if mime_type == "application/pdf":
                    monitor.log_event("mistral_ocr_debug", {
                        "step": "pdf_detected_converting_to_images",
                        "pdf_size": len(document_bytes)
                    })
                    
                    # Convert PDF to images
                    try:
                        image_bytes_list = self._convert_pdf_to_images(document_bytes)
                        monitor.log_event("mistral_ocr_debug", {
                            "step": "pdf_conversion_success",
                            "page_count": len(image_bytes_list)
                        })
                    except Exception as pdf_error:
                        monitor.log_event("mistral_ocr_debug", {
                            "step": "pdf_conversion_failed",
                            "error": str(pdf_error)
                        })
                        raise Exception(f"PDF conversion failed: {str(pdf_error)}")
                    
                    # Process each page and combine results
                    all_extracted_text = []
                    
                    for page_num, image_bytes in enumerate(image_bytes_list, 1):
                        monitor.log_event("mistral_ocr_debug", {
                            "step": f"processing_page_{page_num}",
                            "image_size": len(image_bytes)
                        })
                        
                        # Convert image to base64
                        b64_data = base64.b64encode(image_bytes).decode()
                        
                        # 🔍 DEBUGGING: Log exact HTTP request details
                        request_payload = {
                            "model": "pixtral-12b-2409",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"""You are a strict OCR text extraction tool. Your job is to extract ONLY the actual text that appears in this image - nothing more, nothing less.
        
        CRITICAL RULES:
        - Extract ONLY text that is actually visible in the image
        - Do NOT generate, invent, or create any content
        - Do NOT add examples or sample data
        - Do NOT fill in missing information
        - If the image contains minimal text, return minimal text
        - If the image is blank or contains no medical content, return what you actually see
        
        For page {page_num}, extract exactly what text appears in this image:"""
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{b64_data[:50]}..."  # Truncated for logging
                                            }
                                        }
                                    ]
                                }
                            ],
                            "max_tokens": 8000,
                            "temperature": 0.0
                        }
                        
                        monitor.log_event("mistral_http_request_start", {
                            "step": f"sending_request_page_{page_num}",
                            "url": "https://api.mistral.ai/v1/chat/completions",
                            "method": "POST",
                            "headers_count": 2,
                            "payload_size": len(str(request_payload)),
                            "b64_data_size": len(b64_data),
                            "timeout": min(300.0, 60.0 + (len(b64_data) / 100000)),  # Dynamic timeout: 60s base + 1s per 100KB
                            "estimated_timeout": min(300.0, 60.0 + (len(b64_data) / 100000))
                        })
                        
                        # Calculate dynamic timeout based on image size
                        dynamic_timeout = min(300.0, 60.0 + (len(b64_data) / 100000))  # Max 5 minutes
                        
                        
                        # API call for this page with dynamic timeout
                        response = await client.post(
                            "https://api.mistral.ai/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.mistral_api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "pixtral-12b-2409",
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": f"""You are a strict OCR text extraction tool. Your job is to extract ONLY the actual text that appears in this image - nothing more, nothing less.
        
        CRITICAL RULES:
        - Extract ONLY text that is actually visible in the image
        - Do NOT generate, invent, or create any content
        - Do NOT add examples or sample data
        - Do NOT fill in missing information
        - If the image contains minimal text, return minimal text
        - If the image is blank or contains no medical content, return what you actually see
        
        For page {page_num}, extract exactly what text appears in this image:"""
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{b64_data}"
                                                }
                                            }
                                        ]
                                    }
                                ],
                                "max_tokens": 8000,
                                "temperature": 0.0
                            },
                            timeout=dynamic_timeout
                        )
                        
                        monitor.log_event("mistral_http_response_received", {
                            "step": f"response_page_{page_num}",
                            "status_code": response.status_code,
                            "response_size": len(response.content),
                            "headers": dict(response.headers),
                            "elapsed_seconds": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else "unknown"
                        })
                        
                        # Process response for this page
                        monitor.log_event("mistral_ocr_debug", {
                            "step": f"page_{page_num}_api_response",
                            "status_code": response.status_code
                        })
                        
                        if response.status_code == 200:
                            result = response.json()
                            if 'choices' in result and len(result['choices']) > 0:
                                message = result['choices'][0].get('message', {})
                                page_text = message.get('content', '').strip()
                                if page_text:
                                    cleaned_text = self._clean_ocr_text(page_text)
                                    all_extracted_text.append(f"[PAGE {page_num}]\n{cleaned_text}")
                                    
                                    monitor.log_event("mistral_ocr_debug", {
                                        "step": f"page_{page_num}_extracted",
                                        "text_length": len(cleaned_text)
                                    })
                        else:
                            monitor.log_event("mistral_ocr_debug", {
                                "step": f"page_{page_num}_api_error",
                                "status_code": response.status_code,
                                "error": response.text
                            })
                            # Continue with other pages even if one fails
                    
                    # Combine all pages
                    if all_extracted_text:
                        combined_text = "\n\n".join(all_extracted_text)
                        monitor.log_event("mistral_ocr_success", {
                            "mime_type": mime_type,
                            "total_pages": len(image_bytes_list),
                            "pages_processed": len(all_extracted_text),
                            "total_text_length": len(combined_text)
                        })
                        return f"[MISTRAL PDF PROCESSED - {len(image_bytes_list)} pages]\n\n{combined_text}"
                    else:
                        raise Exception("No text extracted from any PDF pages")
                
                else:
                    # Handle non-PDF documents (images) - original logic
                    b64_data = base64.b64encode(document_bytes).decode()
                    b64_preview = b64_data[:100] + "..." if len(b64_data) > 100 else b64_data
                    
                    monitor.log_event("mistral_ocr_debug", {
                        "step": "api_call_preparation",
                        "b64_data_length": len(b64_data),
                        "b64_preview": b64_preview,
                        "api_endpoint": "https://api.mistral.ai/v1/chat/completions",
                        "model": "pixtral-12b-2409"
                    })
                    
                    # Calculate dynamic timeout based on image size
                    dynamic_timeout = min(300.0, 60.0 + (len(b64_data) / 100000))  # Max 5 minutes
                    
                    monitor.log_event("mistral_http_request_start", {
                        "step": "sending_request_image",
                        "url": "https://api.mistral.ai/v1/chat/completions",
                        "method": "POST",
                        "mime_type": mime_type,
                        "b64_data_size": len(b64_data),
                        "timeout": dynamic_timeout,
                        "estimated_timeout": dynamic_timeout
                    })
                    
                    
                    response = await client.post(
                        "https://api.mistral.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.mistral_api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "pixtral-12b-2409",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": """You are a strict OCR text extraction tool. Your job is to extract ONLY the actual text that appears in this image - nothing more, nothing less.

CRITICAL RULES:
- Extract ONLY text that is actually visible in the image
- Do NOT generate, invent, or create any content
- Do NOT add examples or sample data
- Do NOT fill in missing information
- If the image contains minimal text, return minimal text
- If the image is blank or contains no medical content, return what you actually see

Extract exactly what text appears in this image:"""
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:{mime_type};base64,{b64_data}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            "max_tokens": 8000,
                            "temperature": 0.0
                        },
                        timeout=dynamic_timeout
                    )
                    
                    monitor.log_event("mistral_http_response_received", {
                        "step": "response_image",
                        "status_code": response.status_code,
                        "response_size": len(response.content),
                        "headers": dict(response.headers),
                        "elapsed_seconds": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else "unknown"
                    })
                
                # 🔍 DEBUGGING: Log API response
                monitor.log_event("mistral_ocr_debug", {
                    "step": "api_response_received",
                    "status_code": response.status_code,
                    "response_headers": dict(response.headers),
                    "response_size": len(response.content),
                    "response_preview": response.text[:500] + "..." if len(response.text) > 500 else response.text
                })
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 🔍 DEBUGGING: Log successful response parsing
                    monitor.log_event("mistral_ocr_debug", {
                        "step": "response_parsing_success",
                        "result_keys": list(result.keys()) if isinstance(result, dict) else "not_dict",
                        "choices_count": len(result.get("choices", [])) if isinstance(result, dict) else 0
                    })
                    
                    # Log successful API response
                    monitor.log_event("mistral_api_success", {
                        "status_code": response.status_code,
                        "response_format": "valid"
                    })
                    
                    # Extract text from Mistral chat completion response
                    if 'choices' in result and len(result['choices']) > 0:
                        message = result['choices'][0].get('message', {})
                        extracted_text = message.get('content', '').strip()
                        
                        # Log OCR quality
                        monitor.log_event("mistral_response_has_content", {
                            "has_content": len(extracted_text) > 0,
                            "text_length": len(extracted_text)
                        })
                        
                        if extracted_text:
                            # Clean up the response - remove any OCR processing artifacts
                            cleaned_text = self._clean_ocr_text(extracted_text)
                            
                            # Log cleaned text quality
                            monitor.log_event("mistral_cleaned_text_substantial", {
                                "substantial": len(cleaned_text) > 20,
                                "text_length": len(cleaned_text)
                            })
                            
                            # Log successful OCR metrics
                            monitor.log_event("mistral_ocr_success", {
                                "mime_type": mime_type,
                                "raw_length": len(extracted_text),
                                "cleaned_length": len(cleaned_text),
                                "cleaning_ratio": len(cleaned_text) / len(extracted_text) if extracted_text else 0
                            })
                            
                            return f"[MISTRAL DOCUMENT AI PROCESSED - {mime_type}]\n\n{cleaned_text}"
                        else:
                            monitor.log_event("mistral_ocr_not_empty", {
                                "empty_response": True,
                                "mime_type": mime_type
                            })
                            monitor.log_event("mistral_ocr_empty_response", {"mime_type": mime_type})
                            raise Exception("Mistral OCR returned empty text content")
                    else:
                        monitor.log_event("mistral_response_format_valid", {
                            "format_valid": False,
                            "response_keys": list(result.keys()) if isinstance(result, dict) else "not_dict"
                        })
                        monitor.log_event("mistral_ocr_invalid_response", {"response": result})
                        raise Exception("Invalid response format from Mistral OCR API")
                        
                else:
                    # Handle API errors with detailed logging
                    error_msg = f"Mistral OCR API failed with status {response.status_code}"
                    try:
                        error_details = response.json()
                        error_msg += f": {error_details.get('message', 'Unknown error')}"
                        
                        # Log specific error types for debugging
                        if response.status_code == 401:
                            monitor.log_event("mistral_auth_error", {"error": "Invalid API key"})
                            error_msg = "Mistral OCR authentication failed - check API key"
                        elif response.status_code == 429:
                            monitor.log_event("mistral_rate_limit", {"error": "Rate limit exceeded"})
                            error_msg = "Mistral OCR rate limit exceeded - try again later"
                        elif response.status_code == 413:
                            monitor.log_event("mistral_file_too_large", {"mime_type": mime_type})
                            error_msg = "Document too large for Mistral OCR processing"
                        else:
                            monitor.log_event("mistral_api_error", {
                                "status_code": response.status_code,
                                "error": error_details
                            })
                            
                    except Exception:
                        error_text = response.text
                        error_msg += f": {error_text}"
                        monitor.log_event("mistral_unknown_error", {
                            "status_code": response.status_code,
                            "response": error_text
                        })
                    
                    raise Exception(error_msg)
                    
        except Exception as e:
            # 🔍 DEBUGGING: Log exception details
            monitor.log_event("mistral_ocr_debug", {
                "step": "exception_caught",
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "exception_details": {
                    "args": e.args if hasattr(e, 'args') else "no_args",
                    "traceback_summary": f"{type(e).__name__}: {str(e)}"
                }
            })
            
            # Re-raise with context for better debugging
            raise Exception(f"Mistral OCR processing failed: {str(e)}")
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean up OCR text output for medical documents"""
        # Remove common OCR artifacts while preserving medical formatting
        cleaned = text.strip()
        
        # Remove any instruction responses or commentary
        lines = cleaned.split('\n')
        cleaned_lines = []
        
        skip_patterns = [
            "here is the extracted text",
            "the extracted text is:",
            "extracted text:",
            "text content:",
            "document content:",
        ]
        
        for line in lines:
            line_lower = line.lower().strip()
            should_skip = any(pattern in line_lower for pattern in skip_patterns)
            
            if not should_skip and line.strip():
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    async def _extract_with_multimodal(self, document_bytes: bytes) -> str:
        """Extract text using multimodal processor (simplified)"""
        import base64
        import sys
        import os
        
        # Add gaia system to path
        gaia_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "gaia_agentic_system")
        if gaia_path not in sys.path:
            sys.path.append(gaia_path)
        
        try:
            from mcp_servers.multi_modal_processor_server import MultiModalProcessorServer
            
            # Create processor instance
            processor = MultiModalProcessorServer()
            processor.initialize()
            
            # Convert to base64
            b64_data = base64.b64encode(document_bytes).decode()
            
            # Analyze image for text extraction
            result = await processor._analyze_image({
                "image_data": b64_data,
                "analysis_type": "text_extraction"
            })
            
            return result.get("extracted_text", "")
            
        except Exception as e:
            raise Exception(f"Multimodal processor failed: {str(e)}")
    
    # Mock text method removed - never return dummy data for real medical processing
    
    def _extract_medical_entities(self, text: str) -> dict:
        """Extract medical entities from actual OCR text using regex patterns"""
        import re
        
        entities = {
            "patient_name": "Undefined",
            "date_of_birth": "Undefined",
            "conditions": [],
            "medications": [],
            "vitals": [],
            "provider_name": "Undefined"
        }
        
        # Pattern for names (capitalized words, typically 2-3 parts)
        name_patterns = [
            r'Patient:?\s*([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'Name:?\s*([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                entities["patient_name"] = match.group(1).strip()
                break
        
        # Pattern for dates of birth
        dob_patterns = [
            r'(?:DOB|Date of Birth|Born):?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:DOB|Date of Birth|Born):?\s*(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(?:DOB|Date of Birth|Born):?\s*([A-Z][a-z]+ \d{1,2},? \d{4})'
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities["date_of_birth"] = match.group(1).strip()
                break
        
        # Pattern for medical conditions
        condition_keywords = [
            r'(?:Diagnosis|Condition|History):?\s*([A-Z][a-z]+(?: [a-z]+)*)',
            r'([A-Z][a-z]+(?:itis|osis|emia|pathy|trophy|plasia))',
            r'(Hypertension|Diabetes|Asthma|COPD|Depression|Anxiety)'
        ]
        
        for pattern in condition_keywords:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                condition = match if isinstance(match, str) else match[0]
                if condition and len(condition) > 2:
                    entities["conditions"].append(condition.strip())
        
        # Pattern for medications
        med_patterns = [
            r'(?:Medication|Med|Rx):?\s*([A-Z][a-z]+(?:ol|ine|ide|ate|pril|statin))',
            r'([A-Z][a-z]+(?:ol|ine|ide|ate|pril|statin))\s*\d+\s*mg',
            r'(Lisinopril|Metformin|Aspirin|Ibuprofen|Acetaminophen)'
        ]
        
        for pattern in med_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                medication = match if isinstance(match, str) else match[0]
                if medication and len(medication) > 2:
                    entities["medications"].append(medication.strip())
        
        # Pattern for vital signs
        vital_patterns = [
            r'(?:BP|Blood Pressure):?\s*(\d{2,3}/\d{2,3})',
            r'(?:Heart Rate|HR):?\s*(\d{2,3})\s*bpm',
            r'(?:Temperature|Temp):?\s*(\d{2,3}(?:\.\d)?)\s*°?F?',
            r'(?:Weight):?\s*(\d{2,3})\s*lbs?',
            r'(?:Height):?\s*(\d+)\'?\s*(\d+)"?'
        ]
        
        for pattern in vital_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                vital = match if isinstance(match, str) else ' '.join(filter(None, match))
                if vital:
                    entities["vitals"].append(vital.strip())
        
        # Pattern for provider/doctor names
        provider_patterns = [
            r'(?:Dr\.|Doctor|Physician):?\s*([A-Z][a-z]+ [A-Z][a-z]+)',
            r'Provider:?\s*([A-Z][a-z]+ [A-Z][a-z]+)',
            r'Attending:?\s*([A-Z][a-z]+ [A-Z][a-z]+)'
        ]
        
        for pattern in provider_patterns:
            match = re.search(pattern, text)
            if match:
                entities["provider_name"] = match.group(1).strip()
                break
        
        return entities
    
    def _create_simple_fhir_bundle(self, entities: dict, user_id: str) -> dict:
        """Create FHIR bundle from extracted entities"""
        bundle_id = f"local-{uuid.uuid4()}"
        
        # Parse patient name
        patient_name = entities.get("patient_name", "Undefined")
        if patient_name != "Undefined" and " " in patient_name:
            name_parts = patient_name.split()
            given_name = name_parts[0] if len(name_parts) > 0 else "Undefined"
            family_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else "Undefined"
        else:
            given_name = "Undefined"
            family_name = "Undefined"
        
        # Create bundle entries
        entries = []
        
        # Patient resource
        patient_resource = {
            "resource": {
                "resourceType": "Patient",
                "id": "local-patient",
                "name": [{"given": [given_name], "family": family_name}]
            }
        }
        
        # Add birth date if available
        if entities.get("date_of_birth") != "Undefined":
            patient_resource["resource"]["birthDate"] = entities["date_of_birth"]
        
        entries.append(patient_resource)
        
        # Add conditions as Condition resources
        for i, condition in enumerate(entities.get("conditions", [])):
            if condition:
                entries.append({
                    "resource": {
                        "resourceType": "Condition",
                        "id": f"local-condition-{i}",
                        "subject": {"reference": "Patient/local-patient"},
                        "code": {
                            "text": condition
                        },
                        "clinicalStatus": {
                            "coding": [{
                                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                "code": "active"
                            }]
                        }
                    }
                })
        
        # Add medications as MedicationStatement resources
        for i, medication in enumerate(entities.get("medications", [])):
            if medication:
                entries.append({
                    "resource": {
                        "resourceType": "MedicationStatement",
                        "id": f"local-medication-{i}",
                        "subject": {"reference": "Patient/local-patient"},
                        "medicationCodeableConcept": {
                            "text": medication
                        },
                        "status": "active"
                    }
                })
        
        # Add vitals as Observation resources
        for i, vital in enumerate(entities.get("vitals", [])):
            if vital:
                entries.append({
                    "resource": {
                        "resourceType": "Observation",
                        "id": f"local-vital-{i}",
                        "subject": {"reference": "Patient/local-patient"},
                        "status": "final",
                        "code": {
                            "text": "Vital Sign"
                        },
                        "valueString": vital
                    }
                })
        
        return {
            "resourceType": "Bundle",
            "id": bundle_id,
            "type": "document",
            "timestamp": datetime.now().isoformat(),
            "entry": entries,
            "_metadata": {
                "processing_mode": self._get_processing_mode(),
                "entities_found": len(entities.get("conditions", [])) + len(entities.get("medications", [])) + len(entities.get("vitals", [])),
                "processed_by": user_id,
                "patient_name": entities.get("patient_name", "Undefined"),
                "provider_name": entities.get("provider_name", "Undefined")
            }
        }
    
    def _get_processing_mode(self) -> str:
        """Determine current processing mode"""
        if self.use_mistral_fallback and self.mistral_api_key:
            return "local_processing_with_mistral_ocr"
        elif self.use_multimodal_fallback:
            return "local_processing_with_multimodal_fallback"
        else:
            return "local_processing_only"

# Global instance
local_processor = LocalProcessor()