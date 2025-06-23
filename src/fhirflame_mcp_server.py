"""
FhirFlame MCP Server - Medical Document Intelligence Platform
MCP Server with 2 perfect tools: process_medical_document & validate_fhir_bundle
CodeLlama 13B-instruct + RTX 4090 GPU optimization
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from .monitoring import monitor

# Use correct MCP imports for fast initial testing
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    from mcp import CallToolRequest
except ImportError:
    # Mock for testing if MCP not available
    class Server:
        def __init__(self, name): pass
    class Tool:
        def __init__(self, **kwargs): pass
    class TextContent:
        def __init__(self, **kwargs): pass
    class CallToolRequest:
        pass


class FhirFlameMCPServer:
    """MCP Server for medical document processing with CodeLlama 13B"""
    
    def __init__(self):
        """Initialize FhirFlame MCP Server"""
        self.name = "fhirflame"
        self.server = None  # Will be initialized when needed
        self._tool_definitions = self._register_tools()
        self.tools = [tool["name"] for tool in self._tool_definitions]  # Tool names for compatibility
        
    def _register_tools(self) -> List[Dict[str, Any]]:
        """Register the 2 perfect MCP tools"""
        return [
            {
                "name": "process_medical_document",
                "description": "Process medical documents using CodeLlama 13B-instruct on RTX 4090",
                "parameters": {
                    "document_content": {
                        "type": "string",
                        "description": "Medical document text to process",
                        "required": True
                    },
                    "document_type": {
                        "type": "string",
                        "description": "Type of medical document",
                        "enum": ["discharge_summary", "clinical_note", "lab_report"],
                        "default": "clinical_note",
                        "required": False
                    },
                    "extract_entities": {
                        "type": "boolean",
                        "description": "Whether to extract medical entities",
                        "default": True,
                        "required": False
                    }
                }
            },
            {
                "name": "validate_fhir_bundle",
                "description": "Validate FHIR R4 bundles for healthcare compliance",
                "parameters": {
                    "fhir_bundle": {
                        "type": "object",
                        "description": "FHIR R4 bundle to validate",
                        "required": True
                    },
                    "validation_level": {
                        "type": "string",
                        "description": "Validation strictness level",
                        "enum": ["basic", "standard", "healthcare_grade"],
                        "default": "standard",
                        "required": False
                    }
                }
            }
        ]
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get available MCP tools"""
        return self._tool_definitions
    
    def get_tool(self, name: str) -> Dict[str, Any]:
        """Get a specific tool by name"""
        for tool in self._tool_definitions:
            if tool["name"] == name:
                return tool
        raise ValueError(f"Tool not found: {name}")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool by name"""
        if name == "process_medical_document":
            return await self._process_medical_document(arguments)
        elif name == "validate_fhir_bundle":
            return await self._validate_fhir_bundle(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def _process_medical_document(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Process medical document with CodeLlama 13B"""
        from .codellama_processor import CodeLlamaProcessor
        
        medical_text = args.get("document_content", "")
        document_type = args.get("document_type", "clinical_note")
        extract_entities = args.get("extract_entities", True)
        
        # Edge case: Handle empty document content
        if not medical_text or medical_text.strip() == "":
            return {
                "success": False,
                "error": "Empty document content provided. Cannot process empty medical documents.",
                "processing_metadata": {
                    "model_used": "codellama:13b-instruct",
                    "gpu_used": "RTX_4090",
                    "vram_used": "0GB",
                    "processing_time": 0.0
                }
            }
        
        # Real CodeLlama processing implementation
        processor = CodeLlamaProcessor()
        
        try:
            # Process the medical document with FHIR bundle generation
            processing_result = await processor.process_document(
                medical_text,
                document_type=document_type,
                extract_entities=extract_entities,
                generate_fhir=True
            )
            
            return {
                "success": True,
                "processing_metadata": processing_result.get("metadata", {}),
                "extraction_results": processing_result.get("extraction_results", {}),
                "extracted_data": processing_result.get("extracted_data", "{}"),
                "entities_extracted": extract_entities,
                "fhir_bundle": processing_result.get("fhir_bundle", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "processing_metadata": {
                    "model_used": "codellama:13b-instruct",
                    "gpu_used": "RTX_4090",
                    "vram_used": "0GB",
                    "processing_time": 0.0
                }
            }
    
    async def _validate_fhir_bundle(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate FHIR R4 bundle"""
        from .fhir_validator import FhirValidator
        
        fhir_bundle = args.get("fhir_bundle", {})
        validation_level = args.get("validation_level", "standard")
        
        # Edge case: Handle empty or invalid bundle
        if not fhir_bundle or not isinstance(fhir_bundle, dict):
            return {
                "success": False,
                "error": "Invalid or empty FHIR bundle provided",
                "validation_results": {
                    "is_valid": False,
                    "compliance_score": 0.0,
                    "validation_level": validation_level,
                    "fhir_version": "R4"
                },
                "compliance_summary": {
                    "fhir_r4_compliant": False,
                    "hipaa_ready": False,
                    "terminology_validated": False,
                    "structure_validated": False
                },
                "compliance_score": 0.0,
                "validation_errors": ["Bundle is empty or invalid"],
                "warnings": [],
                "healthcare_grade": False
            }
        
        # Real FHIR validation implementation
        validator = FhirValidator()
        
        try:
            # Validate the FHIR bundle using sync method
            validation_result = validator.validate_bundle(fhir_bundle, validation_level=validation_level)
            
            return {
                "success": True,
                "validation_results": {
                    "is_valid": validation_result["is_valid"],
                    "compliance_score": validation_result["compliance_score"],
                    "validation_level": validation_result["validation_level"],
                    "fhir_version": validation_result["fhir_version"]
                },
                "compliance_summary": {
                    "fhir_r4_compliant": validation_result["fhir_r4_compliant"],
                    "hipaa_ready": validation_result["hipaa_compliant"],
                    "terminology_validated": validation_result["medical_coding_validated"],
                    "structure_validated": validation_result["is_valid"]
                },
                "compliance_score": validation_result["compliance_score"],
                "validation_errors": validation_result["errors"],
                "warnings": validation_result["warnings"],
                "healthcare_grade": validation_level == "healthcare_grade"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Validation failed: {str(e)}",
                "validation_results": {
                    "is_valid": False,
                    "compliance_score": 0.0,
                    "validation_level": validation_level,
                    "fhir_version": "R4"
                },
                "compliance_summary": {
                    "fhir_r4_compliant": False,
                    "hipaa_ready": False,
                    "terminology_validated": False,
                    "structure_validated": False
                },
                "compliance_score": 0.0,
                "validation_errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "healthcare_grade": False
            }
    
    async def run_server(self, port: int = 8000):
        """Run MCP server"""
        # This will be implemented with actual MCP server logic
        pass


# Make class available for import
__all__ = ["FhirFlameMCPServer"]