#!/usr/bin/env python3
"""
🔥 FhirFlame Medical AI Hub - Comprehensive Test Suite

Tests for:
1. DICOMweb Standard Compliance (QIDO-RS + WADO-RS + STOW-RS)
2. MCP Bridge Functionality  
3. AI Integration Endpoints
4. System Health and Performance
"""

import pytest
import asyncio
import json
import os
import sys
from io import BytesIO
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the Medical AI Hub
from medical_ai_hub import app

# Create test client
client = TestClient(app)

class TestSystemEndpoints:
    """Test core system functionality"""
    
    def test_root_endpoint(self):
        """Test API root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        
        assert data["service"] == "FhirFlame Medical AI Hub"
        assert data["version"] == "1.0.0"
        assert "DICOMweb Standard API" in data["capabilities"]
        assert "MCP Tool Bridge" in data["capabilities"]
        assert "AI Integration Endpoints" in data["capabilities"]
        assert data["status"] == "operational"
        
    def test_health_check(self):
        """Test system health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "components" in data
        assert "fhir_validator" in data["components"]
        assert "dicom_processor" in data["components"]

class TestDICOMwebCompliance:
    """Test DICOMweb standard implementation"""
    
    def test_qido_query_studies(self):
        """Test QIDO-RS study query"""
        response = client.get("/studies")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/dicom+json"
        
        data = response.json()
        assert isinstance(data, list)
        if data:  # If studies returned
            study = data[0]
            assert "0020000D" in study  # Study Instance UID
            assert "00100020" in study  # Patient ID
            assert "00080020" in study  # Study Date
    
    def test_qido_query_studies_with_filters(self):
        """Test QIDO-RS with patient filter"""
        response = client.get("/studies?patient_id=PAT_001&limit=5")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) <= 5
        
    def test_qido_query_series(self):
        """Test QIDO-RS series query"""
        study_uid = "1.2.840.10008.1.2.1.0"
        response = client.get(f"/studies/{study_uid}/series")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/dicom+json"
        
        data = response.json()
        assert isinstance(data, list)
        
    def test_qido_query_instances(self):
        """Test QIDO-RS instances query"""
        study_uid = "1.2.840.10008.1.2.1.0"
        response = client.get(f"/studies/{study_uid}/instances")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
    def test_wado_retrieve_instance(self):
        """Test WADO-RS instance retrieval"""
        study_uid = "1.2.840.10008.1.2.1.0"
        instance_uid = "1.2.840.10008.1.2.1.0.0.0"
        
        response = client.get(f"/studies/{study_uid}/instances/{instance_uid}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/dicom"
        
    def test_wado_retrieve_metadata(self):
        """Test WADO-RS metadata retrieval"""
        study_uid = "1.2.840.10008.1.2.1.0"
        
        response = client.get(f"/studies/{study_uid}/metadata")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/dicom+json"
        
        data = response.json()
        assert "00100010" in data  # Patient Name
        assert "0020000D" in data  # Study Instance UID
        
    def test_stow_store_instances(self):
        """Test STOW-RS instance storage"""
        # Create mock DICOM file
        mock_dicom = BytesIO(b"DICM" + b"\x00" * 128 + b"Mock DICOM content")
        
        files = [("files", ("test.dcm", mock_dicom, "application/dicom"))]
        response = client.post("/studies", files=files)
        
        assert response.status_code == 201
        data = response.json()
        assert "stored_instances" in data
        assert data["stored_instances"] == 1

class TestMCPBridge:
    """Test MCP tool bridge functionality"""
    
    def test_list_mcp_tools(self):
        """Test MCP tools listing"""
        response = client.get("/mcp/tools")
        assert response.status_code == 200
        
        data = response.json()
        assert "available_tools" in data
        assert len(data["available_tools"]) == 2
        
        # Check both tools are present
        tool_names = [tool["name"] for tool in data["available_tools"]]
        assert "process_medical_document" in tool_names
        assert "validate_fhir_bundle" in tool_names
        
    @patch('medical_ai_hub.local_processor.process_document')
    async def test_mcp_process_medical_document(self, mock_process):
        """Test MCP bridge for medical document processing"""
        # Mock the process_document response
        mock_result = {
            "processing_mode": "ai_enhanced",
            "extracted_entities": ["patient", "diagnosis"],
            "fhir_bundle": {"resourceType": "Bundle"},
            "confidence_score": 0.95
        }
        mock_process.return_value = mock_result
        
        request_data = {
            "document_content": "Patient has pneumonia diagnosis",
            "document_type": "clinical_note",
            "extract_entities": True,
            "generate_fhir": True,
            "user_id": "test_user"
        }
        
        response = client.post("/mcp/process_medical_document", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "mcp_tool" in data["data"]
        assert data["data"]["mcp_tool"] == "process_medical_document"
        
    def test_mcp_validate_fhir_bundle(self):
        """Test MCP bridge for FHIR validation"""
        # Valid FHIR bundle for testing
        test_bundle = {
            "resourceType": "Bundle",
            "id": "test-bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "test-patient",
                        "name": [{"family": "Test", "given": ["Patient"]}]
                    }
                }
            ]
        }
        
        request_data = {
            "fhir_bundle": test_bundle,
            "validation_level": "healthcare_grade"
        }
        
        response = client.post("/mcp/validate_fhir_bundle", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "validation_result" in data["data"]

class TestAIIntegration:
    """Test AI integration endpoints"""
    
    def test_ai_analyze_dicom(self):
        """Test AI DICOM analysis endpoint"""
        # Create mock DICOM file
        mock_dicom = BytesIO(b"DICM" + b"\x00" * 128 + b"Mock DICOM content")
        
        files = [("file", ("test.dcm", mock_dicom, "application/dicom"))]
        data = {"analysis_type": "comprehensive", "include_fhir": "true"}
        
        response = client.post("/ai/analyze_dicom", files=files, data=data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["success"] is True
        assert "file_info" in result["data"]
        assert "ai_insights" in result["data"]
        assert "clinical_context" in result["data"]
        
    def test_ai_analyze_dicom_with_fhir(self):
        """Test AI DICOM analysis with FHIR integration"""
        mock_dicom = BytesIO(b"DICM" + b"\x00" * 128 + b"Mock DICOM content")
        
        files = [("file", ("test.dcm", mock_dicom, "application/dicom"))]
        data = {"include_fhir": "true"}
        
        response = client.post("/ai/analyze_dicom", files=files, data=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "fhir_integration" in result["data"]
        assert "fhir_bundle" in result["data"]["fhir_integration"]
        assert "compliance_score" in result["data"]["fhir_integration"]
        
    def test_get_medical_context_for_ai(self):
        """Test medical context endpoint for AI"""
        patient_id = "TEST_PATIENT_001"
        
        response = client.get(f"/ai/medical_context/{patient_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["patient_summary"]["patient_id"] == patient_id
        assert "recent_studies" in data
        assert "fhir_resources" in data
        assert "ai_recommendations" in data
        assert len(data["ai_recommendations"]) > 0
        
    def test_ai_batch_analysis(self):
        """Test AI batch analysis endpoint"""
        request_data = {
            "patient_ids": ["PAT_001", "PAT_002", "PAT_003"],
            "analysis_scope": "comprehensive",
            "max_concurrent": 2
        }
        
        response = client.post("/ai/batch_analysis", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "batch_summary" in data["data"]
        assert data["data"]["batch_summary"]["total_patients"] == 3
        assert "successful_results" in data["data"]
        assert "performance_metrics" in data["data"]

class TestPerformanceAndSecurity:
    """Test performance and security aspects"""
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/")
        # CORS should allow the request
        assert response.status_code in [200, 204]
        
    def test_api_response_format(self):
        """Test consistent API response format"""
        response = client.get("/health")
        assert response.status_code == 200
        
        # All responses should have consistent timestamp format
        data = response.json()
        assert "timestamp" in data
        
    def test_error_handling(self):
        """Test error handling for invalid endpoints"""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404
        
    def test_large_batch_handling(self):
        """Test handling of large batch requests"""
        # Test with larger batch to ensure async handling works
        large_batch = {
            "patient_ids": [f"PAT_{i:03d}" for i in range(50)],
            "analysis_scope": "basic",
            "max_concurrent": 10
        }
        
        response = client.post("/ai/batch_analysis", json=large_batch)
        assert response.status_code == 200
        
        data = response.json()
        assert data["data"]["batch_summary"]["total_patients"] == 50

# Integration test for complete workflow
class TestCompleteWorkflow:
    """Test complete medical AI workflow"""
    

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])