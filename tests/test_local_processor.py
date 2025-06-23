"""
Tests for Local Mock Processor
Simple tests to verify mock processing functionality
"""

import pytest
import asyncio
import os
from unittest.mock import patch, Mock
from src.file_processor import LocalProcessor

class TestLocalProcessor:
    """Test suite for the local processor"""
    
    @pytest.fixture
    def local_processor(self):
        """Create a local processor instance"""
        return LocalProcessor()
    
    @pytest.fixture
    def sample_document_bytes(self):
        """Sample document bytes for testing"""
        return b"Mock PDF document content"
    
    @pytest.mark.asyncio
    async def test_basic_document_processing(self, local_processor, sample_document_bytes):
        """Test basic document processing without fallbacks"""
        result = await local_processor.process_document(
            document_bytes=sample_document_bytes,
            user_id="test-user-123",
            filename="test_document.pdf"
        )
        
        # Verify response structure
        assert result["status"] == "success"
        assert result["filename"] == "test_document.pdf"
        assert result["processed_by"] == "test-user-123"
        assert "entities_found" in result
        assert "fhir_bundle" in result
        assert "extracted_text" in result
        
        # Verify FHIR bundle structure
        fhir_bundle = result["fhir_bundle"]
        assert fhir_bundle["resourceType"] == "Bundle"
        assert fhir_bundle["type"] == "document"
        assert len(fhir_bundle["entry"]) >= 2  # Patient + Observation
        
        # Check for required FHIR resources
        resource_types = [entry["resource"]["resourceType"] for entry in fhir_bundle["entry"]]
        assert "Patient" in resource_types
        assert "Observation" in resource_types
    
    def test_mock_text_extraction_by_file_type(self, local_processor):
        """Test text extraction based on file types"""
        # Test PDF/DOC files
        pdf_text = local_processor._get_mock_text_by_type("medical_record.pdf")
        assert "MEDICAL RECORD" in pdf_text
        assert "John Doe" in pdf_text
        
        # Test image files
        image_text = local_processor._get_mock_text_by_type("lab_results.jpg")
        assert "LAB REPORT" in image_text
        assert "Jane Smith" in image_text
        
        # Test other files
        other_text = local_processor._get_mock_text_by_type("notes.txt")
        assert "CLINICAL NOTE" in other_text
    
    def test_medical_entity_extraction(self, local_processor):
        """Test medical entity extraction"""
        test_text = """
        Patient: John Doe
        Diagnosis: Hypertension
        Medication: Lisinopril
        Blood Pressure: 140/90
        """
        
        entities = local_processor._extract_medical_entities(test_text)
        
        # Should find multiple entities
        assert len(entities) > 0
        
        # Check entity types
        entity_types = [entity["type"] for entity in entities]
        assert "PERSON" in entity_types
        assert "CONDITION" in entity_types
        assert "MEDICATION" in entity_types
        assert "VITAL" in entity_types
        
        # Verify entity structure
        for entity in entities:
            assert "text" in entity
            assert "type" in entity
            assert "confidence" in entity
            assert "start" in entity
            assert "end" in entity
    
    def test_processing_mode_detection(self, local_processor):
        """Test processing mode detection"""
        # Test default mode
        mode = local_processor._get_processing_mode()
        assert mode == "local_mock_only"
        
        # Test with environment variables
        with patch.dict(os.environ, {"USE_MISTRAL_FALLBACK": "true", "MISTRAL_API_KEY": "test-key"}):
            processor = LocalProcessor()
            mode = processor._get_processing_mode()
            assert mode == "local_mock_with_mistral_fallback"
        
        with patch.dict(os.environ, {"USE_MULTIMODAL_FALLBACK": "true"}):
            processor = LocalProcessor()
            mode = processor._get_processing_mode()
            assert mode == "local_mock_with_multimodal_fallback"
    
    @pytest.mark.asyncio
    async def test_fallback_handling(self, local_processor, sample_document_bytes):
        """Test fallback mechanisms"""
        # Test with fallbacks disabled (default)
        text = await local_processor._extract_text_with_fallback(sample_document_bytes, "test.pdf")
        assert isinstance(text, str)
        assert len(text) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("MISTRAL_API_KEY"), reason="Mistral API key not available")
    async def test_mistral_fallback(self, local_processor, sample_document_bytes):
        """Test Mistral API fallback (requires API key)"""
        with patch.dict(os.environ, {"USE_MISTRAL_FALLBACK": "true"}):
            processor = LocalProcessor()
            
            # Mock the Mistral API response
            with patch('httpx.AsyncClient.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Extracted medical text from Mistral"}}]
                }
                mock_post.return_value = mock_response
                
                text = await processor._extract_with_mistral(sample_document_bytes)
                assert text == "Extracted medical text from Mistral"
    
    def test_fhir_bundle_creation(self, local_processor):
        """Test FHIR bundle creation"""
        test_entities = [
            {"text": "John Doe", "type": "PERSON", "confidence": 0.95},
            {"text": "Hypertension", "type": "CONDITION", "confidence": 0.89}
        ]
        
        bundle = local_processor._create_simple_fhir_bundle(test_entities, "test-user")
        
        # Verify bundle structure
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "document"
        assert "timestamp" in bundle
        assert "entry" in bundle
        
        # Verify metadata
        assert bundle["_metadata"]["entities_found"] == 2
        assert bundle["_metadata"]["processed_by"] == "test-user"
        
        # Verify LOINC codes in observations
        observation_entry = next(
            entry for entry in bundle["entry"] 
            if entry["resource"]["resourceType"] == "Observation"
        )
        coding = observation_entry["resource"]["code"]["coding"][0]
        assert coding["system"] == "http://loinc.org"
        assert coding["code"] == "85354-9"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])