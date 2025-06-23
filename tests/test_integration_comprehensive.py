#!/usr/bin/env python3
"""
Comprehensive Integration Tests for FhirFlame Medical AI Platform
Tests OCR method selection, Mistral API integration, Ollama processing, and FHIR generation
"""

import asyncio
import pytest
import os
import io
from PIL import Image, ImageDraw, ImageFont
import json
import time

# Add src to path for module imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from workflow_orchestrator import WorkflowOrchestrator
from codellama_processor import CodeLlamaProcessor
from file_processor import FileProcessor

class TestOCRMethodSelection:
    """Test OCR method selection logic"""
    
    def test_mistral_auto_selection_with_api_key(self):
        """Test that Mistral OCR is auto-selected when API key is present"""
        # Simulate environment with Mistral API key
        original_key = os.environ.get("MISTRAL_API_KEY")
        os.environ["MISTRAL_API_KEY"] = "test_key"
        
        try:
            orchestrator = WorkflowOrchestrator()
            assert orchestrator.mistral_api_key == "test_key"
            
            # Test auto-selection logic
            use_mistral_ocr = None  # Trigger auto-selection
            auto_selected = bool(orchestrator.mistral_api_key) if use_mistral_ocr is None else use_mistral_ocr
            
            assert auto_selected == True, "Mistral OCR should be auto-selected when API key present"
            
        finally:
            if original_key:
                os.environ["MISTRAL_API_KEY"] = original_key
            else:
                os.environ.pop("MISTRAL_API_KEY", None)
    
    def test_mistral_not_selected_without_api_key(self):
        """Test that Mistral OCR is not selected when API key is missing"""
        # Simulate environment without Mistral API key
        original_key = os.environ.get("MISTRAL_API_KEY")
        os.environ.pop("MISTRAL_API_KEY", None)
        
        try:
            orchestrator = WorkflowOrchestrator()
            assert orchestrator.mistral_api_key is None
            
            # Test auto-selection logic
            use_mistral_ocr = None  # Trigger auto-selection
            auto_selected = bool(orchestrator.mistral_api_key) if use_mistral_ocr is None else use_mistral_ocr
            
            assert auto_selected == False, "Mistral OCR should not be selected when API key missing"
            
        finally:
            if original_key:
                os.environ["MISTRAL_API_KEY"] = original_key

class TestMistralOCRIntegration:
    """Test Mistral OCR integration and processing"""
    
    @pytest.mark.asyncio
    async def test_mistral_ocr_document_processing(self):
        """Test complete Mistral OCR document processing workflow"""
        # Create test medical document
        test_image = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(test_image)
        
        medical_text = """MEDICAL REPORT
Patient: Jane Smith
DOB: 02/15/1985
Diagnosis: Hypertension
Medication: Lisinopril 10mg
Blood Pressure: 140/90 mmHg
Provider: Dr. Johnson"""
        
        draw.text((50, 50), medical_text, fill='black')
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='JPEG', quality=95)
        document_bytes = img_byte_arr.getvalue()
        
        # Test document processing
        orchestrator = WorkflowOrchestrator()
        
        if orchestrator.mistral_api_key:
            result = await orchestrator.process_complete_workflow(
                document_bytes=document_bytes,
                user_id="test_user",
                filename="test_medical_report.jpg",
                use_mistral_ocr=True
            )
            
            # Validate results
            assert result['workflow_metadata']['mistral_ocr_used'] == True
            assert result['workflow_metadata']['ocr_method'] == "mistral_api"
            assert result['text_extraction']['full_text_length'] > 0
            assert 'Jane Smith' in result['text_extraction']['extracted_text'] or \
                   'Hypertension' in result['text_extraction']['extracted_text']
    
    def test_document_size_calculation(self):
        """Test document size calculation and timeout estimation"""
        # Create test document
        test_image = Image.new('RGB', (800, 600), color='white')
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='JPEG', quality=95)
        document_bytes = img_byte_arr.getvalue()
        
        # Test size calculations
        document_size = len(document_bytes)
        file_size_mb = document_size / (1024 * 1024)
        
        # Test timeout calculation logic
        base64_size = len(document_bytes) * 4 / 3  # Approximate base64 size
        dynamic_timeout = min(300.0, 60.0 + (base64_size / 100000))
        
        assert document_size > 0
        assert file_size_mb > 0
        assert dynamic_timeout >= 60.0
        assert dynamic_timeout <= 300.0

class TestOllamaIntegration:
    """Test Ollama CodeLlama integration"""
    
    @pytest.mark.asyncio
    async def test_ollama_connectivity(self):
        """Test Ollama connection and processing"""
        processor = CodeLlamaProcessor()
        
        if processor.use_real_ollama:
            medical_text = """Patient: John Smith
DOB: 01/15/1980
Diagnosis: Type 2 diabetes, hypertension
Medications: 
- Metformin 1000mg twice daily
- Lisinopril 10mg daily
Vitals: BP 142/88 mmHg, HbA1c 7.2%"""
            
            try:
                result = await processor.process_document(
                    medical_text=medical_text,
                    document_type="clinical_note",
                    extract_entities=True,
                    generate_fhir=False
                )
                
                # Validate Ollama processing results
                assert result['processing_mode'] == 'real_ollama'
                assert result['success'] == True
                assert 'extracted_data' in result
                
                extracted_data = json.loads(result['extracted_data'])
                assert len(extracted_data.get('conditions', [])) > 0
                assert len(extracted_data.get('medications', [])) > 0
                
            except Exception as e:
                pytest.skip(f"Ollama not available: {e}")

class TestRuleBasedFallback:
    """Test rule-based processing fallback"""
    
    @pytest.mark.asyncio
    async def test_rule_based_entity_extraction(self):
        """Test rule-based entity extraction with real medical text"""
        processor = CodeLlamaProcessor()
        
        medical_text = """Patient: Sarah Johnson
DOB: 03/12/1975
Diagnosis: Hypertension, Type 2 diabetes
Medications: 
- Lisinopril 10mg daily
- Metformin 500mg twice daily
- Insulin glargine 15 units at bedtime
Vitals: Blood Pressure: 142/88 mmHg, HbA1c: 7.2%"""
        
        # Force rule-based processing
        original_ollama_setting = processor.use_real_ollama
        processor.use_real_ollama = False
        
        try:
            result = await processor.process_document(
                medical_text=medical_text,
                document_type="clinical_note",
                extract_entities=True,
                generate_fhir=False
            )
            
            # Validate rule-based processing
            extracted_data = json.loads(result['extracted_data'])
            
            # Check patient extraction
            assert 'Sarah Johnson' in extracted_data.get('patient', '') or \
                   extracted_data.get('patient') != 'Unknown Patient'
            
            # Check condition extraction
            conditions = extracted_data.get('conditions', [])
            assert any('hypertension' in condition.lower() for condition in conditions)
            assert any('diabetes' in condition.lower() for condition in conditions)
            
            # Check medication extraction
            medications = extracted_data.get('medications', [])
            assert any('lisinopril' in med.lower() for med in medications)
            assert any('metformin' in med.lower() for med in medications)
            
        finally:
            processor.use_real_ollama = original_ollama_setting

class TestWorkflowIntegration:
    """Test complete workflow integration"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_stages(self):
        """Test all workflow stages complete successfully"""
        orchestrator = WorkflowOrchestrator()
        
        # Test with text input
        medical_text = """MEDICAL RECORD
Patient: Test Patient
DOB: 01/01/1990
Chief Complaint: Chest pain
Assessment: Acute coronary syndrome
Plan: Aspirin 325mg daily, Atorvastatin 40mg daily"""
        
        result = await orchestrator.process_complete_workflow(
            medical_text=medical_text,
            user_id="test_user",
            filename="test_record.txt",
            document_type="clinical_note",
            use_advanced_llm=True,
            generate_fhir=True
        )
        
        # Validate workflow completion
        assert result['status'] == 'success'
        assert result['workflow_metadata']['total_processing_time'] > 0
        assert len(result['workflow_metadata']['stages_completed']) > 0
        
        # Check text extraction stage
        assert 'text_extraction' in result
        assert result['text_extraction']['full_text_length'] > 0
        
        # Check medical analysis stage
        assert 'medical_analysis' in result
        assert result['medical_analysis']['entities_found'] >= 0
        
        # Check FHIR generation if enabled
        if result['workflow_metadata']['fhir_generated']:
            assert 'fhir_bundle' in result
            assert result['fhir_bundle'] is not None

class TestErrorHandling:
    """Test error handling and fallback mechanisms"""
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self):
        """Test handling of invalid or insufficient input"""
        processor = CodeLlamaProcessor()
        
        # Test empty input
        result = await processor.process_document(
            medical_text="",
            document_type="clinical_note",
            extract_entities=True
        )
        
        extracted_data = json.loads(result['extracted_data'])
        assert extracted_data.get('patient') == 'Unknown Patient'
        assert len(extracted_data.get('conditions', [])) == 0
        
        # Test very short input
        result = await processor.process_document(
            medical_text="test",
            document_type="clinical_note",
            extract_entities=True
        )
        
        extracted_data = json.loads(result['extracted_data'])
        assert result['processing_metadata']['reason'] == "Input text too short or empty"

class TestPerformanceMetrics:
    """Test performance and timing metrics"""
    
    @pytest.mark.asyncio
    async def test_processing_time_tracking(self):
        """Test that processing times are tracked correctly"""
        orchestrator = WorkflowOrchestrator()
        
        start_time = time.time()
        
        result = await orchestrator.process_complete_workflow(
            medical_text="Patient: Test Patient, Condition: Test condition",
            user_id="test_user",
            filename="test.txt",
            use_advanced_llm=False  # Use faster processing for timing test
        )
        
        end_time = time.time()
        actual_time = end_time - start_time
        
        # Validate timing tracking
        assert result['workflow_metadata']['total_processing_time'] > 0
        assert result['workflow_metadata']['total_processing_time'] <= actual_time + 1.0  # Allow 1s tolerance

if __name__ == "__main__":
    pytest.main([__file__, "-v"])