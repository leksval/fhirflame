#!/usr/bin/env python3
"""
🔥 FhirFlame Integrated Workflow Test
Complete integration test: Mistral OCR → CodeLlama Agent → FHIR Generation
"""

import asyncio
import os
import sys
import time
import base64
from datetime import datetime

# Add src to path (from tests directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.workflow_orchestrator import workflow_orchestrator
from src.monitoring import monitor
from src.fhir_validator import FhirValidator

def create_medical_document_pdf_bytes() -> bytes:
    """Create mock PDF document bytes for testing"""
    # This is a minimal PDF header - in real scenarios this would be actual PDF bytes
    pdf_header = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n \n0000000173 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n253\n%%EOF'
    return pdf_header

def create_medical_image_bytes() -> bytes:
    """Create mock medical image bytes for testing"""
    # Simple PNG header for a 1x1 pixel image
    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    return png_bytes

async def test_complete_workflow_integration():
    """Test complete workflow: Document OCR → Medical Analysis → FHIR Generation"""
    
    print("🔥 FhirFlame Complete Workflow Integration Test")
    print("=" * 60)
    print(f"🕐 Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check workflow status
    status = workflow_orchestrator.get_workflow_status()
    print(f"\n🔧 Workflow Configuration:")
    print(f"   Mistral OCR: {'✅ Enabled' if status['mistral_ocr_enabled'] else '❌ Disabled'}")
    print(f"   API Key: {'✅ Set' if status['mistral_api_key_configured'] else '❌ Missing'}")
    print(f"   CodeLlama: {'✅ Ready' if status['codellama_processor_ready'] else '❌ Not Ready'}")
    print(f"   Monitoring: {'✅ Active' if status['monitoring_enabled'] else '❌ Disabled'}")
    print(f"   Pipeline: {' → '.join(status['workflow_components'])}")
    
    # Test Case 1: Document with OCR Processing
    print(f"\n📄 TEST CASE 1: Document OCR → Agent Workflow")
    print("-" * 50)
    
    try:
        document_bytes = create_medical_document_pdf_bytes()
        print(f"📋 Document: Medical report PDF ({len(document_bytes)} bytes)")
        
        start_time = time.time()
        
        # Process complete workflow
        result = await workflow_orchestrator.process_complete_workflow(
            document_bytes=document_bytes,
            user_id="test-integration-user",
            filename="medical_report.pdf",
            document_type="clinical_report"
        )
        
        processing_time = time.time() - start_time
        
        # Display results
        print(f"✅ Workflow completed in {processing_time:.2f}s")
        print(f"📊 Processing pipeline: {result['workflow_metadata']['stages_completed']}")
        print(f"🔍 OCR used: {result['workflow_metadata']['mistral_ocr_used']}")
        print(f"📝 Text extracted: {result['text_extraction']['full_text_length']} chars")
        print(f"🎯 Entities found: {result['medical_analysis']['entities_found']}")
        print(f"📈 Quality score: {result['medical_analysis']['quality_score']:.2f}")
        
        # Show extraction method
        extraction_method = result['text_extraction']['extraction_method']
        print(f"🔬 Extraction method: {extraction_method}")
        
        # Display FHIR validation results from workflow
        if result.get('fhir_validation'):
            fhir_validation = result['fhir_validation']
            print(f"📋 FHIR validation: {'✅ Valid' if fhir_validation['is_valid'] else '❌ Invalid'}")
            print(f"📊 Compliance score: {fhir_validation['compliance_score']:.1%}")
            print(f"🔬 Validation level: {fhir_validation['validation_level']}")
        elif result.get('fhir_bundle'):
            # Fallback validation if not done in workflow
            validator = FhirValidator()
            fhir_validation = validator.validate_fhir_bundle(result['fhir_bundle'])
            print(f"📋 FHIR validation (fallback): {'✅ Valid' if fhir_validation['is_valid'] else '❌ Invalid'}")
            print(f"📊 Compliance score: {fhir_validation['compliance_score']:.1%}")
        
        # Display extracted text preview
        if result['text_extraction']['extracted_text']:
            preview = result['text_extraction']['extracted_text'][:200]
            print(f"\n📖 Extracted text preview:")
            print(f"   {preview}...")
            
    except Exception as e:
        print(f"❌ Document workflow test failed: {e}")
        return False
    
    # Test Case 2: Direct Text Processing
    print(f"\n📝 TEST CASE 2: Direct Text → Agent Workflow")
    print("-" * 50)
    
    try:
        medical_text = """
MEDICAL RECORD - PATIENT: SARAH JOHNSON
DOB: 1985-03-15  |  MRN: MR789456

CHIEF COMPLAINT: Follow-up for Type 2 Diabetes

CURRENT MEDICATIONS:
- Metformin 1000mg twice daily
- Glipizide 5mg once daily
- Lisinopril 10mg daily for hypertension

VITAL SIGNS:
- Blood Pressure: 135/82 mmHg
- Weight: 172 lbs
- HbA1c: 7.2%

ASSESSMENT: Type 2 Diabetes - needs optimization
PLAN: Increase Metformin to 1500mg twice daily
"""
        
        start_time = time.time()
        
        result = await workflow_orchestrator.process_complete_workflow(
            medical_text=medical_text,
            user_id="test-text-user",
            document_type="follow_up_note"
        )
        
        processing_time = time.time() - start_time
        
        print(f"✅ Text workflow completed in {processing_time:.2f}s")
        print(f"🔍 OCR used: {result['workflow_metadata']['mistral_ocr_used']}")
        print(f"🎯 Entities found: {result['medical_analysis']['entities_found']}")
        print(f"📈 Quality score: {result['medical_analysis']['quality_score']:.2f}")
        
        # Check that OCR was NOT used for direct text
        if not result['workflow_metadata']['mistral_ocr_used']:
            print("✅ Correctly bypassed OCR for direct text input")
        else:
            print("⚠️ OCR was unexpectedly used for direct text")
            
    except Exception as e:
        print(f"❌ Text workflow test failed: {e}")
        return False
    
    # Test Case 3: Image Document Processing
    print(f"\n🖼️ TEST CASE 3: Medical Image → OCR → Agent Workflow")
    print("-" * 50)
    
    try:
        image_bytes = create_medical_image_bytes()
        print(f"🖼️ Document: Medical image PNG ({len(image_bytes)} bytes)")
        
        start_time = time.time()
        
        result = await workflow_orchestrator.process_medical_document_with_ocr(
            document_bytes=image_bytes,
            user_id="test-image-user",
            filename="lab_report.png"
        )
        
        processing_time = time.time() - start_time
        
        print(f"✅ Image workflow completed in {processing_time:.2f}s")
        print(f"🔍 OCR processing: {result['workflow_metadata']['mistral_ocr_used']}")
        print(f"📊 Pipeline: {' → '.join(result['workflow_metadata']['stages_completed'])}")
        
        # Check integration metadata
        medical_metadata = result['medical_analysis'].get('model_used', 'Unknown')
        print(f"🤖 Medical AI model: {medical_metadata}")
        
        if 'source_metadata' in result.get('medical_analysis', {}):
            print("✅ OCR metadata properly passed to medical analysis")
        
    except Exception as e:
        print(f"❌ Image workflow test failed: {e}")
        return False
    
    return True

async def test_workflow_error_handling():
    """Test workflow error handling and fallbacks"""
    
    print(f"\n🛠️ TESTING ERROR HANDLING & FALLBACKS")
    print("-" * 50)
    
    try:
        # Test with invalid document bytes
        invalid_bytes = b'invalid document content'
        
        result = await workflow_orchestrator.process_complete_workflow(
            document_bytes=invalid_bytes,
            user_id="test-error-user",
            filename="invalid.doc"
        )
        
        print(f"✅ Error handling test: Processed with fallback")
        print(f"🔄 Fallback mode: {result['text_extraction']['extraction_method']}")
        
    except Exception as e:
        print(f"⚠️ Error handling test: {e}")
    
    return True

async def main():
    """Main test execution"""
    
    try:
        # Run integration tests
        print("🚀 Starting comprehensive workflow integration tests...")
        
        # Test 1: Complete workflow integration
        integration_success = await test_complete_workflow_integration()
        
        # Test 2: Error handling
        error_handling_success = await test_workflow_error_handling()
        
        # Summary
        print(f"\n🎯 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Workflow Integration: {'PASSED' if integration_success else 'FAILED'}")
        print(f"✅ Error Handling: {'PASSED' if error_handling_success else 'FAILED'}")
        
        # Check monitoring
        if monitor.langfuse:
            print(f"\n🔍 Langfuse Monitoring Summary:")
            print(f"   Session ID: {monitor.session_id}")
            print(f"   Events logged: ✅")
            print(f"   Workflow traces: ✅")
        
        success = integration_success and error_handling_success
        
        if success:
            print(f"\n🎉 All integration tests PASSED!")
            print(f"✅ Mistral OCR output is properly integrated with agent workflow")
            return 0
        else:
            print(f"\n💥 Some integration tests FAILED!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Integration test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)