#!/usr/bin/env python3
"""
Test Main App Mistral Integration
Test the actual workflow to see enhanced logging
"""

import asyncio
import os
import sys
import base64
from PIL import Image, ImageDraw
import io

# Add the app directory to the path for proper imports
sys.path.insert(0, '/app')

from src.workflow_orchestrator import WorkflowOrchestrator

async def test_main_app_mistral():
    """Test the main app with a sample document to see Mistral API logs"""
    
    print("🧪 Testing Main App Mistral Integration")
    print("=" * 50)
    
    # Create a test medical document
    print("📄 Creating test medical document...")
    test_image = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(test_image)
    
    # Add medical content
    draw.text((10, 10), "MEDICAL REPORT", fill='black')
    draw.text((10, 40), "Patient: Jane Smith", fill='black')
    draw.text((10, 70), "DOB: 02/15/1985", fill='black')
    draw.text((10, 100), "Diagnosis: Hypertension", fill='black')
    draw.text((10, 130), "Medication: Lisinopril 10mg", fill='black')
    draw.text((10, 160), "Blood Pressure: 140/90 mmHg", fill='black')
    draw.text((10, 190), "Provider: Dr. Johnson", fill='black')
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG', quality=95)
    document_bytes = img_byte_arr.getvalue()
    
    print(f"📊 Document size: {len(document_bytes)} bytes")
    
    # Initialize workflow orchestrator
    print("\n🔧 Initializing WorkflowOrchestrator...")
    orchestrator = WorkflowOrchestrator()
    
    # Test the workflow
    print("\n🚀 Testing workflow with enhanced logging...")
    try:
        result = await orchestrator.process_complete_workflow(
            document_bytes=document_bytes,
            user_id="test_user",
            filename="test_medical_report.jpg",
            use_mistral_ocr=True  # 🔥 EXPLICITLY ENABLE MISTRAL OCR
        )
        
        print("\n✅ Workflow completed successfully!")
        # Get correct field paths from workflow result structure
        text_extraction = result.get('text_extraction', {})
        medical_analysis = result.get('medical_analysis', {})
        workflow_metadata = result.get('workflow_metadata', {})
        
        print(f"📝 Extracted text length: {text_extraction.get('full_text_length', 0)}")
        print(f"🏥 Medical entities found: {medical_analysis.get('entities_found', 0)}")
        print(f"📋 FHIR bundle created: {'fhir_bundle' in result}")
        
        # Parse extracted data if available
        extracted_data_str = medical_analysis.get('extracted_data', '{}')
        try:
            import json
            entities = json.loads(extracted_data_str)
        except:
            entities = {}
            
        print(f"\n📊 Medical Entities:")
        print(f"  Patient: {entities.get('patient_name', 'N/A')}")
        print(f"  DOB: {entities.get('date_of_birth', 'N/A')}")
        print(f"  Provider: {entities.get('provider_name', 'N/A')}")
        print(f"  Conditions: {entities.get('conditions', [])}")
        print(f"  Medications: {entities.get('medications', [])}")
        
        # Check for OCR method used
        print(f"\n🔍 OCR method used: {workflow_metadata.get('ocr_method', 'Unknown')}")
        
        # Show extracted text preview
        extracted_text = text_extraction.get('extracted_text', '')
        if extracted_text:
            print(f"\n📄 Extracted text preview: {extracted_text[:200]}...")
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        print(f"📄 Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    print("🔍 Enhanced logging should show:")
    print("  - mistral_attempt_start")
    print("  - mistral_success_in_fallback OR mistral_fallback_failed")
    print("  - Detailed error traces if Mistral fails")
    print()
    
    asyncio.run(test_main_app_mistral())