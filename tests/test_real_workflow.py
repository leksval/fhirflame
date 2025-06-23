#!/usr/bin/env python3
"""
🚀 FhirFlame Real Workflow Demo
Testing CodeLlama 13B + Langfuse monitoring with real medical document processing
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add src to path (from tests directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.codellama_processor import CodeLlamaProcessor
from src.monitoring import monitor
from src.fhir_validator import FhirValidator

async def test_real_medical_workflow():
    """Demonstrate complete real medical AI workflow"""
    
    print("🔥 FhirFlame Real Workflow Demo")
    print("=" * 50)
    
    # Sample medical documents for testing
    medical_documents = [
        {
            "filename": "patient_smith.txt",
            "content": """
MEDICAL RECORD - CONFIDENTIAL

Patient: John Smith
DOB: 1975-03-15
MRN: MR789123

CHIEF COMPLAINT: Chest pain and shortness of breath

HISTORY OF PRESENT ILLNESS:
45-year-old male presents with acute onset chest pain radiating to left arm.
Associated with diaphoresis and nausea. No prior cardiac history.

VITAL SIGNS:
- Blood Pressure: 145/95 mmHg
- Heart Rate: 102 bpm  
- Temperature: 98.6°F
- Oxygen Saturation: 96% on room air

ASSESSMENT AND PLAN:
1. Acute coronary syndrome - rule out myocardial infarction
2. Hypertension - new diagnosis
3. Start aspirin 325mg daily
4. Lisinopril 10mg daily for blood pressure control
5. Atorvastatin 40mg daily

MEDICATIONS PRESCRIBED:
- Aspirin 325mg daily
- Lisinopril 10mg daily  
- Atorvastatin 40mg daily
- Nitroglycerin 0.4mg sublingual PRN chest pain
"""
        },
        {
            "filename": "diabetes_follow_up.txt", 
            "content": """
ENDOCRINOLOGY FOLLOW-UP NOTE

Patient: Maria Rodriguez
DOB: 1962-08-22
MRN: MR456789

DIAGNOSIS: Type 2 Diabetes Mellitus, well controlled

CURRENT MEDICATIONS:
- Metformin 1000mg twice daily
- Glipizide 5mg daily
- Insulin glargine 20 units at bedtime

LABORATORY RESULTS:
- HbA1c: 6.8% (target <7%)
- Fasting glucose: 126 mg/dL
- Creatinine: 1.0 mg/dL (normal kidney function)

VITAL SIGNS:
- Blood Pressure: 128/78 mmHg
- Weight: 165 lbs (stable)
- BMI: 28.5

ASSESSMENT:
Diabetes well controlled. Continue current regimen.
Recommend annual eye exam and podiatry follow-up.
"""
        }
    ]
    
    # Initialize processor with real Ollama
    print("\n🤖 Initializing CodeLlama processor...")
    processor = CodeLlamaProcessor()
    
    # Initialize FHIR validator
    print("📋 Initializing FHIR validator...")
    fhir_validator = FhirValidator()
    
    # Process each document
    results = []
    
    for i, doc in enumerate(medical_documents, 1):
        print(f"\n📄 Processing Document {i}/{len(medical_documents)}: {doc['filename']}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Process with real CodeLlama
            print("🔍 Analyzing with CodeLlama 13B-instruct...")
            result = await processor.process_document(
                medical_text=doc['content'],
                document_type="clinical_note", 
                extract_entities=True,
                generate_fhir=True
            )
            
            processing_time = time.time() - start_time
            
            # Display results
            print(f"✅ Processing completed in {processing_time:.2f}s")
            print(f"📊 Processing mode: {result['metadata']['model_used']}")
            print(f"🎯 Entities found: {result['extraction_results']['entities_found']}")
            print(f"📈 Quality score: {result['extraction_results']['quality_score']:.2f}")
            
            # Extract and display medical entities
            if 'extracted_data' in result:
                import json
                extracted = json.loads(result['extracted_data'])
                
                print("\n🏥 Extracted Medical Information:")
                print(f"   Patient: {extracted.get('patient', 'N/A')}")
                print(f"   Conditions: {', '.join(extracted.get('conditions', []))}")
                print(f"   Medications: {', '.join(extracted.get('medications', []))}")
                print(f"   Confidence: {extracted.get('confidence_score', 0):.1%}")
            
            # Validate FHIR bundle if generated
            if 'fhir_bundle' in result:
                print("\n📋 Validating FHIR bundle...")
                fhir_validation = fhir_validator.validate_fhir_bundle(result['fhir_bundle'])
                print(f"   FHIR R4 Valid: {fhir_validation['is_valid']}")
                print(f"   Compliance Score: {fhir_validation['compliance_score']:.1%}")
                print(f"   Validation Level: {fhir_validation['validation_level']}")
            
            results.append({
                'filename': doc['filename'],
                'processing_time': processing_time,
                'success': True,
                'result': result
            })
            
        except Exception as e:
            print(f"❌ Error processing {doc['filename']}: {e}")
            results.append({
                'filename': doc['filename'], 
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n🎯 WORKFLOW SUMMARY")
    print("=" * 50)
    successful = sum(1 for r in results if r['success'])
    total_time = sum(r.get('processing_time', 0) for r in results if r['success'])
    
    print(f"Documents processed: {successful}/{len(medical_documents)}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per document: {total_time/successful:.2f}s" if successful > 0 else "N/A")
    
    # Langfuse monitoring summary
    print(f"\n🔍 Langfuse Monitoring: {'✅ Active' if monitor.langfuse else '❌ Disabled'}")
    if monitor.langfuse:
        print(f"   Session ID: {monitor.session_id}")
        print(f"   Host: {os.getenv('LANGFUSE_HOST', 'cloud.langfuse.com')}")
    
    return results

async def main():
    """Main workflow execution"""
    from src.monitoring import monitor
    
    print(f"🕐 Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set environment for real processing
    os.environ['USE_REAL_OLLAMA'] = 'true'
    
    try:
        results = await test_real_medical_workflow()
        
        # Log comprehensive workflow summary using centralized monitoring
        successful = sum(1 for r in results if r['success'])
        total_time = sum(r.get('processing_time', 0) for r in results if r['success'])
        
        monitor.log_workflow_summary(
            documents_processed=len(results),
            successful_documents=successful,
            total_time=total_time,
            average_time=total_time/successful if successful > 0 else 0,
            monitoring_active=monitor.langfuse is not None
        )
        
        print("\n🎉 Real workflow demonstration completed successfully!")
        return 0
    except Exception as e:
        print(f"\n💥 Workflow failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)