#!/usr/bin/env python3
"""
Direct workflow orchestrator test
"""
import sys
import asyncio
sys.path.insert(0, '.')

from src.workflow_orchestrator import workflow_orchestrator

async def test_workflow():
    print("🔍 Testing workflow orchestrator directly...")
    
    # Create a small test PDF bytes (simple mock)
    test_pdf_bytes = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n149\n%%EOF'
    
    try:
        print(f"📄 Test document size: {len(test_pdf_bytes)} bytes")
        print("🚀 Calling workflow_orchestrator.process_complete_workflow()...")
        print()
        
        result = await workflow_orchestrator.process_complete_workflow(
            document_bytes=test_pdf_bytes,
            user_id="test_user",
            filename="test.pdf",
            document_type="clinical_document",
            use_mistral_ocr=True,  # Enable Mistral OCR
            use_advanced_llm=True,
            llm_model="codellama",
            generate_fhir=False  # Skip FHIR for this test
        )
        
        print("✅ Workflow completed successfully!")
        print(f"Result keys: {list(result.keys())}")
        
    except Exception as e:
        print(f"❌ Workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_workflow())