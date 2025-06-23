#!/usr/bin/env python3
"""
Test script to verify Ollama connectivity fixes
"""
import sys
import os
sys.path.append('.')

from src.enhanced_codellama_processor import EnhancedCodeLlamaProcessor
import asyncio

async def test_ollama_fix():
    print("🔥 Testing Enhanced CodeLlama Processor with Ollama fixes...")
    
    # Initialize processor
    processor = EnhancedCodeLlamaProcessor()
    
    # Test simple medical text
    test_text = "Patient has diabetes and hypertension. Blood pressure is 140/90."
    
    print(f"📝 Testing text: {test_text}")
    print("🔄 Processing...")
    
    try:
        result = await processor.process_document(
            medical_text=test_text,
            document_type="clinical_note",
            extract_entities=True,
            generate_fhir=False
        )
        
        print("✅ Processing successful!")
        print(f"📋 Provider used: {result.get('provider_metadata', {}).get('provider_used', 'Unknown')}")
        print(f"⏱️ Processing time: {result.get('provider_metadata', {}).get('processing_time', 'Unknown')}")
        print(f"🔍 Entities found: {result.get('extraction_results', {}).get('entities_found', 0)}")
        
        if result.get('extracted_data'):
            print("📊 Sample extracted data available")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ollama_fix())
    if success:
        print("\n🎉 Ollama connectivity fixes are working!")
        sys.exit(0)
    else:
        print("\n❌ Issues still exist")
        sys.exit(1)