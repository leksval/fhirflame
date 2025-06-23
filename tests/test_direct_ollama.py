#!/usr/bin/env python3
"""
Direct Ollama CodeLlama Test - bypassing Docker network limitations
"""

import asyncio
import httpx
import json
import time

async def test_direct_codellama():
    """Test CodeLlama directly for medical entity extraction"""
    
    print("🚀 Direct CodeLlama Medical AI Test")
    print("=" * 40)
    
    medical_text = """
MEDICAL RECORD
Patient: Sarah Johnson
DOB: 1985-09-12
Chief Complaint: Type 2 diabetes follow-up

Current Medications:
- Metformin 1000mg twice daily
- Insulin glargine 15 units at bedtime  
- Lisinopril 10mg daily for hypertension

Vital Signs:
- Blood Pressure: 142/88 mmHg
- HbA1c: 7.2%
- Fasting glucose: 145 mg/dL

Assessment: Diabetes with suboptimal control, hypertension
"""

    prompt = f"""You are a medical AI assistant. Extract medical information from this clinical note and return ONLY a JSON response:

{medical_text}

Return this exact JSON structure:
{{
    "patient_info": "patient name if found",
    "conditions": ["list", "of", "conditions"],
    "medications": ["list", "of", "medications"],
    "vitals": ["list", "of", "vital", "measurements"],
    "confidence_score": 0.85
}}"""

    print("📋 Processing medical text with CodeLlama 13B...")
    print(f"📄 Input length: {len(medical_text)} characters")
    
    start_time = time.time()
    
    try:
        # Use host.docker.internal for Docker networking on Windows
        import os
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": "codellama:13b-instruct",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 1024
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                print(f"✅ CodeLlama processing completed!")
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                print(f"🧠 Model: {result.get('model', 'Unknown')}")
                
                generated_text = result.get("response", "")
                print(f"📝 Raw response length: {len(generated_text)} characters")
                
                # Try to parse JSON from response
                try:
                    json_start = generated_text.find('{')
                    json_end = generated_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = generated_text[json_start:json_end]
                        extracted_data = json.loads(json_str)
                        
                        print("\n🏥 EXTRACTED MEDICAL DATA:")
                        print(f"   Patient: {extracted_data.get('patient_info', 'N/A')}")
                        print(f"   Conditions: {', '.join(extracted_data.get('conditions', []))}")
                        print(f"   Medications: {', '.join(extracted_data.get('medications', []))}")
                        print(f"   Vitals: {', '.join(extracted_data.get('vitals', []))}")
                        print(f"   AI Confidence: {extracted_data.get('confidence_score', 0):.1%}")
                        
                        return True
                    else:
                        print("⚠️ No valid JSON found in response")
                        print(f"Raw response preview: {generated_text[:200]}...")
                        return False
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {e}")
                    print(f"Raw response preview: {generated_text[:200]}...")
                    return False
            else:
                print(f"❌ Ollama API error: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"💥 Connection failed: {e}")
        print("💡 Make sure 'ollama serve' is running")
        return False

async def main():
    success = await test_direct_codellama()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)