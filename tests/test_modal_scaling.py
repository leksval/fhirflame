#!/usr/bin/env python3
"""
Quick Test: Modal Scaling Implementation
Test the key components of our 3-prompt implementation
"""

import asyncio
import os
import sys
import time

def test_environment_config():
    """Test 1: Environment configuration"""
    print("🔍 Test 1: Environment Configuration")
    
    # Test cost configuration loading
    a100_rate = float(os.getenv("MODAL_A100_HOURLY_RATE", "1.32"))
    t4_rate = float(os.getenv("MODAL_T4_HOURLY_RATE", "0.51"))
    platform_fee = float(os.getenv("MODAL_PLATFORM_FEE", "15"))
    
    print(f"✅ A100 Rate: ${a100_rate}/hour")
    print(f"✅ T4 Rate: ${t4_rate}/hour") 
    print(f"✅ Platform Fee: {platform_fee}%")
    
    assert a100_rate > 0 and t4_rate > 0 and platform_fee > 0
    return True

def test_cost_calculation():
    """Test 2: Real cost calculation"""
    print("\n🔍 Test 2: Cost Calculation")
    
    try:
        from src.enhanced_codellama_processor import EnhancedCodeLlamaProcessor, InferenceProvider
        
        processor = EnhancedCodeLlamaProcessor()
        
        # Test different scenarios
        test_cases = [
            ("Short text", "Patient has diabetes", 0.5, "T4"),
            ("Long text", "Patient has diabetes. " * 100, 1.2, "A100"),
            ("Ollama local", "Test text", 0.8, None)
        ]
        
        for name, text, proc_time, gpu_type in test_cases:
            # Test Modal cost
            modal_cost = processor._calculate_cost(
                InferenceProvider.MODAL, len(text), proc_time, gpu_type
            )
            
            # Test Ollama cost
            ollama_cost = processor._calculate_cost(
                InferenceProvider.OLLAMA, len(text)
            )
            
            # Test HuggingFace cost
            hf_cost = processor._calculate_cost(
                InferenceProvider.HUGGINGFACE, len(text)
            )
            
            print(f"  {name}:")
            print(f"    Modal ({gpu_type}): ${modal_cost:.6f}")
            print(f"    Ollama: ${ollama_cost:.6f}")
            print(f"    HuggingFace: ${hf_cost:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cost calculation test failed: {e}")
        return False

async def test_modal_integration():
    """Test 3: Modal integration"""
    print("\n🔍 Test 3: Modal Integration")
    
    try:
        from src.enhanced_codellama_processor import EnhancedCodeLlamaProcessor
        
        processor = EnhancedCodeLlamaProcessor()
        
        # Test with simulation (since Modal endpoint may not be deployed)
        test_text = """
        Patient John Doe, 45 years old, presents with chest pain.
        Diagnosed with acute myocardial infarction.
        Treatment: Aspirin 325mg, Metoprolol 25mg BID.
        """
        
        result = await processor._call_modal_api(
            text=test_text,
            document_type="clinical_note",
            extract_entities=True,
            generate_fhir=False
        )
        
        print("✅ Modal API call completed")
        
        # Check result structure
        if "scaling_metadata" in result:
            scaling = result["scaling_metadata"]
            print(f"✅ Provider: {scaling.get('provider', 'unknown')}")
            print(f"✅ Cost: ${scaling.get('cost_estimate', 0):.6f}")
            print(f"✅ Container: {scaling.get('container_id', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Modal integration test failed: {e}")
        return False

def test_modal_deployment():
    """Test 4: Modal deployment file"""
    print("\n🔍 Test 4: Modal Deployment")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from modal.functions import calculate_real_modal_cost
        
        # Test cost calculation function for L4 (RTX 4090 equivalent)
        cost_l4 = calculate_real_modal_cost(1.0, "L4")
        cost_cpu = calculate_real_modal_cost(1.0, "CPU")
        
        print(f"✅ L4 GPU 1s cost: ${cost_l4:.6f}")
        print(f"✅ CPU 1s cost: ${cost_cpu:.6f}")
        
        # Verify L4 is more expensive than CPU
        if cost_l4 > cost_cpu:
            print("✅ Cost hierarchy correct (L4 > CPU)")
            return True
        else:
            print("⚠️ Cost hierarchy issue")
            return False
        
    except Exception as e:
        print(f"❌ Modal deployment test failed: {e}")
        return False

async def test_end_to_end():
    """Test 5: End-to-end scaling demo"""
    print("\n🔍 Test 5: End-to-End Demo")
    
    try:
        from src.enhanced_codellama_processor import EnhancedCodeLlamaProcessor
        
        processor = EnhancedCodeLlamaProcessor()
        
        # Test auto-selection logic
        short_text = "Patient has hypertension"
        long_text = "Patient John Doe presents with chest pain. " * 30
        
        # Test provider selection
        short_provider = processor.router.select_optimal_provider(short_text)
        long_provider = processor.router.select_optimal_provider(long_text)
        
        print(f"✅ Short text → {short_provider.value}")
        print(f"✅ Long text → {long_provider.value}")
        
        # Test processing with cost calculation
        result = await processor.process_document(
            medical_text=long_text,
            document_type="clinical_note",
            extract_entities=True,
            generate_fhir=False,
            complexity="medium"
        )
        
        if result and "provider_metadata" in result:
            meta = result["provider_metadata"]
            print(f"✅ Processed with: {meta.get('provider_used', 'unknown')}")
            print(f"✅ Cost estimate: ${meta.get('cost_estimate', 0):.6f}")
            print(f"✅ Processing time: {meta.get('processing_time', 0):.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False

async def main():
    """Run focused tests"""
    print("🚀 Testing Modal Scaling Implementation")
    print("=" * 50)
    
    tests = [
        ("Environment Config", test_environment_config),
        ("Cost Calculation", test_cost_calculation),
        ("Modal Integration", test_modal_integration),
        ("Modal Deployment", test_modal_deployment),
        ("End-to-End Demo", test_end_to_end)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Modal scaling implementation is working!")
        print("\n📋 Next Steps:")
        print("1. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env")
        print("2. Deploy: modal deploy modal_deployment.py")
        print("3. Set MODAL_ENDPOINT_URL in .env")
        print("4. Test Dynamic Scaling tab in Gradio UI")
    else:
        print("⚠️ Some tests failed. Check the details above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())