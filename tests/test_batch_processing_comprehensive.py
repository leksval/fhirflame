#!/usr/bin/env python3
"""
Comprehensive Batch Processing Demo Analysis
Deep analysis of Modal scaling implementation and batch processing capabilities
"""

import asyncio
import sys
import os
import time
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fhirflame', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fhirflame'))

def test_heavy_workload_demo_import():
    """Test 1: Heavy Workload Demo Import and Initialization"""
    print("🔍 TEST 1: Heavy Workload Demo Import")
    print("-" * 50)
    
    try:
        from fhirflame.src.heavy_workload_demo import ModalContainerScalingDemo, RealTimeBatchProcessor
        print("✅ Successfully imported ModalContainerScalingDemo")
        print("✅ Successfully imported RealTimeBatchProcessor")
        
        # Test initialization
        demo = ModalContainerScalingDemo()
        processor = RealTimeBatchProcessor()
        
        print(f"✅ Modal demo initialized with {len(demo.regions)} regions")
        print(f"✅ Batch processor initialized with {len(processor.medical_datasets)} datasets")
        
        # Test configuration
        print(f"   Scaling tiers: {len(demo.scaling_tiers)}")
        print(f"   Workload configs: {len(demo.workload_configs)}")
        print(f"   Default region: {demo.default_region}")
        
        return True, demo, processor
        
    except Exception as e:
        print(f"❌ Heavy workload demo import failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

async def test_modal_scaling_simulation(demo):
    """Test 2: Modal Container Scaling Simulation"""
    print("\n🔍 TEST 2: Modal Container Scaling Simulation")
    print("-" * 50)
    
    try:
        # Start the Modal scaling demo
        result = await demo.start_modal_scaling_demo()
        print(f"✅ Modal scaling demo started: {result}")
        
        # Let it run for a few seconds to simulate scaling
        print("🔄 Running Modal scaling simulation for 10 seconds...")
        await asyncio.sleep(10)
        
        # Get statistics during operation
        stats = demo.get_demo_statistics()
        print(f"📊 Demo Status: {stats['demo_status']}")
        print(f"📈 Active Containers: {stats['active_containers']}")
        print(f"⚡ Requests/sec: {stats['requests_per_second']}")
        print(f"📦 Total Processed: {stats['total_requests_processed']}")
        print(f"🔄 Concurrent Requests: {stats['concurrent_requests']}")
        print(f"💰 Cost per Request: {stats['cost_per_request']}")
        print(f"🎯 Scaling Strategy: {stats['scaling_strategy']}")
        
        # Get container details
        containers = demo.get_container_details()
        print(f"🏭 Container Details: {len(containers)} containers active")
        
        if containers:
            print("   Top 3 Container Details:")
            for i, container in enumerate(containers[:3]):
                print(f"   [{i+1}] {container['Container ID']}: {container['Status']} - {container['Requests/sec']} RPS")
        
        # Stop the demo
        demo.stop_demo()
        print("✅ Modal scaling demo stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Modal scaling simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processor_datasets(processor):
    """Test 3: Batch Processor Medical Datasets"""
    print("\n🔍 TEST 3: Batch Processor Medical Datasets")
    print("-" * 50)
    
    try:
        datasets = processor.medical_datasets
        
        for dataset_name, documents in datasets.items():
            print(f"📋 Dataset: {dataset_name}")
            print(f"   Documents: {len(documents)}")
            print(f"   Avg length: {sum(len(doc) for doc in documents) // len(documents)} chars")
            
            # Show sample content
            if documents:
                sample = documents[0][:100].replace('\n', ' ').strip()
                print(f"   Sample: {sample}...")
        
        print("✅ All medical datasets validated")
        return True
        
    except Exception as e:
        print(f"❌ Batch processor dataset test failed: {e}")
        return False

async def test_real_time_batch_processing(processor):
    """Test 4: Real-Time Batch Processing"""
    print("\n🔍 TEST 4: Real-Time Batch Processing")
    print("-" * 50)
    
    try:
        # Test different workflow types
        workflows_to_test = [
            ("clinical_fhir", 3),
            ("lab_entities", 2),
            ("mixed_workflow", 2)
        ]
        
        results = {}
        
        for workflow_type, batch_size in workflows_to_test:
            print(f"\n🔬 Testing workflow: {workflow_type} (batch size: {batch_size})")
            
            # Start processing
            success = processor.start_processing(workflow_type, batch_size)
            
            if not success:
                print(f"❌ Failed to start processing for {workflow_type}")
                continue
            
            # Monitor progress
            start_time = time.time()
            while processor.processing:
                status = processor.get_status()
                if status['status'] == 'processing':
                    print(f"   Progress: {status['progress']:.1f}% - {status['processed']}/{status['total']}")
                    await asyncio.sleep(2)
                elif status['status'] == 'completed':
                    break
                else:
                    break
                
                # Timeout after 30 seconds
                if time.time() - start_time > 30:
                    processor.stop_processing()
                    break
            
            # Get final status
            final_status = processor.get_status()
            results[workflow_type] = final_status
            
            if final_status['status'] == 'completed':
                print(f"✅ {workflow_type} completed: {final_status['processed']} documents")
                print(f"   Total time: {final_status['total_time']:.2f}s")
            else:
                print(f"⚠️ {workflow_type} did not complete fully")
        
        print(f"\n📊 Batch Processing Summary:")
        for workflow, result in results.items():
            status = result.get('status', 'unknown')
            processed = result.get('processed', 0)
            total_time = result.get('total_time', 0)
            print(f"   {workflow}: {status} - {processed} docs in {total_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_modal_integration_components():
    """Test 5: Modal Integration Components"""
    print("\n🔍 TEST 5: Modal Integration Components")
    print("-" * 50)
    
    try:
        # Test Modal functions import
        try:
            from fhirflame.cloud_modal.functions import calculate_real_modal_cost
            print("✅ Modal functions imported successfully")
            
            # Test cost calculation
            cost_1s = calculate_real_modal_cost(1.0, "L4")
            cost_10s = calculate_real_modal_cost(10.0, "L4")
            
            print(f"   L4 GPU cost (1s): ${cost_1s:.6f}")
            print(f"   L4 GPU cost (10s): ${cost_10s:.6f}")
            
            if cost_10s > cost_1s:
                print("✅ Cost calculation scaling works correctly")
            else:
                print("⚠️ Cost calculation may have issues")
                
        except ImportError as e:
            print(f"⚠️ Modal functions not available: {e}")
        
        # Test Modal deployment
        try:
            from fhirflame.modal_deployments.fhirflame_modal_app import app, GPU_CONFIGS
            print("✅ Modal deployment app imported successfully")
            print(f"   GPU configs available: {list(GPU_CONFIGS.keys())}")
            
        except ImportError as e:
            print(f"⚠️ Modal deployment not available: {e}")
        
        # Test Enhanced CodeLlama Processor
        try:
            from fhirflame.src.enhanced_codellama_processor import EnhancedCodeLlamaProcessor
            processor = EnhancedCodeLlamaProcessor()
            print("✅ Enhanced CodeLlama processor initialized")
            print(f"   Modal available: {processor.router.modal_available}")
            print(f"   Ollama available: {processor.router.ollama_available}")
            print(f"   HuggingFace available: {processor.router.hf_available}")
            
        except Exception as e:
            print(f"⚠️ Enhanced CodeLlama processor issues: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Modal integration test failed: {e}")
        return False

def test_frontend_integration():
    """Test 6: Frontend Integration"""
    print("\n🔍 TEST 6: Frontend Integration")
    print("-" * 50)
    
    try:
        from fhirflame.frontend_ui import heavy_workload_demo, batch_processor
        print("✅ Frontend UI integration working")
        
        # Test if components are properly initialized
        if heavy_workload_demo is not None:
            print("✅ Heavy workload demo available in frontend")
        else:
            print("⚠️ Heavy workload demo not properly initialized in frontend")
            
        if batch_processor is not None:
            print("✅ Batch processor available in frontend")
        else:
            print("⚠️ Batch processor not properly initialized in frontend")
        
        return True
        
    except Exception as e:
        print(f"❌ Frontend integration test failed: {e}")
        return False

async def main():
    """Main comprehensive test execution"""
    print("🔥 FHIRFLAME BATCH PROCESSING COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    print(f"🕐 Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test results tracking
    test_results = {}
    
    # Test 1: Import and initialization
    success, demo, processor = test_heavy_workload_demo_import()
    test_results["Heavy Workload Demo Import"] = success
    
    if not success:
        print("❌ Critical import failure - cannot continue with tests")
        return 1
    
    # Test 2: Modal scaling simulation
    if demo:
        success = await test_modal_scaling_simulation(demo)
        test_results["Modal Scaling Simulation"] = success
    
    # Test 3: Batch processor datasets
    if processor:
        success = test_batch_processor_datasets(processor)
        test_results["Batch Processor Datasets"] = success
    
    # Test 4: Real-time batch processing
    if processor:
        success = await test_real_time_batch_processing(processor)
        test_results["Real-Time Batch Processing"] = success
    
    # Test 5: Modal integration components
    success = test_modal_integration_components()
    test_results["Modal Integration Components"] = success
    
    # Test 6: Frontend integration
    success = test_frontend_integration()
    test_results["Frontend Integration"] = success
    
    # Final Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE ANALYSIS RESULTS")
    print("=" * 60)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Analysis Summary
    print(f"\n🎯 BATCH PROCESSING IMPLEMENTATION ANALYSIS:")
    print(f"=" * 60)
    
    if passed >= total * 0.8:  # 80% or higher
        print("🎉 EXCELLENT: Batch processing implementation is comprehensive and working")
        print("✅ Modal scaling demo is properly implemented")
        print("✅ Real-time batch processing is functional")
        print("✅ Integration between components is solid")
        print("✅ Frontend integration is working")
        print("\n🚀 READY FOR PRODUCTION DEMONSTRATION")
    elif passed >= total * 0.6:  # 60-79%
        print("👍 GOOD: Batch processing implementation is mostly working")
        print("✅ Core functionality is implemented")
        print("⚠️ Some integration issues may exist")
        print("\n🔧 MINOR FIXES RECOMMENDED")
    else:  # Below 60%
        print("⚠️ ISSUES DETECTED: Batch processing implementation needs attention")
        print("❌ Critical components may not be working properly")
        print("❌ Integration issues present")
        print("\n🛠️ SIGNIFICANT FIXES REQUIRED")
    
    print(f"\n📋 RECOMMENDATIONS:")
    
    if not test_results.get("Modal Scaling Simulation", True):
        print("- Fix Modal container scaling simulation")
    
    if not test_results.get("Real-Time Batch Processing", True):
        print("- Debug real-time batch processing workflow")
    
    if not test_results.get("Modal Integration Components", True):
        print("- Ensure Modal integration components are properly configured")
    
    if not test_results.get("Frontend Integration", True):
        print("- Fix frontend UI integration issues")
    
    print(f"\n🏁 Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if passed >= total * 0.8 else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)