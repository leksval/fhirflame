#!/usr/bin/env python3
"""
Test the Processing Queue Implementation
Quick test to verify the processing queue interface works
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_processing_queue():
    """Test the processing queue functionality"""
    print("🧪 Testing Processing Queue Implementation...")
    
    try:
        # Import the processing queue components
        from frontend_ui import ProcessingQueue, ProcessingJob, processing_queue
        print("✅ Successfully imported processing queue components")
        
        # Test queue initialization
        assert len(processing_queue.jobs) > 0, "Queue should have demo data"
        print(f"✅ Queue initialized with {len(processing_queue.jobs)} demo jobs")
        
        # Test adding a new job
        test_job = processing_queue.add_job("test_document.pdf", "Text Processing")
        assert test_job.document_name == "test_document.pdf"
        print("✅ Successfully added new job to queue")
        
        # Test updating job completion
        processing_queue.update_job(test_job, True, "Test AI Model", 5)
        assert test_job.success == True
        assert test_job.entities_found == 5
        print("✅ Successfully updated job completion status")
        
        # Test getting queue as DataFrame
        df = processing_queue.get_queue_dataframe()
        assert len(df) > 0, "DataFrame should have data"
        print(f"✅ Successfully generated DataFrame with {len(df)} rows")
        
        # Test getting session statistics
        stats = processing_queue.get_session_statistics()
        assert "total_processed" in stats
        assert "avg_processing_time" in stats
        print("✅ Successfully generated session statistics")
        
        print("\n🎉 All processing queue tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Processing queue test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradio_interface():
    """Test that the Gradio interface can be created"""
    print("\n🎨 Testing Gradio Interface Creation...")
    
    try:
        import gradio as gr
        from frontend_ui import create_processing_queue_tab
        
        # Test creating the processing queue tab
        with gr.Blocks() as test_interface:
            queue_components = create_processing_queue_tab()
        
        assert "queue_df" in queue_components
        assert "stats_json" in queue_components
        print("✅ Successfully created processing queue Gradio interface")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradio interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_functions():
    """Test the workflow integration functions"""
    print("\n🔗 Testing Workflow Integration...")
    
    try:
        from frontend_ui import integrate_with_workflow, complete_workflow_job
        
        # Test integration
        job = integrate_with_workflow("integration_test.txt", "Integration Test")
        assert job.document_name == "integration_test.txt"
        print("✅ Successfully integrated with workflow")
        
        # Test completion
        complete_workflow_job(job, True, "Integration AI", 10)
        assert job.success == True
        print("✅ Successfully completed workflow job")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔥 FhirFlame Processing Queue Test Suite")
    print("=" * 50)
    
    tests = [
        ("Processing Queue Core", test_processing_queue),
        ("Gradio Interface", test_gradio_interface),
        ("Workflow Integration", test_integration_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Processing Queue is ready!")
        print("\n🚀 To see the processing queue in action:")
        print("   1. Run: python app.py")
        print("   2. Navigate to the '🔄 Processing Queue' tab")
        print("   3. Click 'Add Demo Job' to see real-time updates")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())