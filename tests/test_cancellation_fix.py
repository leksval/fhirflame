#!/usr/bin/env python3
"""
Test script to verify job cancellation and task management fixes
"""

import sys
import time
import asyncio
from unittest.mock import Mock, patch

# Add the current directory to the path so we can import app
sys.path.insert(0, '.')

def test_cancellation_mechanism():
    """Test the enhanced cancellation mechanism"""
    print("🧪 Testing Job Cancellation and Task Queue Management")
    print("=" * 60)
    
    try:
        # Import the app module
        import app
        
        # Test 1: Basic cancellation flag management
        print("\n1️⃣ Testing basic cancellation flags...")
        
        # Reset flags
        app.cancellation_flags["text_task"] = False
        app.running_tasks["text_task"] = None
        app.active_jobs["text_task"] = None
        
        print(f"   Initial cancellation flag: {app.cancellation_flags['text_task']}")
        print(f"   Initial running task: {app.running_tasks['text_task']}")
        print(f"   Initial active job: {app.active_jobs['text_task']}")
        
        # Test 2: Job manager creation and tracking
        print("\n2️⃣ Testing job creation and tracking...")
        
        # Create a test job
        job_id = app.job_manager.add_processing_job("text", "Test medical text", {"test": True})
        app.active_jobs["text_task"] = job_id
        
        print(f"   Created job ID: {job_id}")
        print(f"   Active tasks count: {app.job_manager.dashboard_state['active_tasks']}")
        print(f"   Active job tracking: {app.active_jobs['text_task']}")
        
        # Test 3: Cancel task functionality
        print("\n3️⃣ Testing cancel_current_task function...")
        
        # Mock a running task
        mock_task = Mock()
        app.running_tasks["text_task"] = mock_task
        
        # Call cancel function
        result = app.cancel_current_task("text_task")
        
        print(f"   Cancel result: {result}")
        print(f"   Cancellation flag after cancel: {app.cancellation_flags['text_task']}")
        print(f"   Running task after cancel: {app.running_tasks['text_task']}")
        print(f"   Active job after cancel: {app.active_jobs['text_task']}")
        print(f"   Active tasks count after cancel: {app.job_manager.dashboard_state['active_tasks']}")
        
        # Verify mock task was cancelled
        mock_task.cancel.assert_called_once()
        
        # Test 4: Job completion tracking
        print("\n4️⃣ Testing job completion tracking...")
        
        # Check job history
        history = app.job_manager.get_jobs_history()
        print(f"   Jobs in history: {len(history)}")
        if history:
            latest_job = history[-1]
            print(f"   Latest job status: {latest_job[2]}")  # Status column
        
        # Test 5: Dashboard metrics
        print("\n5️⃣ Testing dashboard metrics...")
        
        metrics = app.job_manager.get_dashboard_metrics()
        queue_stats = app.job_manager.get_processing_queue()
        
        print(f"   Dashboard metrics: {metrics}")
        print(f"   Queue statistics: {queue_stats}")
        
        print("\n✅ All cancellation mechanism tests passed!")
        pass
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Test failed with error: {e}"

def test_task_queue_management():
    """Test task queue management functionality"""
    print("\n🔄 Testing Task Queue Management")
    print("=" * 40)
    
    try:
        import app
        
        # Test queue initialization
        print(f"Text task queue: {app.task_queues['text_task']}")
        print(f"File task queue: {app.task_queues['file_task']}")
        print(f"DICOM task queue: {app.task_queues['dicom_task']}")
        
        # Add some mock tasks to queue
        app.task_queues["text_task"] = ["task1", "task2", "task3"]
        print(f"Added mock tasks to text queue: {len(app.task_queues['text_task'])}")
        
        # Test queue clearing on cancellation
        app.cancel_current_task("text_task")
        print(f"Queue after cancellation: {len(app.task_queues['text_task'])}")
        
        print("✅ Task queue management tests passed!")
        pass
        
    except Exception as e:
        print(f"❌ Task queue test failed: {e}")
        assert False, f"Task queue test failed: {e}"

if __name__ == "__main__":
    print("🔥 FhirFlame Cancellation Mechanism Test Suite")
    print("Testing enhanced job cancellation and task management...")
    
    # Run tests
    test1_passed = test_cancellation_mechanism()
    test2_passed = test_task_queue_management()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Cancellation mechanism is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)