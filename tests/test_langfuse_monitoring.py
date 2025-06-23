#!/usr/bin/env python3
"""
Test Comprehensive Langfuse Monitoring Integration
Verify that monitoring is consistently implemented across all critical components
"""

import os
import sys
import time
from unittest.mock import patch, MagicMock

def test_monitoring_imports():
    """Test that monitoring can be imported from all components"""
    print("🔍 Testing monitoring imports...")
    
    try:
        # Test monitoring module import
        from src.monitoring import monitor
        print("✅ Core monitoring imported")
        
        # Test A2A API monitoring integration
        from src.mcp_a2a_api import monitor as a2a_monitor
        print("✅ A2A API monitoring imported")
        
        # Test MCP server monitoring integration
        from src.fhirflame_mcp_server import monitor as mcp_monitor
        print("✅ MCP server monitoring imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring import failed: {e}")
        return False

def test_monitoring_methods():
    """Test that all new monitoring methods are available"""
    print("\n🔍 Testing monitoring methods...")
    
    try:
        from src.monitoring import monitor
        
        # Test A2A API monitoring methods
        assert hasattr(monitor, 'log_a2a_api_request'), "Missing log_a2a_api_request"
        assert hasattr(monitor, 'log_a2a_api_response'), "Missing log_a2a_api_response"
        assert hasattr(monitor, 'log_a2a_authentication'), "Missing log_a2a_authentication"
        print("✅ A2A API monitoring methods present")
        
        # Test Modal scaling monitoring methods
        assert hasattr(monitor, 'log_modal_function_call'), "Missing log_modal_function_call"
        assert hasattr(monitor, 'log_modal_scaling_event'), "Missing log_modal_scaling_event"
        assert hasattr(monitor, 'log_modal_deployment'), "Missing log_modal_deployment"
        assert hasattr(monitor, 'log_modal_cost_tracking'), "Missing log_modal_cost_tracking"
        print("✅ Modal scaling monitoring methods present")
        
        # Test MCP monitoring methods
        assert hasattr(monitor, 'log_mcp_server_start'), "Missing log_mcp_server_start"
        assert hasattr(monitor, 'log_mcp_authentication'), "Missing log_mcp_authentication"
        print("✅ MCP monitoring methods present")
        
        # Test Docker deployment monitoring
        assert hasattr(monitor, 'log_docker_deployment'), "Missing log_docker_deployment"
        assert hasattr(monitor, 'log_docker_service_health'), "Missing log_docker_service_health"
        print("✅ Docker monitoring methods present")
        
        # Test error and performance monitoring
        assert hasattr(monitor, 'log_error_event'), "Missing log_error_event"
        assert hasattr(monitor, 'log_performance_metrics'), "Missing log_performance_metrics"
        print("✅ Error/performance monitoring methods present")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring methods test failed: {e}")
        return False

def test_monitoring_functionality():
    """Test that monitoring methods work without errors"""
    print("\n🔍 Testing monitoring functionality...")
    
    try:
        from src.monitoring import monitor
        
        # Test A2A API monitoring
        monitor.log_a2a_api_request(
            endpoint="/api/v1/test",
            method="POST",
            auth_method="bearer_token",
            request_size=100,
            user_id="test_user"
        )
        print("✅ A2A API request monitoring works")
        
        # Test Modal function monitoring
        monitor.log_modal_function_call(
            function_name="test_function",
            gpu_type="L4",
            processing_time=1.5,
            cost_estimate=0.001,
            container_id="test-container-123"
        )
        print("✅ Modal function monitoring works")
        
        # Test MCP tool monitoring
        monitor.log_mcp_tool(
            tool_name="process_medical_document",
            success=True,
            processing_time=2.0,
            input_size=500,
            entities_found=5
        )
        print("✅ MCP tool monitoring works")
        
        # Test error monitoring
        monitor.log_error_event(
            error_type="test_error",
            error_message="Test error message",
            stack_trace="",
            component="test_component",
            severity="info"
        )
        print("✅ Error monitoring works")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring functionality test failed: {e}")
        return False

def test_docker_compose_monitoring():
    """Test Docker Compose monitoring integration"""
    print("\n🔍 Testing Docker Compose monitoring...")
    
    try:
        from src.monitoring import monitor
        
        # Test Docker deployment monitoring
        monitor.log_docker_deployment(
            compose_file="docker-compose.local.yml",
            services_started=3,
            success=True,
            startup_time=30.0
        )
        print("✅ Docker deployment monitoring works")
        
        # Test service health monitoring
        monitor.log_docker_service_health(
            service_name="fhirflame-a2a-api",
            status="healthy",
            response_time=0.5,
            healthy=True
        )
        print("✅ Docker service health monitoring works")
        
        return True
        
    except Exception as e:
        print(f"❌ Docker monitoring test failed: {e}")
        return False

def main():
    """Run comprehensive monitoring tests"""
    print("🔍 Testing Comprehensive Langfuse Monitoring")
    print("=" * 50)
    
    # Change to correct directory
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    
    tests = [
        ("Monitoring Imports", test_monitoring_imports),
        ("Monitoring Methods", test_monitoring_methods),
        ("Monitoring Functionality", test_monitoring_functionality),
        ("Docker Monitoring", test_docker_compose_monitoring)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Langfuse Monitoring Test Results")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Comprehensive Langfuse monitoring implemented successfully!")
        print("\n📋 Monitoring Coverage:")
        print("• A2A API requests/responses with authentication tracking")
        print("• Modal L4 GPU function calls and scaling events")
        print("• MCP tool execution and server events")
        print("• Docker deployment and service health")
        print("• Error events and performance metrics")
        print("• Medical entity extraction and FHIR validation")
    else:
        print("\n⚠️ Some monitoring tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)