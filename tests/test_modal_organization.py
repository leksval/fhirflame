#!/usr/bin/env python3
"""
Test: Modal Organization and Structure
Test that the organized Modal files structure works correctly
"""

import os
import sys
import importlib

def test_modal_imports():
    """Test that Modal functions can be imported from organized structure"""
    print("🔍 Test: Modal Import Structure")
    
    try:
        # Add current directory to Python path
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        
        # Test that modal.functions can be imported
        from modal import functions
        print("✅ Modal functions module imported")
        
        # Test that modal.config can be imported
        from modal import config
        print("✅ Modal config module imported")
        
        # Test that specific functions exist
        assert hasattr(functions, 'app'), "Modal app not found"
        assert hasattr(functions, 'calculate_real_modal_cost'), "Cost calculation function not found"
        
        print("✅ Modal functions accessible")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Modal import test failed: {e}")
        return False

def test_deployment_files():
    """Test that deployment files exist and are accessible"""
    print("\n🔍 Test: Deployment Files")
    
    try:
        # Check modal deployment file
        modal_deploy_path = "modal/deploy.py"
        assert os.path.exists(modal_deploy_path), f"Modal deploy file not found: {modal_deploy_path}"
        print("✅ Modal deployment file exists")
        
        # Check local deployment file
        local_deploy_path = "deploy_local.py"
        assert os.path.exists(local_deploy_path), f"Local deploy file not found: {local_deploy_path}"
        print("✅ Local deployment file exists")
        
        # Check main README
        readme_path = "README.md"
        assert os.path.exists(readme_path), f"Main README not found: {readme_path}"
        print("✅ Main README exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment files test failed: {e}")
        return False

def test_environment_config():
    """Test environment configuration"""
    print("\n🔍 Test: Environment Configuration")
    
    try:
        # Test environment variables
        modal_token_id = os.getenv("MODAL_TOKEN_ID", "")
        modal_token_secret = os.getenv("MODAL_TOKEN_SECRET", "")
        
        if modal_token_id and modal_token_secret:
            print("✅ Modal tokens configured")
        else:
            print("⚠️ Modal tokens not configured (expected for tests)")
        
        # Test cost configuration
        l4_rate = float(os.getenv("MODAL_L4_HOURLY_RATE", "0.73"))
        platform_fee = float(os.getenv("MODAL_PLATFORM_FEE", "15"))
        
        assert l4_rate > 0, "L4 rate should be positive"
        assert platform_fee > 0, "Platform fee should be positive"
        
        print(f"✅ L4 Rate: ${l4_rate}/hour")
        print(f"✅ Platform Fee: {platform_fee}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment config test failed: {e}")
        return False

def test_cost_calculation():
    """Test cost calculation function"""
    print("\n🔍 Test: Cost Calculation Function")
    
    try:
        # Add current directory to Python path
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        
        from modal.functions import calculate_real_modal_cost
        
        # Test L4 cost calculation
        cost_l4_1s = calculate_real_modal_cost(1.0, "L4")
        cost_l4_10s = calculate_real_modal_cost(10.0, "L4")
        
        assert cost_l4_1s > 0, "L4 cost should be positive"
        assert cost_l4_10s > cost_l4_1s, "10s should cost more than 1s"
        
        print(f"✅ L4 1s cost: ${cost_l4_1s:.6f}")
        print(f"✅ L4 10s cost: ${cost_l4_10s:.6f}")
        
        # Test CPU cost calculation
        cost_cpu = calculate_real_modal_cost(1.0, "CPU")
        assert cost_cpu >= 0, "CPU cost should be non-negative"
        
        print(f"✅ CPU 1s cost: ${cost_cpu:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cost calculation test failed: {e}")
        return False

def main():
    """Run organization tests"""
    print("🚀 Testing Modal Organization Structure")
    print("=" * 50)
    
    tests = [
        ("Modal Imports", test_modal_imports),
        ("Deployment Files", test_deployment_files),
        ("Environment Config", test_environment_config),
        ("Cost Calculation", test_cost_calculation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Organization Test Results")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Modal organization structure is working!")
        print("\n📋 Ready for deployment:")
        print("1. Modal production: python modal/deploy.py")
        print("2. Local development: python deploy_local.py")
    else:
        print("⚠️ Some organization tests failed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)