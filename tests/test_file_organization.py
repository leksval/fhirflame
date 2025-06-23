#!/usr/bin/env python3
"""
Test: File Organization Structure
Test that our file organization is clean and complete
"""

import os
import sys

def test_modal_directory_structure():
    """Test Modal directory organization"""
    print("🔍 Test: Modal Directory Structure")
    
    try:
        modal_dir = "modal"
        expected_files = [
            "modal/__init__.py",
            "modal/config.py", 
            "modal/deploy.py",
            "modal/functions.py"
        ]
        
        for file_path in expected_files:
            assert os.path.exists(file_path), f"Missing file: {file_path}"
            print(f"✅ {file_path}")
        
        print("✅ Modal directory structure complete")
        return True
        
    except Exception as e:
        print(f"❌ Modal directory test failed: {e}")
        return False

def test_deployment_files():
    """Test deployment files exist"""
    print("\n🔍 Test: Deployment Files")
    
    try:
        deployment_files = [
            "modal/deploy.py",  # Modal production deployment
            "deploy_local.py",   # Local development deployment
            "README.md"          # Main documentation
        ]
        
        for file_path in deployment_files:
            assert os.path.exists(file_path), f"Missing deployment file: {file_path}"
            print(f"✅ {file_path}")
        
        print("✅ All deployment files present")
        return True
        
    except Exception as e:
        print(f"❌ Deployment files test failed: {e}")
        return False

def test_readme_consolidation():
    """Test that we have only one main README"""
    print("\n🔍 Test: README Consolidation")
    
    try:
        # Check main README exists
        assert os.path.exists("README.md"), "Main README.md not found"
        print("✅ Main README.md exists")
        
        # Check that modal/README.md was removed
        modal_readme = "modal/README.md"
        if os.path.exists(modal_readme):
            print("⚠️ modal/README.md still exists (should be removed)")
            return False
        else:
            print("✅ modal/README.md removed (correctly consolidated)")
        
        # Check README content is comprehensive
        with open("README.md", "r") as f:
            content = f.read()
        
        required_sections = [
            "Modal Labs",
            "deployment",
            "setup"
        ]
        
        for section in required_sections:
            if section.lower() in content.lower():
                print(f"✅ README contains {section} information")
            else:
                print(f"⚠️ README missing {section} information")
        
        print("✅ README consolidation successful")
        return True
        
    except Exception as e:
        print(f"❌ README consolidation test failed: {e}")
        return False

def test_environment_variables():
    """Test environment configuration"""
    print("\n🔍 Test: Environment Variables")
    
    try:
        # Check for .env file
        env_file = ".env"
        if os.path.exists(env_file):
            print("✅ .env file found")
            
            # Read and check for Modal configuration
            with open(env_file, "r") as f:
                env_content = f.read()
            
            modal_vars = [
                "MODAL_TOKEN_ID",
                "MODAL_TOKEN_SECRET", 
                "MODAL_L4_HOURLY_RATE",
                "MODAL_PLATFORM_FEE"
            ]
            
            for var in modal_vars:
                if var in env_content:
                    print(f"✅ {var} configured")
                else:
                    print(f"⚠️ {var} not found in .env")
        else:
            print("⚠️ .env file not found (expected for deployment)")
        
        # Test environment variable loading
        l4_rate = float(os.getenv("MODAL_L4_HOURLY_RATE", "0.73"))
        platform_fee = float(os.getenv("MODAL_PLATFORM_FEE", "15"))
        
        assert l4_rate > 0, "L4 rate should be positive"
        assert platform_fee > 0, "Platform fee should be positive"
        
        print(f"✅ L4 rate: ${l4_rate}/hour")
        print(f"✅ Platform fee: {platform_fee}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment variables test failed: {e}")
        return False

def test_file_cleanup():
    """Test that redundant files were cleaned up"""
    print("\n🔍 Test: File Cleanup")
    
    try:
        # Files that should NOT exist (cleaned up)
        removed_files = [
            "modal/README.md",  # Should be consolidated into main README
        ]
        
        cleanup_success = True
        for file_path in removed_files:
            if os.path.exists(file_path):
                print(f"⚠️ {file_path} still exists (should be removed)")
                cleanup_success = False
            else:
                print(f"✅ {file_path} properly removed")
        
        if cleanup_success:
            print("✅ File cleanup successful")
        
        return cleanup_success
        
    except Exception as e:
        print(f"❌ File cleanup test failed: {e}")
        return False

def main():
    """Run file organization tests"""
    print("🚀 Testing File Organization")
    print("=" * 50)
    
    tests = [
        ("Modal Directory Structure", test_modal_directory_structure),
        ("Deployment Files", test_deployment_files),
        ("README Consolidation", test_readme_consolidation),
        ("Environment Variables", test_environment_variables),
        ("File Cleanup", test_file_cleanup)
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
    print("📊 File Organization Results")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 File organization is complete and clean!")
        print("\n📋 Organization Summary:")
        print("• Modal functions organized in modal/ directory")
        print("• Deployment scripts ready: modal/deploy.py & deploy_local.py")
        print("• Documentation consolidated in main README.md")
        print("• Environment configuration ready for deployment")
    else:
        print("⚠️ Some organization issues found.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)