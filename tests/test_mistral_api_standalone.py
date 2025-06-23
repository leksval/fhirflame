#!/usr/bin/env python3
"""
Standalone Mistral API Test Script
Comprehensive diagnostic tool to identify why Mistral API calls aren't reaching the console
"""

import asyncio
import httpx
import base64
import os
import json
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

class MistralAPITester:
    """Comprehensive Mistral API testing suite"""
    
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.base_url = "https://api.mistral.ai/v1"
        self.test_results = {}
        
        # Test configuration
        self.timeout = 30.0
        self.test_model = "pixtral-12b-2409"
        
        print(f"🔧 Mistral API Diagnostic Tool")
        print(f"⏰ Timestamp: {datetime.now().isoformat()}")
        print(f"🔑 API Key: {'✅ Present' if self.api_key else '❌ Missing'}")
        if self.api_key:
            print(f"🔑 Key format: {self.api_key[:8]}...{self.api_key[-4:]}")
        print(f"🌐 Base URL: {self.base_url}")
        print(f"🤖 Test Model: {self.test_model}")
        print("=" * 70)

    async def test_1_basic_connectivity(self):
        """Test 1: Basic network connectivity to Mistral API"""
        print("\n🔌 TEST 1: Basic Connectivity")
        print("-" * 30)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Test basic connectivity to the API endpoint
                response = await client.get(f"{self.base_url}/models")
                
                print(f"📡 Network Status: ✅ Connected")
                print(f"🌐 Response Code: {response.status_code}")
                print(f"⏱️  Response Time: {response.elapsed.total_seconds():.3f}s")
                
                if response.status_code == 401:
                    print("🔐 Authentication Required (Expected for /models endpoint)")
                    self.test_results["connectivity"] = "✅ PASS - Network reachable"
                elif response.status_code == 200:
                    print("📋 Models endpoint accessible")
                    self.test_results["connectivity"] = "✅ PASS - Full access"
                else:
                    print(f"⚠️  Unexpected status: {response.status_code}")
                    print(f"📄 Response: {response.text[:200]}")
                    self.test_results["connectivity"] = f"⚠️  PARTIAL - Status {response.status_code}"
                    
        except httpx.ConnectTimeout:
            print("❌ Connection timeout - Network/firewall issue")
            self.test_results["connectivity"] = "❌ FAIL - Connection timeout"
        except httpx.ConnectError as e:
            print(f"❌ Connection error: {e}")
            self.test_results["connectivity"] = f"❌ FAIL - Connection error: {e}"
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.test_results["connectivity"] = f"❌ FAIL - {type(e).__name__}: {e}"

    async def test_2_authentication(self):
        """Test 2: API key authentication"""
        print("\n🔐 TEST 2: Authentication")
        print("-" * 30)
        
        if not self.api_key:
            print("❌ No API key provided")
            self.test_results["authentication"] = "❌ FAIL - No API key"
            return
            
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Test authentication with a simple chat completion
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "mistral-tiny",  # Use basic model for auth test
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 10
                    }
                )
                
                print(f"🔑 Auth Status: {response.status_code}")
                print(f"📊 Response Size: {len(response.content)} bytes")
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Authentication successful")
                    print(f"📝 Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')[:50]}...")
                    self.test_results["authentication"] = "✅ PASS - Valid API key"
                elif response.status_code == 401:
                    print("❌ Authentication failed - Invalid API key")
                    error_detail = response.text[:200]
                    print(f"📄 Error: {error_detail}")
                    self.test_results["authentication"] = f"❌ FAIL - Invalid key: {error_detail}"
                elif response.status_code == 429:
                    print("⏸️  Rate limited - API key works but quota exceeded")
                    self.test_results["authentication"] = "✅ PASS - Valid key (rate limited)"
                else:
                    print(f"⚠️  Unexpected status: {response.status_code}")
                    print(f"📄 Response: {response.text[:200]}")
                    self.test_results["authentication"] = f"⚠️  UNKNOWN - Status {response.status_code}"
                    
        except Exception as e:
            print(f"❌ Authentication test failed: {e}")
            self.test_results["authentication"] = f"❌ FAIL - {type(e).__name__}: {e}"

    async def test_3_vision_model_availability(self):
        """Test 3: Vision model availability"""
        print("\n👁️  TEST 3: Vision Model Availability")
        print("-" * 30)
        
        if not self.api_key:
            print("⏭️  Skipping - No API key")
            self.test_results["vision_model"] = "⏭️  SKIP - No API key"
            return
            
        try:
            # Create a simple test image
            test_image = Image.new('RGB', (100, 100), color='white')
            
            # Add some text to the image
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(test_image)
            try:
                # Try to use default font
                draw.text((10, 10), "TEST IMAGE", fill='black')
            except:
                # If font fails, just draw without text
                pass
            
            # Convert to base64
            img_byte_arr = io.BytesIO()
            test_image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            b64_data = base64.b64encode(img_bytes).decode()
            
            print(f"🖼️  Created test image: {len(img_bytes)} bytes")
            print(f"📊 Base64 length: {len(b64_data)} chars")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.test_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Describe this image briefly."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{b64_data}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 50
                    }
                )
                
                print(f"🤖 Vision API Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')
                    print(f"✅ Vision model works: {content[:100]}...")
                    self.test_results["vision_model"] = "✅ PASS - Vision API working"
                elif response.status_code == 400:
                    error_detail = response.text[:200]
                    print(f"❌ Bad request - Model or format issue: {error_detail}")
                    self.test_results["vision_model"] = f"❌ FAIL - Bad request: {error_detail}"
                elif response.status_code == 404:
                    print(f"❌ Model not found - {self.test_model} may not exist")
                    self.test_results["vision_model"] = f"❌ FAIL - Model not found: {self.test_model}"
                else:
                    print(f"⚠️  Unexpected status: {response.status_code}")
                    print(f"📄 Response: {response.text[:200]}")
                    self.test_results["vision_model"] = f"⚠️  UNKNOWN - Status {response.status_code}"
                    
        except Exception as e:
            print(f"❌ Vision model test failed: {e}")
            self.test_results["vision_model"] = f"❌ FAIL - {type(e).__name__}: {e}"

    async def test_4_exact_app_request(self):
        """Test 4: Exact request format from main application"""
        print("\n🎯 TEST 4: Exact App Request Format")
        print("-" * 30)
        
        if not self.api_key:
            print("⏭️  Skipping - No API key")
            self.test_results["app_request"] = "⏭️  SKIP - No API key"
            return
            
        try:
            # Create the same test image as the app would process
            test_image = Image.new('RGB', (200, 100), color='white')
            draw = ImageDraw.Draw(test_image)
            draw.text((10, 10), "MEDICAL DOCUMENT TEST", fill='black')
            draw.text((10, 30), "Patient: John Doe", fill='black')
            draw.text((10, 50), "DOB: 01/01/1980", fill='black')
            
            # Convert exactly like the app does
            if test_image.mode != 'RGB':
                test_image = test_image.convert('RGB')
            
            img_byte_arr = io.BytesIO()
            test_image.save(img_byte_arr, format='JPEG', quality=95)
            img_bytes = img_byte_arr.getvalue()
            b64_data = base64.b64encode(img_bytes).decode()
            
            print(f"📄 Simulated medical document: {len(img_bytes)} bytes")
            
            # Use EXACT request format from the main app
            request_payload = {
                "model": "pixtral-12b-2409",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """You are a strict OCR text extraction tool. Your job is to extract ONLY the actual text that appears in this image - nothing more, nothing less.

CRITICAL RULES:
- Extract ONLY text that is actually visible in the image
- Do NOT generate, invent, or create any content
- Do NOT add examples or sample data
- Do NOT fill in missing information
- If the image contains minimal text, return minimal text
- If the image is blank or contains no medical content, return what you actually see

Extract exactly what text appears in this image:"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 8000,
                "temperature": 0.0
            }
            
            print(f"📝 Request payload size: {len(json.dumps(request_payload))} chars")
            
            async with httpx.AsyncClient(timeout=180.0) as client:  # Same timeout as app
                print("🚀 Sending exact app request...")
                
                response = await client.post(
                    "https://api.mistral.ai/v1/chat/completions",  # Exact URL from app
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_payload
                )
                
                print(f"📊 App Format Status: {response.status_code}")
                print(f"📏 Response Size: {len(response.content)} bytes")
                print(f"🕒 Response Headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')
                    print(f"✅ Exact app request works!")
                    print(f"📝 Extracted text: {content[:200]}...")
                    self.test_results["app_request"] = "✅ PASS - App format works perfectly"
                    
                    # This is the smoking gun - if this works, the app should work too
                    print("\n🚨 CRITICAL: This exact request format WORKS!")
                    print("🚨 The main app should be using Mistral API successfully!")
                    print("🚨 Check app logs for why it's falling back to multimodal processor!")
                    
                else:
                    error_detail = response.text[:300]
                    print(f"❌ App request format failed: {error_detail}")
                    self.test_results["app_request"] = f"❌ FAIL - {response.status_code}: {error_detail}"
                    
        except Exception as e:
            print(f"❌ App request test failed: {e}")
            self.test_results["app_request"] = f"❌ FAIL - {type(e).__name__}: {e}"

    async def test_5_environment_check(self):
        """Test 5: Environment and configuration check"""
        print("\n🌍 TEST 5: Environment Check")
        print("-" * 30)
        
        # Check environment variables
        env_vars = {
            "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
            "USE_MISTRAL_FALLBACK": os.getenv("USE_MISTRAL_FALLBACK"),
            "USE_MULTIMODAL_FALLBACK": os.getenv("USE_MULTIMODAL_FALLBACK"),
            "PYTHONPATH": os.getenv("PYTHONPATH"),
        }
        
        print("📋 Environment Variables:")
        for key, value in env_vars.items():
            if key == "MISTRAL_API_KEY" and value:
                print(f"  {key}: {value[:8]}...{value[-4:]}")
            else:
                print(f"  {key}: {value}")
        
        # Check if we're in Docker
        in_docker = os.path.exists('/.dockerenv') or os.path.exists('/proc/1/cgroup')
        print(f"🐳 Docker Environment: {'Yes' if in_docker else 'No'}")
        
        # Check Python environment
        print(f"🐍 Python Version: {sys.version}")
        print(f"📁 Working Directory: {os.getcwd()}")
        
        # Check required libraries
        try:
            import httpx
            print(f"📦 httpx version: {httpx.__version__}")
        except ImportError:
            print("❌ httpx not available")
        
        # Check if main app files exist
        app_files = ["src/file_processor.py", "src/workflow_orchestrator.py", ".env"]
        print("\n📁 App Files:")
        for file in app_files:
            exists = Path(file).exists()
            print(f"  {file}: {'✅ Exists' if exists else '❌ Missing'}")
        
        self.test_results["environment"] = "✅ Environment checked"

    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n" + "=" * 70)
        print("📊 DIAGNOSTIC REPORT")
        print("=" * 70)
        
        print(f"⏰ Test completed: {datetime.now().isoformat()}")
        print(f"🔑 API Key: {'Present' if self.api_key else 'Missing'}")
        
        print("\n🧪 Test Results:")
        for test_name, result in self.test_results.items():
            print(f"  {test_name.replace('_', ' ').title()}: {result}")
        
        # Analysis and recommendations
        print("\n🔍 ANALYSIS:")
        
        connectivity_ok = "✅ PASS" in self.test_results.get("connectivity", "")
        auth_ok = "✅ PASS" in self.test_results.get("authentication", "")
        vision_ok = "✅ PASS" in self.test_results.get("vision_model", "")
        app_format_ok = "✅ PASS" in self.test_results.get("app_request", "")
        
        if not connectivity_ok:
            print("❌ NETWORK ISSUE: Cannot reach Mistral API servers")
            print("   → Check firewall, DNS, or network connectivity")
        elif not auth_ok:
            print("❌ AUTHENTICATION ISSUE: API key is invalid")
            print("   → Verify API key in Mistral dashboard")
        elif not vision_ok:
            print("❌ MODEL ISSUE: Vision model unavailable or incorrect")
            print("   → Check if pixtral-12b-2409 model exists")
        elif app_format_ok:
            print("🚨 CRITICAL FINDING: Mistral API works perfectly!")
            print("   → The main app SHOULD be working")
            print("   → Issue is in the app's error handling or fallback logic")
            print("   → Check app logs for silent failures")
        else:
            print("❓ UNKNOWN ISSUE: API reachable but requests failing")
            print("   → Check request format or API changes")
        
        print("\n🎯 NEXT STEPS:")
        if app_format_ok:
            print("1. Check main app logs for 'mistral_fallback_failed' events")
            print("2. Add more detailed error logging in _extract_with_mistral()")
            print("3. Verify environment variables in Docker container")
            print("4. Check if multimodal fallback is masking Mistral errors")
        else:
            print("1. Fix the identified API issues above")
            print("2. Re-run this test script")
            print("3. Test the main application after fixes")

async def main():
    """Run all diagnostic tests"""
    tester = MistralAPITester()
    
    # Run all tests in sequence
    await tester.test_1_basic_connectivity()
    await tester.test_2_authentication()
    await tester.test_3_vision_model_availability()
    await tester.test_4_exact_app_request()
    await tester.test_5_environment_check()
    
    # Generate final report
    tester.generate_report()

if __name__ == "__main__":
    # Load environment variables from .env file if present
    env_file = Path(".env")
    if env_file.exists():
        print(f"📄 Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    os.environ[key.strip()] = value.strip()
    
    # Run the diagnostic tests
    asyncio.run(main())