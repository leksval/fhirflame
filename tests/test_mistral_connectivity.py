#!/usr/bin/env python3
"""
🔍 Mistral API Connectivity Diagnostic Tool
Standalone tool to debug and isolate Mistral OCR API issues
"""

import os
import sys
import json
import time
import base64
import socket
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import httpx
    import ssl
except ImportError:
    print("❌ Missing dependencies. Install with: pip install httpx")
    sys.exit(1)

class MistralConnectivityTester:
    """Comprehensive Mistral API connectivity and authentication tester"""
    
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.api_base = "https://api.mistral.ai"
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "environment": "container" if os.getenv("CONTAINER_MODE") else "host",
            "tests": {}
        }
        
    def log_test(self, test_name: str, success: bool, details: Dict[str, Any], error: str = None):
        """Log test results with detailed information"""
        self.test_results["tests"][test_name] = {
            "success": success,
            "details": details,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {details.get('summary', 'No summary')}")
        if error:
            print(f"   Error: {error}")
        if details.get("metrics"):
            for key, value in details["metrics"].items():
                print(f"   {key}: {value}")
        print()

    def test_environment_variables(self) -> bool:
        """Test 1: Environment Variable Validation"""
        print("🔧 Testing Environment Variables...")
        
        details = {
            "summary": "Environment variable validation",
            "api_key_present": bool(self.api_key),
            "api_key_format": "valid" if self.api_key and len(self.api_key) > 20 else "invalid",
            "container_mode": os.getenv("CONTAINER_MODE", "false"),
            "use_mistral_fallback": os.getenv("USE_MISTRAL_FALLBACK", "false"),
            "python_version": sys.version,
            "environment_vars": {
                "MISTRAL_API_KEY": "present" if self.api_key else "missing",
                "USE_MISTRAL_FALLBACK": os.getenv("USE_MISTRAL_FALLBACK", "not_set"),
                "PYTHONPATH": os.getenv("PYTHONPATH", "not_set")
            }
        }
        
        success = bool(self.api_key) and len(self.api_key) > 20
        error = None if success else "MISTRAL_API_KEY missing or invalid format"
        
        self.log_test("Environment Variables", success, details, error)
        return success

    async def test_dns_resolution(self) -> bool:
        """Test 2: DNS Resolution"""
        print("🌐 Testing DNS Resolution...")
        
        start_time = time.time()
        try:
            # Test DNS resolution for Mistral API
            host = "api.mistral.ai"
            addresses = socket.getaddrinfo(host, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
            resolution_time = time.time() - start_time
            
            details = {
                "summary": f"DNS resolution successful ({resolution_time:.3f}s)",
                "host": host,
                "resolved_addresses": [addr[4][0] for addr in addresses],
                "metrics": {
                    "resolution_time": f"{resolution_time:.3f}s",
                    "address_count": len(addresses)
                }
            }
            
            self.log_test("DNS Resolution", True, details)
            return True
            
        except Exception as e:
            details = {
                "summary": "DNS resolution failed",
                "host": "api.mistral.ai",
                "metrics": {
                    "resolution_time": f"{time.time() - start_time:.3f}s"
                }
            }
            
            self.log_test("DNS Resolution", False, details, str(e))
            return False

    async def test_https_connectivity(self) -> bool:
        """Test 3: HTTPS Connectivity"""
        print("🔗 Testing HTTPS Connectivity...")
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.api_base}/")
                connection_time = time.time() - start_time
                
                details = {
                    "summary": f"HTTPS connection successful ({connection_time:.3f}s)",
                    "status_code": response.status_code,
                    "response_headers": dict(response.headers),
                    "metrics": {
                        "connection_time": f"{connection_time:.3f}s",
                        "status_code": response.status_code
                    }
                }
                
                success = response.status_code in [200, 404, 405]  # Any valid HTTP response
                error = None if success else f"Unexpected status code: {response.status_code}"
                
                self.log_test("HTTPS Connectivity", success, details, error)
                return success
                
        except Exception as e:
            details = {
                "summary": "HTTPS connection failed",
                "url": f"{self.api_base}/",
                "metrics": {
                    "connection_time": f"{time.time() - start_time:.3f}s"
                }
            }
            
            self.log_test("HTTPS Connectivity", False, details, str(e))
            return False

    async def test_api_authentication(self) -> bool:
        """Test 4: API Authentication"""
        print("🔐 Testing API Authentication...")
        
        if not self.api_key:
            details = {"summary": "Cannot test authentication - no API key"}
            self.log_test("API Authentication", False, details, "MISTRAL_API_KEY not available")
            return False
        
        start_time = time.time()
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Test with a minimal valid request to check authentication
            test_payload = {
                "model": "pixtral-12b-2409",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Hello"
                            }
                        ]
                    }
                ],
                "max_tokens": 10
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base}/v1/chat/completions",
                    headers=headers,
                    json=test_payload
                )
                
                auth_time = time.time() - start_time
                
                details = {
                    "summary": f"Authentication test completed ({auth_time:.3f}s)",
                    "status_code": response.status_code,
                    "api_key_format": f"sk-...{self.api_key[-4:]}" if self.api_key else "none",
                    "metrics": {
                        "auth_time": f"{auth_time:.3f}s",
                        "status_code": response.status_code
                    }
                }
                
                if response.status_code == 200:
                    # Successfully authenticated and got a response
                    success = True
                    error = None
                elif response.status_code == 401:
                    # Authentication failed
                    success = False
                    error = "Invalid API key - authentication failed"
                elif response.status_code == 429:
                    # Rate limited but API key is valid
                    success = True  # Auth is working, just rate limited
                    error = None
                    details["summary"] = "Authentication successful (rate limited)"
                else:
                    # Other error
                    try:
                        error_data = response.json()
                        error = f"API error: {error_data.get('message', response.text)}"
                    except:
                        error = f"HTTP {response.status_code}: {response.text}"
                    success = False
                
                self.log_test("API Authentication", success, details, error)
                return success
                
        except Exception as e:
            details = {
                "summary": "Authentication test failed",
                "api_key_format": f"sk-...{self.api_key[-4:]}" if self.api_key else "none",
                "metrics": {
                    "auth_time": f"{time.time() - start_time:.3f}s"
                }
            }
            
            self.log_test("API Authentication", False, details, str(e))
            return False

    async def test_ocr_api_call(self) -> bool:
        """Test 5: Simple OCR API Call"""
        print("📄 Testing OCR API Call...")
        
        if not self.api_key:
            details = {"summary": "Cannot test OCR - no API key"}
            self.log_test("OCR API Call", False, details, "MISTRAL_API_KEY not available")
            return False
        
        start_time = time.time()
        try:
            # Create a minimal test image (1x1 white pixel PNG)
            test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "pixtral-12b-2409",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract text from this image. If no text is found, respond with 'NO_TEXT_FOUND'."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{test_image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 100
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                ocr_time = time.time() - start_time
                
                details = {
                    "summary": f"OCR API call completed ({ocr_time:.3f}s)",
                    "status_code": response.status_code,
                    "request_size": len(json.dumps(payload)),
                    "metrics": {
                        "ocr_time": f"{ocr_time:.3f}s",
                        "status_code": response.status_code,
                        "payload_size": f"{len(json.dumps(payload))} bytes"
                    }
                }
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        details["response_content"] = content[:200] + "..." if len(content) > 200 else content
                        details["summary"] = f"OCR successful ({ocr_time:.3f}s)"
                        success = True
                        error = None
                    except Exception as parse_error:
                        success = False
                        error = f"Failed to parse response: {parse_error}"
                else:
                    try:
                        error_data = response.json()
                        error = f"API error: {error_data.get('message', response.text)}"
                    except:
                        error = f"HTTP {response.status_code}: {response.text}"
                    success = False
                
                self.log_test("OCR API Call", success, details, error)
                return success
                
        except Exception as e:
            details = {
                "summary": "OCR API call failed",
                "metrics": {
                    "ocr_time": f"{time.time() - start_time:.3f}s"
                }
            }
            
            self.log_test("OCR API Call", False, details, str(e))
            return False

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all connectivity tests"""
        print("🔍 Mistral API Connectivity Diagnostic Tool")
        print("=" * 50)
        
        # Run tests sequentially
        test_1 = self.test_environment_variables()
        test_2 = await self.test_dns_resolution()
        test_3 = await self.test_https_connectivity()
        test_4 = await self.test_api_authentication()
        test_5 = await self.test_ocr_api_call()
        
        # Summary
        total_tests = 5
        passed_tests = sum([test_1, test_2, test_3, test_4, test_5])
        
        print("=" * 50)
        print(f"📊 Test Summary: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("✅ All tests passed - Mistral OCR API is fully functional!")
        elif passed_tests >= 3:
            print("⚠️  Some tests failed - Mistral OCR may work with limitations")
        else:
            print("❌ Multiple tests failed - Mistral OCR likely won't work")
        
        # Add summary to results
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": f"{(passed_tests/total_tests)*100:.1f}%",
            "overall_status": "success" if passed_tests == total_tests else "partial" if passed_tests >= 3 else "failure"
        }
        
        return self.test_results

    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            env = self.test_results["environment"]
            filename = f"mistral_connectivity_test_{env}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"📄 Test results saved to: {filename}")

async def main():
    """Main entry point"""
    print("Starting Mistral API connectivity diagnostics...")
    
    tester = MistralConnectivityTester()
    results = await tester.run_all_tests()
    
    # Save results
    tester.save_results()
    
    # Exit with appropriate code
    overall_status = results["summary"]["overall_status"]
    if overall_status == "success":
        sys.exit(0)
    elif overall_status == "partial":
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(3)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(4)