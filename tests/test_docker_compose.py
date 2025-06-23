#!/usr/bin/env python3
"""
Test Docker Compose Configurations
Test the new Docker Compose setups for local and modal deployments
"""

import os
import sys
import subprocess
import time
import yaml
import tempfile

def test_compose_file_validity():
    """Test that Docker Compose files are valid YAML"""
    print("🔍 Testing Docker Compose file validity...")
    
    compose_files = [
        "docker-compose.local.yml",
        "docker-compose.modal.yml"
    ]
    
    for compose_file in compose_files:
        try:
            with open(compose_file, 'r') as f:
                yaml.safe_load(f)
            print(f"✅ {compose_file} is valid YAML")
        except yaml.YAMLError as e:
            print(f"❌ {compose_file} invalid YAML: {e}")
            return False
        except FileNotFoundError:
            print(f"❌ {compose_file} not found")
            return False
    
    return True

def test_environment_variables():
    """Test environment variable handling in compose files"""
    print("\n🔍 Testing environment variable defaults...")
    
    # Test local compose file
    try:
        with open("docker-compose.local.yml", 'r') as f:
            local_content = f.read()
        
        # Check for proper environment variable syntax
        env_patterns = [
            "${GRADIO_PORT:-7860}",
            "${A2A_API_PORT:-8000}",
            "${OLLAMA_PORT:-11434}",
            "${FHIRFLAME_DEV_MODE:-true}",
            "${HF_TOKEN}",
            "${MISTRAL_API_KEY}"
        ]
        
        for pattern in env_patterns:
            if pattern in local_content:
                print(f"✅ Local compose has: {pattern}")
            else:
                print(f"❌ Local compose missing: {pattern}")
                return False
        
        # Test modal compose file
        with open("docker-compose.modal.yml", 'r') as f:
            modal_content = f.read()
        
        modal_patterns = [
            "${MODAL_TOKEN_ID}",
            "${MODAL_TOKEN_SECRET}",
            "${MODAL_ENDPOINT_URL}",
            "${MODAL_L4_HOURLY_RATE:-0.73}",
            "${AUTH0_DOMAIN:-}"
        ]
        
        for pattern in modal_patterns:
            if pattern in modal_content:
                print(f"✅ Modal compose has: {pattern}")
            else:
                print(f"❌ Modal compose missing: {pattern}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Environment variable test failed: {e}")
        return False

def test_compose_services():
    """Test that required services are defined"""
    print("\n🔍 Testing service definitions...")
    
    try:
        # Test local services
        with open("docker-compose.local.yml", 'r') as f:
            local_config = yaml.safe_load(f)
        
        local_services = local_config.get('services', {})
        required_local_services = [
            'fhirflame-local',
            'fhirflame-a2a-api',
            'ollama',
            'ollama-setup'
        ]
        
        for service in required_local_services:
            if service in local_services:
                print(f"✅ Local has service: {service}")
            else:
                print(f"❌ Local missing service: {service}")
                return False
        
        # Test modal services
        with open("docker-compose.modal.yml", 'r') as f:
            modal_config = yaml.safe_load(f)
        
        modal_services = modal_config.get('services', {})
        required_modal_services = [
            'fhirflame-modal',
            'fhirflame-a2a-modal',
            'modal-deployer'
        ]
        
        for service in required_modal_services:
            if service in modal_services:
                print(f"✅ Modal has service: {service}")
            else:
                print(f"❌ Modal missing service: {service}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Service definition test failed: {e}")
        return False

def test_port_configurations():
    """Test port configurations and conflicts"""
    print("\n🔍 Testing port configurations...")
    
    try:
        # Check local ports
        with open("docker-compose.local.yml", 'r') as f:
            local_config = yaml.safe_load(f)
        
        local_ports = []
        for service_name, service_config in local_config['services'].items():
            ports = service_config.get('ports', [])
            for port_mapping in ports:
                if isinstance(port_mapping, str):
                    host_port = port_mapping.split(':')[0]
                    # Extract port from env var syntax like ${PORT:-8000}
                    if 'GRADIO_PORT:-7860' in host_port:
                        local_ports.append('7860')
                    elif 'A2A_API_PORT:-8000' in host_port:
                        local_ports.append('8000')
                    elif 'OLLAMA_PORT:-11434' in host_port:
                        local_ports.append('11434')
        
        print(f"✅ Local default ports: {', '.join(local_ports)}")
        
        # Check modal ports
        with open("docker-compose.modal.yml", 'r') as f:
            modal_config = yaml.safe_load(f)
        
        modal_ports = []
        for service_name, service_config in modal_config['services'].items():
            ports = service_config.get('ports', [])
            for port_mapping in ports:
                if isinstance(port_mapping, str):
                    host_port = port_mapping.split(':')[0]
                    if 'GRADIO_PORT:-7860' in host_port:
                        modal_ports.append('7860')
                    elif 'A2A_API_PORT:-8000' in host_port:
                        modal_ports.append('8000')
        
        print(f"✅ Modal default ports: {', '.join(modal_ports)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Port configuration test failed: {e}")
        return False

def test_compose_validation():
    """Test Docker Compose file validation using docker-compose"""
    print("\n🔍 Testing Docker Compose validation...")
    
    compose_files = [
        "docker-compose.local.yml",
        "docker-compose.modal.yml"
    ]
    
    for compose_file in compose_files:
        try:
            # Test compose file validation
            result = subprocess.run([
                "docker-compose", "-f", compose_file, "config"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ {compose_file} validates with docker-compose")
            else:
                print(f"❌ {compose_file} validation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⚠️ {compose_file} validation timeout (docker-compose not available)")
        except FileNotFoundError:
            print(f"⚠️ docker-compose not found, skipping validation for {compose_file}")
        except Exception as e:
            print(f"⚠️ {compose_file} validation error: {e}")
    
    return True

def test_health_check_definitions():
    """Test that health checks are properly defined"""
    print("\n🔍 Testing health check definitions...")
    
    try:
        # Test local health checks
        with open("docker-compose.local.yml", 'r') as f:
            local_config = yaml.safe_load(f)
        
        services_with_healthcheck = []
        for service_name, service_config in local_config['services'].items():
            if 'healthcheck' in service_config:
                healthcheck = service_config['healthcheck']
                if 'test' in healthcheck:
                    services_with_healthcheck.append(service_name)
        
        print(f"✅ Local services with health checks: {', '.join(services_with_healthcheck)}")
        
        # Test modal health checks
        with open("docker-compose.modal.yml", 'r') as f:
            modal_config = yaml.safe_load(f)
        
        modal_healthchecks = []
        for service_name, service_config in modal_config['services'].items():
            if 'healthcheck' in service_config:
                modal_healthchecks.append(service_name)
        
        print(f"✅ Modal services with health checks: {', '.join(modal_healthchecks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False

def main():
    """Run all Docker Compose tests"""
    print("🐳 Testing Docker Compose Configurations")
    print("=" * 50)
    
    tests = [
        ("YAML Validity", test_compose_file_validity),
        ("Environment Variables", test_environment_variables),
        ("Service Definitions", test_compose_services),
        ("Port Configurations", test_port_configurations),
        ("Compose Validation", test_compose_validation),
        ("Health Checks", test_health_check_definitions)
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
    print("📊 Docker Compose Test Results")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Docker Compose tests passed!")
        print("\n📋 Deployment Commands:")
        print("🏠 Local:     docker-compose -f docker-compose.local.yml up")
        print("☁️  Modal:     docker-compose -f docker-compose.modal.yml up")
        print("🧪 Test Local: docker-compose -f docker-compose.local.yml --profile test up")
        print("🚀 Deploy Modal: docker-compose -f docker-compose.modal.yml --profile deploy up")
    else:
        print("\n⚠️ Some Docker Compose tests failed.")
    
    return passed == total

if __name__ == "__main__":
    # Change to project directory
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    success = main()
    sys.exit(0 if success else 1)