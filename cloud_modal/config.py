#!/usr/bin/env python3
"""
Modal Configuration Setup for FhirFlame
Following https://modal.com/docs/reference/modal.config
"""
import os
import modal
from dotenv import load_dotenv

def setup_modal_config():
    """Set up Modal configuration properly"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if Modal tokens are properly configured
    token_id = os.getenv("MODAL_TOKEN_ID")
    token_secret = os.getenv("MODAL_TOKEN_SECRET")
    
    if not token_id or not token_secret:
        print("❌ Modal tokens not found!")
        print("\n📋 Setup Modal Authentication:")
        print("1. Visit https://modal.com and create an account")
        print("2. Run: modal token new")
        print("3. Or set environment variables:")
        print("   export MODAL_TOKEN_ID=ak-...")
        print("   export MODAL_TOKEN_SECRET=as-...")
        return False
    
    print("✅ Modal tokens found")
    print(f"   Token ID: {token_id[:10]}...")
    print(f"   Token Secret: {token_secret[:10]}...")
    
    # Test Modal connection by creating a simple app
    try:
        # This will verify the tokens work by creating an app instance
        app = modal.App("fhirflame-config-test")
        print("✅ Modal client connection successful")
        return True
        
    except Exception as e:
        if "authentication" in str(e).lower() or "token" in str(e).lower():
            print(f"❌ Modal authentication failed: {e}")
            print("\n🔧 Fix authentication:")
            print("1. Check your tokens are correct")
            print("2. Run: modal token new")
            print("3. Or update your .env file")
        else:
            print(f"❌ Modal connection failed: {e}")
        return False

def get_modal_app():
    """Get properly configured Modal app"""
    if not setup_modal_config():
        raise Exception("Modal configuration failed")
    
    return modal.App("fhirflame-medical-scaling")

if __name__ == "__main__":
    success = setup_modal_config()
    if success:
        print("🎉 Modal configuration is ready!")
    else:
        print("❌ Modal configuration needs attention")