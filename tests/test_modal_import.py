#!/usr/bin/env python3
"""Quick test for Modal deployment import"""

try:
    import modal_deployment
    print("✅ Modal deployment imported successfully")
    
    # Test the cost calculation function
    cost = modal_deployment.calculate_real_modal_cost(1.0, "A100")
    print(f"✅ Cost calculation works: ${cost:.6f}")
    
except Exception as e:
    print(f"❌ Modal deployment import failed: {e}")
    import traceback
    traceback.print_exc()