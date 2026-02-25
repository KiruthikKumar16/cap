"""
Test Setup Script

Tests that all components are working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"[FAIL] PyTorch import failed: {e}")
        return False
    
    try:
        import torch_geometric
        print(f"[OK] PyTorch Geometric {torch_geometric.__version__}")
    except ImportError as e:
        print(f"[FAIL] PyTorch Geometric import failed: {e}")
        return False
    
    try:
        import stable_baselines3
        print(f"[OK] Stable Baselines3")
    except ImportError as e:
        print(f"[FAIL] Stable Baselines3 import failed: {e}")
        return False
    
    try:
        import gymnasium
        print(f"[OK] Gymnasium")
    except ImportError as e:
        print(f"[FAIL] Gymnasium import failed: {e}")
        return False
    
    return True


def test_graph_builder():
    """Test graph builder module."""
    print("\nTesting graph builder...")
    
    try:
        from src.phase1.graph_builder import TrafficGraphBuilder
        
        # Test with placeholder
        builder = TrafficGraphBuilder("dummy.net.xml")
        assert builder.get_num_nodes() > 0, "Should have nodes"
        edge_index = builder.get_edge_index()
        assert edge_index.shape[0] == 2, "Edge index should have 2 rows"
        print("[OK] Graph builder works")
        return True
    except Exception as e:
        print(f"[FAIL] Graph builder test failed: {e}")
        return False


def test_feature_extractor():
    """Test feature extractor module."""
    print("\nTesting feature extractor...")
    
    try:
        from src.phase1.feature_extractor import TrafficFeatureExtractor
        
        intersections = ["J0", "J1", "J2", "J3"]
        extractor = TrafficFeatureExtractor(intersections)
        features = extractor.extract()
        
        assert features.shape[0] == len(intersections), "Should have features for all intersections"
        assert features.shape[1] == 12, "Should have 12 features per intersection"
        print("[OK] Feature extractor works")
        return True
    except Exception as e:
        print(f"[FAIL] Feature extractor test failed: {e}")
        return False


def test_config_loading():
    """Test configuration file loading."""
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        config_path = project_root / "configs" / "phase1.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            assert 'sumo' in config, "Config should have 'sumo' section"
            assert 'model' in config, "Config should have 'model' section"
            print("[OK] Configuration loading works")
            return True
        else:
            print("[WARN] Configuration file not found (this is OK if not created yet)")
            return True
    except Exception as e:
        print(f"[FAIL] Configuration loading test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Capstone Project - Setup Test")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Graph Builder", test_graph_builder()))
    results.append(("Feature Extractor", test_feature_extractor()))
    results.append(("Config Loading", test_config_loading()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("[SUCCESS] All tests passed! Environment is ready.")
    else:
        print("[WARN] Some tests failed. Check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
