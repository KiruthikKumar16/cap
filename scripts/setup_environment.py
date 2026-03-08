"""
Environment Setup Script

This script helps set up the development environment for the capstone project.
It checks for required dependencies and provides installation instructions.
"""

import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.10 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("[FAIL] Python 3.10+ required. Current version:", sys.version)
        return False
    print(f"[OK] Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"[OK] {package_name} is installed")
        return True
    except ImportError:
        print(f"[FAIL] {package_name} is NOT installed")
        return False


def check_sumo():
    """Check if SUMO is installed and accessible. SUMO is now MANDATORY."""
    try:
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"[OK] SUMO is installed: {version_line}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
    
    print("[ERROR] SUMO is NOT installed or not in PATH - MANDATORY REQUIREMENT")
    print("\n   ┌─ INSTALLATION INSTRUCTIONS ─────────────────────────────────┐")
    print("   │ Linux/Ubuntu:                                               │")
    print("   │   sudo apt-get update && apt-get install -y sumo sumo-tools│")
    print("   │                                                             │")
    print("   │ macOS (Homebrew):                                          │")
    print("   │   brew install sumo                                         │")
    print("   │                                                             │")
    print("   │ Windows:                                                    │")
    print("   │   Download from: https://sumo.dlr.de/docs/Installing/      │")
    print("   │   Then add to PATH or set SUMO_HOME environment variable   │")
    print("   └─────────────────────────────────────────────────────────────┘")
    return False


def check_sumo_python():
    """Check if SUMO Python libraries are available."""
    sumo_ok = check_package("sumolib", "sumolib")
    traci_ok = check_package("traci", "traci")
    return sumo_ok and traci_ok


def install_requirements():
    """Install requirements from requirements.txt."""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("[FAIL] requirements.txt not found")
        return False
    
    print("\n[INFO] Installing requirements from requirements.txt...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                      check=True)
        print("[OK] Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("[FAIL] Failed to install requirements")
        return False


def main():
    """Main setup check."""
    print("=" * 60)
    print("Capstone Project - Environment Setup Check")
    print("=" * 60)
    print()
    
    # Check Python version
    python_ok = check_python_version()
    if not python_ok:
        print("\n⚠️  Please upgrade Python to 3.10+")
        return False
    
    print("\n" + "=" * 60)
    print("Checking Python Packages")
    print("=" * 60)
    
    # Check core packages
    packages = [
        ("torch", "torch"),
        ("torch-geometric", "torch_geometric"),
        ("stable-baselines3", "stable_baselines3"),
        ("gymnasium", "gymnasium"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("pyyaml", "yaml"),
        ("networkx", "networkx"),
        ("matplotlib", "matplotlib"),
    ]
    
    missing_packages = []
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            missing_packages.append(pkg_name)
    
    print("\n" + "=" * 60)
    print("Checking SUMO")
    print("=" * 60)
    
    sumo_ok = check_sumo()
    sumo_python_ok = check_sumo_python()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if missing_packages:
        print(f"\n[WARN] Missing packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        
        response = input("\nWould you like to install requirements now? (y/n): ")
        if response.lower() == 'y':
            if not install_requirements():
                return False
    else:
        print("\n[OK] All Python packages are installed")
    
    # SUMO is now MANDATORY
    if not sumo_ok:
        print("\n[ERROR] SUMO is REQUIRED (no placeholder mode)")
        print("   Install SUMO from: https://sumo.dlr.de/docs/Installing/index.html")
        print("   Then verify with: sumo --version")
        print("\n   Without SUMO, the project will not run.")
        return False
    
    if not sumo_python_ok and sumo_ok:
        print("\n[WARN] SUMO Python libraries not found")
        print("   Install with: pip install sumolib traci")
        return False
    
    print("\n[OK] Environment setup complete! Ready to run training.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
