import subprocess
import sys
import importlib

def get_required_packages():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def check_and_install_package(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_dependencies():
    required_packages = get_required_packages()
    for package in required_packages:
        check_and_install_package(package)
    print("All required packages have been installed.")

if __name__ == "__main__":
    install_dependencies()
