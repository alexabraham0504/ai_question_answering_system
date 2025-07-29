#!/usr/bin/env python3
"""
GPU Setup Verification Script
Run this script to check if your GPU is properly configured for the AI Question Answering System.
"""

import sys
import subprocess

def check_nvidia_drivers():
    """Check if NVIDIA drivers are installed"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA drivers are installed")
            print(result.stdout)
            return True
        else:
            print("❌ NVIDIA drivers not found or not working")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found. NVIDIA drivers may not be installed.")
        return False

def check_cuda_toolkit():
    """Check if CUDA toolkit is installed"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA toolkit is installed")
            print(result.stdout)
            return True
        else:
            print("❌ CUDA toolkit not found")
            return False
    except FileNotFoundError:
        print("❌ nvcc not found. CUDA toolkit may not be installed.")
        return False

def check_pytorch_cuda():
    """Check if PyTorch has CUDA support"""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"✅ GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test GPU memory allocation
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                print(f"✅ GPU memory test: {allocated:.3f} GB allocated")
                del test_tensor
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                print(f"❌ GPU memory test failed: {str(e)}")
                return False
        else:
            print("❌ PyTorch not compiled with CUDA support")
            print("💡 Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_gpu_libraries():
    """Check if GPU optimization libraries are installed"""
    libraries = {
        'xformers': 'Memory efficient attention'
    }
    
    all_installed = True
    for lib, description in libraries.items():
        try:
            __import__(lib)
            print(f"✅ {lib}: {description}")
        except ImportError:
            print(f"⚠️  {lib}: {description} (optional)")
            all_installed = False
    
    return all_installed

def main():
    print("🚀 GPU Setup Verification for AI Question Answering System")
    print("=" * 60)
    
    # Check all components
    nvidia_ok = check_nvidia_drivers()
    print()
    
    cuda_ok = check_cuda_toolkit()
    print()
    
    pytorch_ok = check_pytorch_cuda()
    print()
    
    gpu_libs_ok = check_gpu_libraries()
    print()
    
    # Summary
    print("=" * 60)
    print("📊 SUMMARY:")
    print(f"NVIDIA Drivers: {'✅' if nvidia_ok else '❌'}")
    print(f"CUDA Toolkit: {'✅' if cuda_ok else '❌'}")
    print(f"PyTorch CUDA: {'✅' if pytorch_ok else '❌'}")
    print(f"GPU Libraries: {'✅' if gpu_libs_ok else '⚠️'}")
    
    if nvidia_ok and cuda_ok and pytorch_ok:
        print("\n🎉 GPU setup is ready! You can run the app with GPU acceleration.")
    else:
        print("\n🔧 GPU setup needs attention. Please install missing components.")
        print("\n📋 Installation commands:")
        print("1. Install NVIDIA drivers from: https://www.nvidia.com/drivers/")
        print("2. Install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads")
        print("3. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("4. Install GPU libraries: pip install xformers")

if __name__ == "__main__":
    main() 