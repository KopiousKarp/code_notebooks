# save as pytorch_rocm_benchmark.py
import torch
import time
import argparse
import sys

def run_benchmark(max_size=4096, device="cuda"):
    """Run increasingly complex operations to test PyTorch-ROCm integration"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA/ROCm requested but not available. Exiting.")
        sys.exit(1)
    
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    print("\n1. Basic tensor operations:")
    # Test 1: Basic tensor creation and operations
    torch.cuda.empty_cache()
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    torch.cuda.synchronize()
    start = time.time()
    c = a @ b  # Matrix multiplication
    torch.cuda.synchronize()
    print(f"  Matrix multiplication time: {time.time() - start:.4f}s")
    print(f"  Memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
    
    print("\n2. Convolutional operations with increasing size:")
    # Test 2: Conv operations with increasing size
    sizes = [512, 1024, 2048]
    if max_size > 2048:
        sizes.append(max_size)
    
    conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
    
    for size in sizes:
        try:
            torch.cuda.empty_cache()
            print(f"\n  Testing size {size}x{size}:")
            input_tensor = torch.randn(1, 3, size, size, device=device)
            print(f"  Input tensor created, shape: {input_tensor.shape}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
            
            torch.cuda.synchronize()
            start = time.time()
            output = conv(input_tensor)
            torch.cuda.synchronize()
            
            print(f"  Conv operation completed in {time.time() - start:.4f}s")
            print(f"  Output shape: {output.shape}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
        except RuntimeError as e:
            print(f"  Error on size {size}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PyTorch with ROCm")
    parser.add_argument("--max_size", type=int, default=4096, help="Maximum image size to test")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    args = parser.parse_args()
    
    run_benchmark(args.max_size, args.device)
