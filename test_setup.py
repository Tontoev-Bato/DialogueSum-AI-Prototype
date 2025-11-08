print("🚀 Project setup successful!")
print("Testing basic imports...")

try:
    import torch
    import transformers
    print("✅ All imports work correctly!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Import error: {e}")