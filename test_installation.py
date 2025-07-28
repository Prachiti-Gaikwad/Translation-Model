#!/usr/bin/env python3
"""
Test script to verify installation works without TensorFlow issues
"""

import sys
print("Python version:", sys.version)

# Test basic imports
try:
    import torch
    print("✅ PyTorch imported successfully")
    print("   PyTorch version:", torch.__version__)
    print("   CUDA available:", torch.cuda.is_available())
except ImportError as e:
    print("❌ PyTorch import failed:", e)

try:
    import transformers
    print("✅ Transformers imported successfully")
    print("   Transformers version:", transformers.__version__)
except ImportError as e:
    print("❌ Transformers import failed:", e)

try:
    from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
    print("✅ M2M-100 components imported successfully")
except ImportError as e:
    print("❌ M2M-100 import failed:", e)

try:
    import datasets
    print("✅ Datasets imported successfully")
except ImportError as e:
    print("❌ Datasets import failed:", e)

try:
    import pandas
    print("✅ Pandas imported successfully")
except ImportError as e:
    print("❌ Pandas import failed:", e)

try:
    import flask
    print("✅ Flask imported successfully")
except ImportError as e:
    print("❌ Flask import failed:", e)

# Test evaluation imports
try:
    from sacrebleu import BLEU
    print("✅ SacreBLEU imported successfully")
except ImportError as e:
    print("⚠️  SacreBLEU not available (will use simple BLEU):", e)

try:
    import nltk
    nltk.download('wordnet', quiet=True)
    from nltk.translate.meteor_score import meteor_score
    print("✅ NLTK METEOR imported successfully")
except ImportError as e:
    print("⚠️  NLTK METEOR not available (will use simple similarity):", e)

print("\n🎉 Installation test completed!")
print("If you see mostly ✅ marks, you're ready to run the main script!") 