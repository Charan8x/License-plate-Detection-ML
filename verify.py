"""
Run this to verify your environment is set up correctly before running detect.py
    python verify.py
"""

import sys

print("=" * 50)
print("  Environment Check")
print("=" * 50)

# Python version
print(f"\n[Python]  {sys.version.split()[0]}", end="")
major, minor = sys.version_info[:2]
if major == 3 and minor >= 10:
    print("  ✓")
else:
    print("  ✗  Need Python 3.10+")

# PyTorch
try:
    import torch
    print(f"[PyTorch] {torch.__version__}  ✓")
    if torch.cuda.is_available():
        print(f"[CUDA]    available  ✓")
        print(f"[GPU]     {torch.cuda.get_device_name(0)}  ✓")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM]    {vram:.1f} GB  ✓")
    else:
        print("[CUDA]    NOT available  ✗")
        print("          Reinstall PyTorch:")
        print("          pip uninstall torch torchvision -y")
        print("          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
except ImportError:
    print("[PyTorch] NOT installed  ✗")
    print("          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")

# OpenCV
try:
    import cv2
    print(f"[OpenCV]  {cv2.__version__}  ✓")
except ImportError:
    print("[OpenCV]  NOT installed  ✗  →  pip install opencv-python")

# Ultralytics
try:
    import ultralytics
    print(f"[YOLO]    ultralytics {ultralytics.__version__}  ✓")
except ImportError:
    print("[YOLO]    NOT installed  ✗  →  pip install ultralytics")

# EasyOCR
try:
    import easyocr
    print(f"[EasyOCR] installed  ✓")
except ImportError:
    print("[EasyOCR] NOT installed  ✗  →  pip install easyocr")

# HuggingFace Hub
try:
    import huggingface_hub
    print(f"[HF Hub]  {huggingface_hub.__version__}  ✓")
except ImportError:
    print("[HF Hub]  NOT installed  ✗  →  pip install huggingface_hub")

# Check model file
import os
model_path = os.path.join("models", "yolov11x-license-plate.pt")
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"[Model]   found ({size_mb:.0f} MB)  ✓")
else:
    print(f"[Model]   not downloaded yet — will auto-download on first run  (ok)")

# Check input video
if os.path.exists("input.mp4"):
    size_mb = os.path.getsize("input.mp4") / 1e6
    print(f"[Video]   input.mp4 found ({size_mb:.0f} MB)  ✓")
else:
    print(f"[Video]   input.mp4 NOT found  ✗  →  Drop your video here as input.mp4")

print("\n" + "=" * 50)
print("  If all lines show ✓ you are ready to run:")
print("  python detect.py")
print("=" * 50 + "\n")
