# Qwen vLLM Troubleshooting - Python.h Missing Error

## Problem
When launching the vLLM server, you see this error:
```
fatal error: Python.h: No such file or directory
    5 | #include <Python.h>
      |          ^~~~~~~~~~
compilation terminated.
```

## Root Cause
Python development headers are not installed. These are required for Triton (used by vLLM) to compile CUDA utilities at runtime.

## Solution

### For Ubuntu/Debian:

```bash
# Check your Python version first
python3 --version

# If using Python 3.12 (as shown in your error)
sudo apt-get update
sudo apt-get install python3.12-dev

# If using Python 3.10
sudo apt-get install python3.10-dev

# If using Python 3.11
sudo apt-get install python3.11-dev

# Also install build essentials if not already installed
sudo apt-get install build-essential
```

### For Red Hat/CentOS/Fedora:

```bash
# For Python 3.12
sudo yum install python3.12-devel

# Or using dnf
sudo dnf install python3.12-devel

# Also install development tools
sudo yum groupinstall "Development Tools"
```

## Verify Installation

After installing the development headers, verify they're available:

```bash
# Check if Python.h is now available
ls /usr/include/python3.12/Python.h

# Should output: /usr/include/python3.12/Python.h
```

## Retry vLLM Server Launch

After installing the Python development headers, clear any cached compilation artifacts and retry:

```bash
# Clean up any partial compilation artifacts
rm -rf ~/.triton/

# Activate your virtual environment
source venv_qwen_vllm/bin/activate

# Now launch the server again
python qwen.py --mode server --model Qwen/Qwen3-VL-2B-Instruct --gpu-id 0
```

## Additional Dependencies (if still having issues)

If you continue to have compilation issues, you may also need:

```bash
# Install additional development libraries
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools

# For CUDA support
sudo apt-get install -y cuda-toolkit-12-1  # Or your CUDA version
```

## Alternative: Use System Python with venv

If you're still having issues, consider using system Python directly:

```bash
# Remove the old virtual environment
rm -rf venv_qwen_vllm

# Make sure python3-dev is installed
sudo apt-get install python3.12-dev build-essential

# Recreate virtual environment
python3 -m venv venv_qwen_vllm
source venv_qwen_vllm/bin/activate

# Reinstall everything
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install vllm
pip install -r requirements_qwen_vllm.txt
```

## Quick Fix One-Liner

For Ubuntu/Debian users with Python 3.12:

```bash
sudo apt-get update && sudo apt-get install -y python3.12-dev build-essential && rm -rf ~/.triton/
```

After running this, retry launching your vLLM server.
