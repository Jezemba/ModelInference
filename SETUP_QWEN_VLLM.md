# Qwen VL + vLLM Setup Guide

This guide will help you set up a Python virtual environment for running Qwen VL models with vLLM.

## Prerequisites

- Python 3.10 or 3.11 (recommended)
- CUDA 11.8 or 12.1+ (for GPU support)
- At least 16GB GPU memory for 2B model, 40GB+ for 8B model
- **Python development headers** (CRITICAL - see below)
- Build tools (gcc, g++, make)

### 0. Install System Dependencies (REQUIRED)

**Ubuntu/Debian:**
```bash
# Install Python development headers (REQUIRED for vLLM/Triton)
sudo apt-get update
sudo apt-get install -y python3.12-dev build-essential

# Adjust python3.12-dev to match your Python version:
# - python3.10-dev for Python 3.10
# - python3.11-dev for Python 3.11
# - python3.12-dev for Python 3.12
```

**Red Hat/CentOS/Fedora:**
```bash
sudo yum install -y python3.12-devel
sudo yum groupinstall "Development Tools"

# Or with dnf:
sudo dnf install -y python3.12-devel
sudo dnf groupinstall "Development Tools"
```

**⚠️ Important:** Without Python development headers, you'll get a "Python.h: No such file or directory" error when launching vLLM. See `TROUBLESHOOTING_PYTHON_HEADERS.md` for details.

## Setup Steps

### 1. Create Virtual Environment

```bash
# Navigate to the project directory
cd /home/user/ModelInference

# Create a new virtual environment
python3 -m venv venv_qwen_vllm

# Activate the virtual environment
source venv_qwen_vllm/bin/activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch (with CUDA support)

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install vLLM

```bash
pip install vllm
```

### 5. Install Other Dependencies

```bash
pip install -r requirements_qwen_vllm.txt
```

### 6. Verify Installation

```bash
# Check vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### 7. Login to Hugging Face (for private datasets)

```bash
# Install huggingface-cli if not already installed
pip install huggingface-hub

# Login to Hugging Face
huggingface-cli login
```

Enter your Hugging Face token when prompted.

## Usage

### Launch vLLM Server

```bash
# Activate the virtual environment
source venv_qwen_vllm/bin/activate

# Launch server for 2B model
python qwen.py --mode server --model Qwen/Qwen3-VL-2B-Instruct --gpu-id 0

# Or for 8B model
python qwen.py --mode server --model Qwen/Qwen3-VL-8B-Instruct --gpu-id 1
```

### Process Dataset (in a separate terminal)

```bash
# Activate the virtual environment
source venv_qwen_vllm/bin/activate

# Process dataset
python qwen.py --mode process --model Qwen/Qwen3-VL-2B-Instruct --media-type all
```

## Troubleshooting

### Issue: CUDA out of memory

**Solution:** Reduce batch size or use smaller model
```bash
# For server mode, adjust these parameters:
python qwen.py --mode server \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --gpu-memory-utilization 0.6 \
    --max-num-seqs 4 \
    --max-model-len 512
```

### Issue: vLLM import error

**Solution:** Reinstall vLLM
```bash
pip uninstall vllm -y
pip install vllm --no-cache-dir
```

### Issue: Decord video loading error

**Solution:** Install decord from source or use conda
```bash
# Option 1: Try pip install
pip install decord

# Option 2: Use conda if available
conda install -c conda-forge decord
```

### Issue: Can't connect to vLLM server

**Solution:** Check if server is running and accessible
```bash
# Check if server is running on the correct port
curl http://localhost:8000/v1/models

# Or use netstat
netstat -tuln | grep 8000
```

## Environment Variables

You may want to set these environment variables:

```bash
# In your ~/.bashrc or ~/.zshrc
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export HF_HOME=/path/to/huggingface/cache  # Custom cache location
export TRANSFORMERS_CACHE=/path/to/transformers/cache
```

## Deactivating Virtual Environment

When you're done:
```bash
deactivate
```

## Complete Installation Script

Here's a one-shot installation script:

```bash
#!/bin/bash
# setup_qwen_vllm.sh

# Create and activate virtual environment
python3 -m venv venv_qwen_vllm
source venv_qwen_vllm/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.1 - adjust if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
pip install vllm

# Install other requirements
pip install -r requirements_qwen_vllm.txt

# Verify installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Setup complete! Activate the environment with: source venv_qwen_vllm/bin/activate"
```

Save this as `setup_qwen_vllm.sh`, make it executable with `chmod +x setup_qwen_vllm.sh`, and run it with `./setup_qwen_vllm.sh`.
