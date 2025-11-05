# Running llama_medium.py with Groq API

This guide explains how to run the Llama Medium model using Groq's API instead of loading it locally on GPUs.

## Why Use Groq?

The original `llama_medium.py` requires loading a 17B parameter model locally, which needs significant GPU memory (doesn't fit on 2x 5090s). Groq provides API access to Llama vision models, allowing you to run inference without any local GPU requirements.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements_groq.txt
```

Or install manually:
```bash
pip install groq pandas numpy datasets Pillow opencv-python tqdm
```

### 2. Get Your Groq API Key

1. Go to [https://console.groq.com/](https://console.groq.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy your API key

### 3. Set Your API Key

**Linux/Mac:**
```bash
export GROQ_API_KEY='your_api_key_here'
```

**Windows (Command Prompt):**
```cmd
set GROQ_API_KEY=your_api_key_here
```

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

**Optional: Add to your shell profile for persistence**
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export GROQ_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Run the Script

```bash
python llama_medium_groq.py
```

## Available Models

The script supports Groq's Llama vision models:

- **`llama-3.2-90b-vision-preview`** (Default) - Larger, more capable model
- **`llama-3.2-11b-vision-preview`** - Smaller, faster model

To change the model, edit the `MODEL_NAME` variable in the script:

```python
MODEL_NAME = "llama-3.2-11b-vision-preview"  # Use smaller model
```

## Key Differences from Local Version

| Feature | Local (`llama_medium.py`) | Groq API (`llama_medium_groq.py`) |
|---------|---------------------------|-------------------------------------|
| GPU Required | Yes (2x 5090 insufficient) | No |
| Video Frames | 32 frames | 8 frames (API efficiency) |
| Model Loading | Downloads ~30GB+ | API call only |
| Speed | Depends on GPU | Fast (Groq's infrastructure) |
| Cost | Hardware cost | API usage cost |

## Configuration Options

Edit these variables at the bottom of `llama_medium_groq.py`:

```python
DATASET_NAME = "JessicaE/OpenSeeSimE-Structural"  # Your dataset
OUTPUT_FILE = "llama_groq_benchmark_results.csv"  # Output CSV file
CHECKPOINT_FILE = "llama_groq_checkpoint.pkl"     # Checkpoint for resume
MODEL_NAME = "llama-3.2-90b-vision-preview"       # Groq model
NUM_VIDEO_FRAMES = 8                               # Frames per video
```

## Checkpointing

The script automatically saves progress:
- Checkpoints are saved every 10 examples
- If interrupted, rerun the script to resume from last checkpoint
- On completion, checkpoint file is automatically deleted

## Output

Results are saved to a CSV file with columns:
- `file_name`, `source_file`, `question`, `question_type`, etc.
- `model_answer` - The model's prediction
- `explanation` - Model's reasoning
- `correct` - Boolean indicating if prediction matches ground truth
- `media_type` - Whether example is image or video

## Troubleshooting

### Error: "GROQ_API_KEY environment variable not set"
**Solution:** Set your API key as shown in step 3 above.

### Error: "Failed to load dataset"
**Solution:** Ensure you're authenticated with HuggingFace:
```bash
huggingface-cli login
```

### Rate Limiting
If you hit Groq's rate limits, the script will show errors. You can:
1. Add retry logic with backoff
2. Reduce concurrent requests
3. Upgrade your Groq plan

### API Errors
If you encounter API errors:
1. Check your API key is valid
2. Verify you have API credits/quota
3. Check [Groq's status page](https://status.groq.com/)

## Comparison with Original Code

The Groq version maintains the same structure as the original:
- Same dataset loading
- Same prompt structure
- Same evaluation metrics
- Same checkpointing system

Main changes:
- No local model loading
- Images/frames converted to base64
- API calls instead of local inference
- Fewer video frames for efficiency

## Cost Estimation

Groq pricing (as of current rates):
- Check [https://groq.com/pricing/](https://groq.com/pricing/) for current rates
- Vision models typically charge per image + per token
- Estimate costs based on your dataset size

## Need Help?

- Groq Documentation: [https://console.groq.com/docs](https://console.groq.com/docs)
- Groq Community: [https://discord.gg/groq](https://discord.gg/groq)
