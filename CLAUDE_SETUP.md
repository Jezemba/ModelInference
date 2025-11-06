# Running llama_medium.py with Claude API

This guide explains how to run the Llama Medium benchmark using Anthropic's Claude API instead of loading models locally on GPUs.

## Why Use Claude?

The original `llama_medium.py` requires loading a 17B parameter model locally, which needs significant GPU memory (doesn't fit on 2x 5090s). Anthropic provides API access to Claude vision models, allowing you to run inference without any local GPU requirements.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements_claude.txt
```

Or install manually:
```bash
pip install anthropic pandas numpy datasets Pillow opencv-python tqdm
```

### 2. Get Your Anthropic API Key

1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy your API key

### 3. Set Your API Key

**Linux/Mac:**
```bash
export ANTHROPIC_API_KEY='your_api_key_here'
```

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=your_api_key_here
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="your_api_key_here"
```

**Optional: Add to your shell profile for persistence**
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export ANTHROPIC_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Run the Script

```bash
python llama_medium_claude.py
```

## Available Models

The script supports Anthropic's Claude vision models:

- **`claude-sonnet-4-5-20250929`** (Default) - Claude 4.5 Sonnet, latest and most capable
- **`claude-3-5-sonnet-20241022`** - Claude 3.5 Sonnet
- **`claude-3-opus-20240229`** - Claude 3 Opus

To change the model, edit the `MODEL_NAME` variable in the script:

```python
MODEL_NAME = "claude-3-5-sonnet-20241022"  # Use Claude 3.5 Sonnet
```

## Key Differences from Local Version

| Feature | Local (`llama_medium.py`) | Claude API (`llama_medium_claude.py`) |
|---------|---------------------------|----------------------------------------|
| GPU Required | Yes (2x 5090 insufficient) | No |
| Video Frames | 32 frames | 8 frames (API efficiency) |
| Model Loading | Downloads ~30GB+ | API call only |
| Speed | Depends on GPU | Fast (Anthropic's infrastructure) |
| Cost | Hardware cost | API usage cost |

## Configuration Options

Edit these variables at the bottom of `llama_medium_claude.py`:

```python
DATASET_NAME = "JessicaE/OpenSeeSimE-Structural"  # Your dataset
OUTPUT_FILE = "claude_benchmark_results.csv"      # Output CSV file
CHECKPOINT_FILE = "claude_checkpoint.pkl"         # Checkpoint for resume
MODEL_NAME = "claude-sonnet-4-5-20250929"         # Claude model
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

### Error: "ANTHROPIC_API_KEY environment variable not set"
**Solution:** Set your API key as shown in step 3 above.

### Error: "Failed to load dataset"
**Solution:** Ensure you're authenticated with HuggingFace:
```bash
huggingface-cli login
```

### Rate Limiting
If you hit Anthropic's rate limits, the script will show errors. You can:
1. Add retry logic with backoff
2. Reduce concurrent requests
3. Upgrade your Anthropic plan

### API Errors
If you encounter API errors:
1. Check your API key is valid
2. Verify you have API credits/quota
3. Check [Anthropic's status page](https://status.anthropic.com/)

## Comparison with Original Code

The Claude version maintains the same structure as the original:
- Same dataset loading
- Same prompt structure
- Same evaluation metrics
- Same checkpointing system

Main changes:
- No local model loading
- Images/frames converted to base64
- API calls instead of local inference
- Fewer video frames for efficiency
- Uses Anthropic's Messages API format

## Cost Estimation

Anthropic pricing (as of current rates):
- Check [https://www.anthropic.com/pricing](https://www.anthropic.com/pricing) for current rates
- Vision models charge per image token + text token
- Claude 4.5 Sonnet: $3 per million input tokens / $15 per million output tokens
- Images typically count as ~1600 tokens each (varies by size)
- Estimate costs based on your dataset size

### Example Cost Calculation
For a dataset with:
- 1000 examples
- 50% images, 50% videos (8 frames each)
- Average response: 50 tokens

Estimated token usage:
- Images: 500 examples × 1600 tokens = 800K tokens
- Videos: 500 examples × 8 frames × 1600 tokens = 6.4M tokens
- Text prompts: 1000 × ~200 tokens = 200K tokens
- Responses: 1000 × 50 tokens = 50K tokens

Total input: ~7.4M tokens × $3/1M = ~$22.20
Total output: ~50K tokens × $15/1M = ~$0.75
**Estimated total: ~$23**

## Performance Considerations

Claude 4.5 Sonnet is highly capable for vision tasks:
- Excellent at understanding images and videos
- Strong reasoning capabilities
- Good at following structured output instructions
- Supports multiple images in a single request (great for video frames)

## Need Help?

- Anthropic Documentation: [https://docs.anthropic.com/](https://docs.anthropic.com/)
- Anthropic Support: [https://support.anthropic.com/](https://support.anthropic.com/)
- Community Discord: [https://www.anthropic.com/discord](https://www.anthropic.com/discord)

## Comparison: Claude vs Groq

| Aspect | Claude API | Groq API |
|--------|-----------|----------|
| Model | claude-sonnet-4-5-20250929 | llama-3.2-90b-vision-preview |
| Provider | Anthropic | Groq |
| Performance | Excellent vision understanding | Fast inference |
| Cost | ~$3-15/1M tokens | Varies, check Groq pricing |
| API Format | Messages API | Chat completions API |

Both are excellent choices for running vision benchmarks without local GPU requirements!
