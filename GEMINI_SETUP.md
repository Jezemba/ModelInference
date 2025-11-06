# Running llama_medium.py with Gemini API

This guide explains how to run the Llama Medium benchmark using Google's Gemini API instead of loading models locally on GPUs.

## Why Use Gemini?

The original `llama_medium.py` requires loading a 17B parameter model locally, which needs significant GPU memory (doesn't fit on 2x 5090s). Google provides API access to Gemini vision models, allowing you to run inference without any local GPU requirements.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements_gemini.txt
```

Or install manually:
```bash
pip install google-generativeai pandas numpy datasets Pillow opencv-python tqdm
```

### 2. Get Your Google API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 3. Set Your API Key

**Linux/Mac:**
```bash
export GOOGLE_API_KEY='your_api_key_here'
```

**Windows (Command Prompt):**
```cmd
set GOOGLE_API_KEY=your_api_key_here
```

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

**Optional: Add to your shell profile for persistence**
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export GOOGLE_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Run the Script

```bash
python llama_medium_gemini.py
```

## Available Models

The script supports Google's Gemini vision models:

- **`gemini-2.0-flash-exp`** (Default) - Gemini 2.0 Flash, experimental and latest
- **`gemini-1.5-pro`** - Gemini 1.5 Pro, stable and more capable
- **`gemini-1.5-flash`** - Gemini 1.5 Flash, stable and faster

To change the model, edit the `MODEL_NAME` variable in the script:

```python
MODEL_NAME = "gemini-1.5-pro"  # Use Gemini 1.5 Pro
```

## Key Differences from Local Version

| Feature | Local (`llama_medium.py`) | Gemini API (`llama_medium_gemini.py`) |
|---------|---------------------------|----------------------------------------|
| GPU Required | Yes (2x 5090 insufficient) | No |
| Video Frames | 32 frames | 8 frames (API efficiency) |
| Model Loading | Downloads ~30GB+ | API call only |
| Speed | Depends on GPU | Fast (Google's infrastructure) |
| Cost | Hardware cost | API usage cost |

## Configuration Options

Edit these variables at the bottom of `llama_medium_gemini.py`:

```python
DATASET_NAME = "JessicaE/OpenSeeSimE-Structural"  # Your dataset
OUTPUT_FILE = "gemini_benchmark_results.csv"      # Output CSV file
CHECKPOINT_FILE = "gemini_checkpoint.pkl"         # Checkpoint for resume
MODEL_NAME = "gemini-2.0-flash-exp"               # Gemini model
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

### Error: "GOOGLE_API_KEY environment variable not set"
**Solution:** Set your API key as shown in step 3 above.

### Error: "Failed to load dataset"
**Solution:** Ensure you're authenticated with HuggingFace:
```bash
huggingface-cli login
```

### Rate Limiting
If you hit Gemini's rate limits, the script will show errors. You can:
1. Add retry logic with backoff
2. Slow down requests
3. Use a different model tier

### API Errors
If you encounter API errors:
1. Check your API key is valid
2. Verify you have API quota
3. Check [Google AI Studio](https://aistudio.google.com/) for service status

## Comparison with Original Code

The Gemini version maintains the same structure as the original:
- Same dataset loading
- Same prompt structure
- Same evaluation metrics
- Same checkpointing system

Main changes:
- No local model loading
- PIL images passed directly to Gemini API (no base64 conversion needed!)
- API calls instead of local inference
- Fewer video frames for efficiency
- Uses Google's GenerativeAI SDK

## Cost Estimation

Gemini pricing (as of current rates):

### Free Tier
- **Gemini 2.0 Flash**: 1,500 requests per day (free)
- **Gemini 1.5 Flash**: 15 requests per minute, 1,500 per day (free)
- **Gemini 1.5 Pro**: 2 requests per minute, 50 per day (free)

### Paid Tier (Pay-as-you-go)
Check [https://ai.google.dev/pricing](https://ai.google.dev/pricing) for current rates:
- **Gemini 1.5 Flash**: $0.075 per 1M input tokens, $0.30 per 1M output tokens
- **Gemini 1.5 Pro**: $1.25 per 1M input tokens, $5.00 per 1M output tokens

### Example Cost Calculation (Gemini 1.5 Flash)
For a dataset with:
- 1000 examples
- 50% images, 50% videos (8 frames each)
- Average response: 50 tokens

Estimated token usage:
- Images: 500 examples × ~258 tokens = 129K tokens
- Videos: 500 examples × 8 frames × 258 tokens = 1.03M tokens
- Text prompts: 1000 × ~200 tokens = 200K tokens
- Responses: 1000 × 50 tokens = 50K tokens

Total input: ~1.36M tokens × $0.075/1M = ~$0.10
Total output: ~50K tokens × $0.30/1M = ~$0.015
**Estimated total: ~$0.12** (very affordable!)

**Note:** Many users can run this entirely on the free tier!

## Performance Considerations

Gemini models are excellent for vision tasks:
- Strong understanding of images and videos
- Fast inference times
- Good at following structured output instructions
- Supports multiple images in a single request (great for video frames)
- PIL images can be passed directly (no encoding needed)

### Model Recommendations
- **For best quality**: Use `gemini-1.5-pro`
- **For speed/cost**: Use `gemini-1.5-flash` or `gemini-2.0-flash-exp`
- **For experimentation**: Use `gemini-2.0-flash-exp` (latest features)

## Need Help?

- Gemini Documentation: [https://ai.google.dev/](https://ai.google.dev/)
- API Reference: [https://ai.google.dev/api](https://ai.google.dev/api)
- Google AI Studio: [https://aistudio.google.com/](https://aistudio.google.com/)
- Community: [https://discuss.ai.google.dev/](https://discuss.ai.google.dev/)

## Comparison: Gemini vs Claude vs Groq

| Aspect | Gemini API | Claude API | Groq API |
|--------|-----------|-----------|----------|
| Model | gemini-2.0-flash-exp | claude-sonnet-4-5-20250929 | llama-3.2-90b-vision-preview |
| Provider | Google | Anthropic | Groq |
| Free Tier | Yes (generous limits) | No | Varies |
| Cost (paid) | ~$0.12/1K examples | ~$23/1K examples | Varies |
| Image Format | PIL images directly | Base64 encoded | Base64 encoded |
| Performance | Excellent | Excellent | Fast |

All three are excellent choices for running vision benchmarks without local GPU requirements!

## Tips for Best Results

1. **Start with free tier**: Test with a small subset on free tier before scaling
2. **Use Flash models**: For most tasks, Flash models provide excellent quality at lower cost
3. **Monitor quota**: Check your API usage in [Google AI Studio](https://aistudio.google.com/)
4. **Batch efficiently**: The script processes one example at a time with checkpointing
5. **Handle errors gracefully**: The script automatically saves progress on errors
