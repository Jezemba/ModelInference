#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3-VL-235B-A22B-Instruct VQA Benchmark — Using Alibaba Cloud API
----------------------------------------------------------------------------

✅ Uses: Alibaba Cloud DashScope API (OpenAI-compatible endpoint)
✅ Model: Qwen3-VL-235B-A22B-Instruct (235B parameters, 22B activated)
✅ Dataset: JessicaE/OpenSeeSimE-Structural (same as qwen.py)
✅ Prompt: Same system prompt and structure as qwen.py
✅ Output: Identical format to qwen.py results
✅ Features: Checkpoint support, video/image handling, rate limiting
✅ Parallelism: Support for multiple API keys to bypass rate limits

Key features:
- API-based inference (no local GPU needed)
- Processes both images and videos (extracts frames from videos)
- Checkpoint/resume capability for long-running jobs
- Rate limiting to stay within API quotas (100k TPM)
- Multiple API key support for parallel processing
"""

import argparse
import os
import time
import json
import pandas as pd
import pickle
import shutil
import base64
import tempfile
from io import BytesIO
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
import cv2
import numpy as np
from PIL import Image

# ----------------------------- CONFIG ---------------------------------
MODEL_ID = "qwen3-vl-235b-a22b-instruct"

# API Configuration
# Singapore region endpoint (use this for international access)
API_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
# For Beijing region, use: "https://dashscope.aliyuncs.com/compatible-mode/v1"

MAX_OUTPUT_TOKENS = 512

# Generation parameters
GEN_KW = dict(
    max_tokens=MAX_OUTPUT_TOKENS,
    temperature=0.7,
    top_p=0.8,
)

# Rate limiting: 100k tokens per minute limit
# Assuming ~2k tokens per request, that's ~50 requests/minute
# Safe to do 1 request per 2 seconds = 30 requests/minute
DEFAULT_API_DELAY = 2.0  # seconds between API calls

_CLIENTS = []  # List of API clients for multi-key support


# ------------------------ API CLIENT SETUP --------------------------
def initialize_api_clients(api_keys=None):
    """
    Initialize OpenAI client(s) for Alibaba Cloud API.

    Args:
        api_keys: List of API keys, or None to read from environment
    """
    global _CLIENTS

    if api_keys is None:
        # Try to read from environment
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY environment variable not set. "
                "Please obtain an API key from Alibaba Cloud Model Studio: "
                "https://www.alibabacloud.com/help/en/model-studio/get-api-key"
            )
        api_keys = [api_key]

    _CLIENTS = []
    for api_key in api_keys:
        client = OpenAI(
            api_key=api_key,
            base_url=API_BASE_URL,
        )
        _CLIENTS.append(client)

    print(f"Initialized {len(_CLIENTS)} API client(s)")


def get_api_client(index=0):
    """Get API client by index (for multi-key rotation)"""
    if not _CLIENTS:
        raise RuntimeError("API clients not initialized. Call initialize_api_clients() first.")
    return _CLIENTS[index % len(_CLIENTS)]


# ------------------------ IMAGE/VIDEO PROCESSING --------------------------
def extract_frames_from_video(video_path_or_decoder, num_frames=32, temp_dir=None):
    """
    Extract frames from video path or VideoDecoder object.

    Args:
        video_path_or_decoder: Path to video file or VideoDecoder object
        num_frames: Number of frames to extract
        temp_dir: Temporary directory for processing

    Returns:
        List of PIL Image objects
    """
    try:
        # Check if input is a VideoDecoder object
        from torchcodec.decoders import VideoDecoder
        if isinstance(video_path_or_decoder, VideoDecoder):
            frames = extract_frames_from_videodecoder(video_path_or_decoder, num_frames)
        else:
            # Regular video file path
            frames = extract_video_frames(video_path_or_decoder, num_frames)

        return frames
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return None


def extract_frames_from_videodecoder(video_decoder, num_frames=32):
    """
    Extract frames from a VideoDecoder object.

    Args:
        video_decoder: torchcodec VideoDecoder object
        num_frames: Number of frames to extract

    Returns:
        List of PIL Image objects
    """
    total_frames = len(video_decoder)

    # Calculate middle frame
    middle_frame = total_frames // 2

    # Generate frame indices with middle frame guaranteed
    if num_frames % 2 == 1:
        # Odd number: sample symmetrically around middle
        half_segments = num_frames // 2
        indices_before = np.linspace(0, middle_frame - 1, half_segments, dtype=int)
        indices_after = np.linspace(middle_frame + 1, total_frames - 1, half_segments, dtype=int)
        frame_indices = np.concatenate([indices_before, [middle_frame], indices_after])
    else:
        # Even number: force middle frame at position num_frames//2
        half_segments = num_frames // 2
        if half_segments > 1:
            indices_before = np.linspace(0, middle_frame - 1, half_segments - 1, dtype=int)
        else:
            indices_before = np.array([], dtype=int)
        indices_after = np.linspace(middle_frame + 1, total_frames - 1, half_segments, dtype=int)
        frame_indices = np.concatenate([indices_before, [middle_frame], indices_after])

    # Ensure we have exactly num_frames
    frame_indices = frame_indices[:num_frames]

    # Extract frames as PIL Images
    frames = []
    for idx in frame_indices:
        # Get tensor frame
        tensor_frame = video_decoder[int(idx)]
        # Convert to PIL
        frame_np = tensor_frame.permute(1, 2, 0).cpu().numpy()
        pil_image = Image.fromarray(frame_np.astype(np.uint8))
        frames.append(pil_image)

    return frames


def extract_video_frames(video_path, num_frames=32):
    """
    Extract uniformly sampled frames from video, ensuring middle frame is included.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 32)

    Returns:
        List of PIL Image objects
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    # Calculate middle frame
    middle_frame = total_frames // 2

    # Generate frame indices with middle frame guaranteed
    if num_frames % 2 == 1:
        # Odd number: sample symmetrically around middle
        half_segments = num_frames // 2
        indices_before = np.linspace(0, middle_frame - 1, half_segments, dtype=int)
        indices_after = np.linspace(middle_frame + 1, total_frames - 1, half_segments, dtype=int)
        frame_indices = np.concatenate([indices_before, [middle_frame], indices_after])
    else:
        # Even number: force middle frame at position num_frames//2
        half_segments = num_frames // 2
        if half_segments > 1:
            indices_before = np.linspace(0, middle_frame - 1, half_segments - 1, dtype=int)
        else:
            indices_before = np.array([], dtype=int)
        indices_after = np.linspace(middle_frame + 1, total_frames - 1, half_segments, dtype=int)
        frame_indices = np.concatenate([indices_before, [middle_frame], indices_after])

    # Ensure we have exactly num_frames
    frame_indices = frame_indices[:num_frames]

    # Extract frames and save to files
    frames = []
    for idx in frame_indices:
        # Set position
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

        # Read frame
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

    cap.release()
    return frames


def encode_image_to_base64(image):
    """
    Convert a PIL Image to base64-encoded string.

    Args:
        image: PIL Image object

    Returns:
        str: base64-encoded image string with data URI scheme
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_image}"


# ------------------------ PROMPT BUILDING ----------------------------
def build_system_prompt():
    """Build a system prompt to guide the model's behavior"""
    return (
        "You are a visual question answering assistant. You MUST follow this exact format:\n\n"
        "FORMAT REQUIREMENTS:\n"
        "Line 1: Copy the EXACT answer text from the provided options (word-for-word, including all symbols)\n"
        "Line 2: One brief explanation sentence (10-15 words)\n\n"
        "CRITICAL RULES:\n"
        "1. The first line MUST be an EXACT COPY of one option - do not paraphrase or summarize\n"
        "2. Copy ALL words, punctuation, and mathematical symbols exactly as shown in the option\n"
        "3. Do NOT add phrases like 'The answer is' or explanatory text on line 1\n"
        "4. Do NOT shorten or reword long options - copy them completely\n\n"
        "EXAMPLE 1 (Simple):\n"
        "Question: Is the sky blue?\n"
        "Options: Yes, No\n"
        "CORRECT:\n"
        "Yes\n"
        "The clear atmosphere scatters blue wavelengths effectively.\n\n"
        "EXAMPLE 2 (Complex option with symbols):\n"
        "Question: What is the range?\n"
        "Options: Less than 10× min, More than 1000× min\n"
        "CORRECT:\n"
        "More than 1000× min\n"
        "The values span from 7 billion to 1.6 trillion.\n\n"
        "INCORRECT:\n"
        "More than three orders of magnitude\n"
        "(This paraphrases instead of copying the exact option)\n\n"
        "Remember: Line 1 = EXACT COPY of option. Line 2 = explanation."
    )


def prepare_user_prompt(question, answer_choices):
    """
    Prepare user prompt (without system prompt).

    Args:
        question: Question text
        answer_choices: List of answer choices

    Returns:
        Formatted user prompt string
    """
    prompt = ""

    # Add the question
    prompt += f"Question: {question}\n\n"

    # Add answer choices
    if answer_choices and len(answer_choices) > 0:
        prompt += "Options:\n"
        for choice in answer_choices:
            prompt += f"- {choice}\n"
        prompt += "\n"

    # Add explicit instructions
    prompt += "Instructions:\n"
    prompt += "1. First line: Provide ONLY your answer exactly as it appears in the options above.\n"
    prompt += "2. Second line onwards: Provide a brief summary explaining your reasoning.\n\n"
    prompt += "Answer:"

    return prompt


def parse_model_response(response_text, answer_choices):
    """
    Parse the model's structured response into answer and explanation.
    Validates that the answer is one of the valid choices.

    Args:
        response_text: Raw text from model
        answer_choices: List of valid answer choices

    Returns:
        tuple: (answer, explanation) - answer is None if invalid
    """
    lines = response_text.strip().split('\n')

    if len(lines) == 0:
        return None, ""

    # First line is the answer
    potential_answer = lines[0].strip()

    # Rest is explanation
    explanation = '\n'.join(lines[1:]).strip()

    # Validate answer is in the choices
    if answer_choices and len(answer_choices) > 0:
        # Check if the potential answer matches any of the choices (case-insensitive)
        valid_answer = False
        matched_choice = None

        for choice in answer_choices:
            if potential_answer.lower() == choice.strip().lower():
                valid_answer = True
                matched_choice = choice.strip()
                break

        if not valid_answer:
            # Model failed to provide a valid answer from the options
            return None, explanation

        return matched_choice, explanation
    else:
        # No choices provided, accept whatever the model said
        return potential_answer, explanation


def evaluate_response(model_answer, ground_truth, answer_choices):
    """
    Evaluate model prediction against ground truth.

    Args:
        model_answer: Model's extracted answer (first line only), or None if invalid
        ground_truth: Correct answer
        answer_choices: List of answer choices

    Returns:
        Boolean indicating if prediction is correct
    """
    # If model failed to provide a valid answer, it's incorrect
    if model_answer is None:
        return False

    model_answer_clean = model_answer.strip()
    ground_truth_clean = ground_truth.strip()

    # Direct exact match (case-insensitive)
    if model_answer_clean.lower() == ground_truth_clean.lower():
        return True

    # Check if model answer exactly matches any of the choices
    # and that choice is the ground truth
    if answer_choices:
        for choice in answer_choices:
            if model_answer_clean.lower() == choice.strip().lower():
                if choice.strip().lower() == ground_truth_clean.lower():
                    return True

    return False


# ------------------------ CHECKPOINT MANAGEMENT --------------------------
def load_checkpoint(checkpoint_file):
    """
    Load checkpoint if it exists.

    Args:
        checkpoint_file: Path to checkpoint file

    Returns:
        tuple: (processed_indices, results, problematic_indices) or (set(), [], set()) if no checkpoint
    """
    if os.path.exists(checkpoint_file):
        print(f"\n📂 Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)

        processed_indices = checkpoint_data['processed_indices']
        results = checkpoint_data['results']
        # Get problematic indices with a default empty set if not in checkpoint
        problematic_indices = checkpoint_data.get('problematic_indices', set())

        print(f"   Resuming from checkpoint: {len(processed_indices)} examples already processed")
        if problematic_indices:
            print(f"   Skipping {len(problematic_indices)} problematic files")

        return processed_indices, results, problematic_indices
    else:
        return set(), [], set()


def save_checkpoint(checkpoint_file, processed_indices, results, problematic_indices):
    """
    Save checkpoint.

    Args:
        checkpoint_file: Path to checkpoint file
        processed_indices: Set of processed example indices
        results: List of results
        problematic_indices: Set of problematic example indices
    """
    checkpoint_data = {
        'processed_indices': processed_indices,
        'results': results,
        'problematic_indices': problematic_indices
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)


# ------------------------ API INFERENCE ------------------------------
def qwen_api_call(images, prompt, system_prompt, client_index=0, max_retries=3):
    """
    Call Alibaba Cloud API with images and prompt.

    Args:
        images: List of PIL Image objects
        prompt: User prompt text
        system_prompt: System prompt text
        client_index: Which API client to use (for multi-key rotation)
        max_retries: Maximum number of retries on failure

    Returns:
        str: Model's response text
    """
    client = get_api_client(client_index)

    # Build content array
    content = []
    for img in images:
        img_base64 = encode_image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": img_base64}
        })
    content.append({"type": "text", "text": prompt})

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

    # Retry loop
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                **GEN_KW
            )
            return completion.choices[0].message.content

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"API Error (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"API Error (final attempt): {e}")
                raise


# ------------------------ DATASET PROCESSING --------------------------
def process_dataset(args):
    """
    Process the dataset using Alibaba Cloud API.
    """
    # Initialize API clients
    api_keys = args.api_keys if hasattr(args, 'api_keys') and args.api_keys else None
    initialize_api_clients(api_keys)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split, token=True)
    print(f"Total examples before filtering: {len(dataset)}")

    # Filter by media type if specified
    if args.media_type != 'all':
        print(f"Filtering dataset for {args.media_type} examples only...")
        media_types = dataset["media_type"]
        filtered_indices = [i for i, mt in enumerate(media_types) if mt == args.media_type]
        dataset = dataset.select(filtered_indices)
        print(f"Remaining examples after filter: {len(dataset)}")

    # Limit examples if requested
    if args.max_examples is not None and args.max_examples > 0 and args.max_examples < len(dataset):
        print(f"Limiting to first {args.max_examples} examples")
        dataset = dataset.select(range(args.max_examples))

    # Load checkpoint
    processed_indices, results, problematic_indices = load_checkpoint(args.checkpoint)

    # Simple log file for problematic files
    problematic_log = os.path.splitext(args.checkpoint)[0] + '_problematic.log'

    # Statistics
    correct = sum(1 for r in results if r.get('correct', False))
    total = len(results)
    failed_answers = sum(1 for r in results if r.get('model_answer') == 'None')

    # Statistics by question type
    stats_by_type = {}
    for r in results:
        q_type = r.get('question_type')
        if q_type:
            if q_type not in stats_by_type:
                stats_by_type[q_type] = {'correct': 0, 'total': 0}
            stats_by_type[q_type]['total'] += 1
            if r.get('correct', False):
                stats_by_type[q_type]['correct'] += 1

    # Get unprocessed indices (excluding known problematic ones)
    unprocessed_indices = [i for i in range(len(dataset))
                          if i not in processed_indices and i not in problematic_indices]

    # Process examples
    print(f"\n🚀 Starting processing...")
    print(f"   Total examples: {len(dataset)}")
    print(f"   Already processed: {len(processed_indices)}")
    print(f"   Problematic examples to skip: {len(problematic_indices)}")
    print(f"   Remaining: {len(dataset) - len(processed_indices) - len(problematic_indices)}")
    print(f"   API delay: {args.api_delay} seconds between requests")
    print(f"   Number of API keys: {len(_CLIENTS)}")

    # Create temporary directory for processing if needed
    temp_dir = tempfile.mkdtemp(prefix="qwen_large_tmp_")

    try:
        # Process examples in order
        for idx in tqdm(unprocessed_indices, desc="Processing examples"):
            # Determine which API client to use (round-robin)
            client_index = idx % len(_CLIENTS)

            try:
                # Try to access the example - this is where errors might occur
                try:
                    example = dataset[idx]
                except Exception as e:
                    # Log the error
                    error_msg = str(e)
                    print(f"Skipped problematic file at index {idx}: {error_msg}")

                    # Log to file
                    with open(problematic_log, 'a') as log:
                        log.write(f"{idx}: {error_msg}\n")

                    # Add to problematic indices
                    problematic_indices.add(idx)

                    # Save checkpoint to record this problematic file
                    save_checkpoint(args.checkpoint, processed_indices, results, problematic_indices)

                    # Skip to next example
                    continue

                media_type = example['media_type']

                # Prepare prompts
                question = example['question']
                answer_choices = example['answer_choices']
                user_prompt = prepare_user_prompt(question, answer_choices)
                system_prompt = build_system_prompt()

                # Prepare images based on media type
                images = []

                if media_type == 'image':
                    # Single image
                    images = [example['image']]

                else:  # video
                    # Extract frames from video
                    try:
                        frames = extract_frames_from_video(
                            example['video'],
                            num_frames=args.num_frames,
                            temp_dir=temp_dir
                        )

                        if not frames or len(frames) == 0:
                            raise ValueError("Failed to extract frames from video")

                        images = frames

                        # Update prompt to indicate these are video frames
                        user_prompt = f"These are {len(frames)} frames from a video. {user_prompt}"

                    except Exception as e:
                        print(f"Error processing video for example {idx}: {e}")
                        import traceback
                        traceback.print_exc()

                        # Log to problematic
                        with open(problematic_log, 'a') as log:
                            log.write(f"{idx}: Video processing error - {e}\n")
                        problematic_indices.add(idx)
                        save_checkpoint(args.checkpoint, processed_indices, results, problematic_indices)
                        continue

                # Start timing
                start_time = time.time()

                # Make API request
                output_text = qwen_api_call(
                    images,
                    user_prompt,
                    system_prompt,
                    client_index=client_index,
                    max_retries=3
                )

                # Calculate inference time
                inference_time = time.time() - start_time

                # Parse and evaluate response
                model_answer, explanation = parse_model_response(output_text, answer_choices)

                # Check if model failed to provide valid answer
                if model_answer is None:
                    failed_answers += 1

                # Evaluate
                is_correct = evaluate_response(
                    model_answer,
                    example['answer'],
                    answer_choices
                )

                # Update counters
                if is_correct:
                    correct += 1
                total += 1

                # Update statistics by question type
                q_type = example['question_type']
                if q_type not in stats_by_type:
                    stats_by_type[q_type] = {'correct': 0, 'total': 0}
                stats_by_type[q_type]['total'] += 1
                if is_correct:
                    stats_by_type[q_type]['correct'] += 1

                # Store result
                result = {
                    'file_name': example['file_name'],
                    'source_file': example['source_file'],
                    'question': example['question'],
                    'question_type': example['question_type'],
                    'question_id': example['question_id'],
                    'answer': example['answer'],  # Ground truth
                    'answer_choices': str(answer_choices),  # Convert list to string for CSV
                    'correct_choice_idx': example['correct_choice_idx'],
                    'model': MODEL_ID,
                    'model_answer': model_answer if model_answer is not None else 'None',
                    'explanation': explanation,
                    'correct': is_correct,
                    'media_type': media_type,
                    'inference_time': inference_time
                }
                results.append(result)
                processed_indices.add(idx)

                # Save checkpoint periodically
                if len(processed_indices) % args.save_interval == 0:
                    save_checkpoint(args.checkpoint, processed_indices, results, problematic_indices)

                    # Save intermediate CSV
                    df = pd.DataFrame(results)
                    df.to_csv(args.output, index=False)
                    print(f"\nCheckpoint saved after processing {len(processed_indices)} examples")
                    print(f"Current accuracy: {correct/total:.2%} ({correct}/{total})")
                    if results:
                        print(f"Average inference time: {sum(r['inference_time'] for r in results)/len(results):.2f}s per example")

                # Rate limiting: sleep to avoid hitting API limits
                time.sleep(args.api_delay)

            except Exception as e:
                print(f"\n❌ Error processing example {idx}: {e}")
                import traceback
                traceback.print_exc()

                # Log to problematic
                with open(problematic_log, 'a') as log:
                    log.write(f"{idx}: General error - {e}\n")
                problematic_indices.add(idx)

                # Save checkpoint on error
                save_checkpoint(args.checkpoint, processed_indices, results, problematic_indices)
                continue

        # Calculate final statistics
        overall_accuracy = correct / total if total > 0 else 0

        # Calculate accuracy by question type
        accuracy_by_type = {}
        for q_type, stats in stats_by_type.items():
            accuracy_by_type[q_type] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

        # Save final results
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)

        # Remove checkpoint file on successful completion
        if os.path.exists(args.checkpoint) and len(unprocessed_indices) == 0:
            os.remove(args.checkpoint)
            print(f"\n✅ Checkpoint file removed (processing complete)")

        # Print summary
        print("\n" + "="*80)
        print("QWEN3-VL-235B-A22B-INSTRUCT API BENCHMARK RESULTS")
        print("="*80)
        print(f"\nModel: {MODEL_ID}")
        print(f"Overall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
        print(f"Failed to provide valid answer: {failed_answers}/{total} ({failed_answers/total*100:.1f}%)")
        print(f"Skipped problematic files: {len(problematic_indices)}")
        if results:
            print(f"Average inference time: {sum(r['inference_time'] for r in results)/len(results):.2f}s per example")
        print(f"\nAccuracy by Question Type:")
        for q_type, acc in accuracy_by_type.items():
            stats = stats_by_type[q_type]
            print(f"  {q_type}: {acc:.2%} ({stats['correct']}/{stats['total']})")
        print(f"\nResults saved to: {args.output}")
        print("="*80)

    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# ------------------------ CLI ----------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-235B-A22B-Instruct VQA using Alibaba Cloud API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process dataset with single API key (from environment)
  export DASHSCOPE_API_KEY='your-api-key'
  python qwen_large.py --dataset JessicaE/OpenSeeSimE-Structural --split test

  # Process only images with faster rate limiting
  python qwen_large.py --dataset JessicaE/OpenSeeSimE-Structural --media-type image --api-delay 1.0

  # Process only videos with multiple API keys
  python qwen_large.py --dataset JessicaE/OpenSeeSimE-Structural --media-type video \\
    --api-keys key1 key2 key3

  # Resume from checkpoint
  python qwen_large.py --dataset JessicaE/OpenSeeSimE-Structural --checkpoint qwen_large_checkpoint.pkl
        """
    )

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="JessicaE/OpenSeeSimE-Structural",
                       help="Dataset name on Hugging Face")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to use")
    parser.add_argument("--media-type", type=str, choices=['video', 'image', 'all'], default='all',
                       help="Filter by media type: 'image', 'video', or 'all'")
    parser.add_argument("--max-examples", type=int, default=0,
                       help="Maximum number of examples to process (0 = all)")

    # API parameters
    parser.add_argument("--api-keys", type=str, nargs='+', default=None,
                       help="List of API keys for parallel processing (uses DASHSCOPE_API_KEY env if not provided)")
    parser.add_argument("--api-delay", type=float, default=DEFAULT_API_DELAY,
                       help="Delay in seconds between API calls (default: 2.0)")
    parser.add_argument("--region", type=str, choices=["singapore", "beijing"], default="singapore",
                       help="API region: singapore (international) or beijing (China)")

    # Model inference parameters
    parser.add_argument("--num-frames", type=int, default=32,
                       help="Number of frames to extract from videos")

    # Output parameters
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save results CSV (auto-generated if not specified)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to save checkpoint file (auto-generated if not specified)")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Save checkpoint every N examples")

    args = parser.parse_args()

    # Set API endpoint based on region
    global API_BASE_URL
    if args.region == "beijing":
        API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Auto-generate output and checkpoint filenames based on model and media_type
    if args.output is None:
        args.output = f"qwen_large_{args.media_type}_results.csv"

    if args.checkpoint is None:
        args.checkpoint = f"qwen_large_{args.media_type}_checkpoint.pkl"

    # Convert max_examples to None if 0
    if args.max_examples == 0:
        args.max_examples = None

    # Print configuration
    print("\n" + "="*80)
    print("QWEN3-VL-235B-A22B-INSTRUCT - DATASET PROCESSING (API)")
    print("="*80)
    print(f"Model: {MODEL_ID}")
    print(f"API Endpoint: {API_BASE_URL}")
    print(f"Dataset: {args.dataset}")
    print(f"Media Type Filter: {args.media_type}")
    print(f"Output File: {args.output}")
    print(f"Checkpoint File: {args.checkpoint}")
    print(f"API Delay: {args.api_delay}s between requests")
    if args.api_keys:
        print(f"Number of API keys: {len(args.api_keys)}")
    print("="*80 + "\n")

    # Process dataset
    process_dataset(args)


if __name__ == "__main__":
    main()
