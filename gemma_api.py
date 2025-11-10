import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import pickle
from pathlib import Path
import cv2
import tempfile
import shutil
from openai import OpenAI
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

def load_benchmark_dataset(dataset_name, split='test', media_type='all'):
    """
    Load the private benchmark dataset from HuggingFace.

    Args:
        dataset_name: Name of the dataset (e.g., 'JessicaE/OpenSeeSimE-Structural')
        split: Dataset split to load (default: 'test')
        media_type: 'video', 'image', or 'all' for all types (default: 'all')

    Returns:
        Dataset object
    """
    print(f"Loading dataset: {dataset_name}")
    dset = load_dataset(dataset_name, split=split, token=True)
    print(f"Loaded {len(dset)} examples")

    # Filter by media type using faster approach
    if media_type != "all":
        print(f"Filtering dataset for {media_type} examples only...")
        media_types = dset["media_type"]
        filtered_indices = [i for i, mt in enumerate(media_types) if mt == media_type]
        dset = dset.select(filtered_indices)
        print(f"Remaining examples after filter: {len(dset)}")

    return dset

def initialize_client(api_key=None):
    """
    Initialize Scaleway OpenAI client.

    Args:
        api_key: Scaleway API key (if None, reads from SCW_SECRET_KEY env variable)

    Returns:
        OpenAI client
    """
    if api_key is None:
        api_key = os.environ.get("SCW_SECRET_KEY")
        if not api_key:
            raise ValueError("SCW_SECRET_KEY environment variable must be set or api_key must be provided")

    client = OpenAI(
        base_url="https://api.scaleway.ai/b6a31dc6-b8a3-4347-98ae-3f7c3f9dff9d/v1",
        api_key=api_key
    )

    print("Scaleway API client initialized successfully")
    return client

def image_to_base64(image):
    """
    Convert PIL Image to base64 string.

    Args:
        image: PIL Image object

    Returns:
        base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

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

def prepare_messages(example, media_type, frames=None):
    """
    Prepare message format for API with structured output.

    Args:
        example: Single example from dataset
        media_type: Either 'image' or 'video'
        frames: List of PIL Images (for video)

    Returns:
        messages list in proper format
    """
    question = example['question']
    answer_choices = example['answer_choices']

    # Build structured prompt
    prompt = f"{question}\n\n"

    # Add answer choices
    if answer_choices and len(answer_choices) > 0:
        prompt += "Answer options:\n"
        for choice in answer_choices:
            prompt += f"- {choice}\n"
        prompt += "\n"

    # Add instruction for structured response
    prompt += "Instructions:\n"
    prompt += "1. First line: Provide ONLY your answer exactly as it appears in the options above (e.g., 'A', 'Yes', 'X axis', etc.). Do NOT add any other text on this line.\n"
    prompt += "2. Second line onwards: Provide a brief summary (1-2 sentences) explaining your reasoning.\n\n"
    prompt += "Answer:"

    # Get the system prompt
    system_prompt = build_system_prompt()

    # Format message based on media type
    # Note: Scaleway API uses OpenAI format which supports vision
    if media_type == 'image':
        # Convert image to base64
        image_base64 = image_to_base64(example['image'])

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    else:  # video
        # For video, we pass multiple frames as separate images
        content = []
        for frame in frames:
            frame_base64 = image_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{frame_base64}"
                }
            })
        content.append({
            "type": "text",
            "text": f"These are {len(frames)} frames from a video. {prompt}"
        })

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]

    return messages

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

def run_inference_single(client, example, num_video_frames=32, model_name="gemma-3-27b-it", max_retries=3):
    """
    Run inference on a single example using Scaleway API.

    Args:
        client: OpenAI client configured for Scaleway
        example: Single example from dataset
        num_video_frames: Number of frames to extract from video (default: 32)
        model_name: Model name for API
        max_retries: Maximum number of retries for API calls

    Returns:
        tuple: (answer, explanation, full_response)
    """
    media_type = example['media_type']

    try:
        # Create temporary directory for video processing if needed
        temp_dir = tempfile.mkdtemp(prefix="gemma_video_")

        try:
            # Prepare media
            frames = None
            if media_type == 'video':
                # Check if the video is a VideoDecoder object
                try:
                    from torchcodec.decoders import VideoDecoder
                    if isinstance(example['video'], VideoDecoder):
                        # Extract frames directly from VideoDecoder
                        frames = extract_frames_from_videodecoder(example['video'], num_frames=num_video_frames)
                    else:
                        # Extract frames from video path
                        frames = extract_video_frames(example['video'], num_frames=num_video_frames)
                except (ImportError, Exception):
                    # Fallback to standard video extraction
                    frames = extract_video_frames(example['video'], num_frames=num_video_frames)

                if not frames:
                    raise ValueError("Failed to extract frames from video")

            # Prepare messages
            messages = prepare_messages(example, media_type, frames)

            # Call API with retry logic
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=4096,
                        temperature=0,  # Deterministic
                        stream=False
                    )

                    # Extract response text
                    decoded = response.choices[0].message.content

                    # Parse structured response with validation
                    answer, explanation = parse_model_response(decoded, example['answer_choices'])

                    return (answer, explanation, decoded)

                except Exception as api_error:
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        print(f"API error, retrying in {wait_time}s: {str(api_error)}")
                        time.sleep(wait_time)
                    else:
                        raise

        except Exception as e:
            return (None, str(e), "")

        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    except Exception as e:
        return (None, str(e), "")

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

def load_checkpoint(checkpoint_file):
    """
    Load checkpoint if it exists.

    Args:
        checkpoint_file: Path to checkpoint file

    Returns:
        tuple: (processed_indices, results) or (set(), []) if no checkpoint
    """
    if os.path.exists(checkpoint_file):
        print(f"\n📂 Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)

        processed_indices = checkpoint_data['processed_indices']
        results = checkpoint_data['results']
        print(f"   Resuming from checkpoint: {len(processed_indices)} examples already processed")
        return processed_indices, results
    else:
        return set(), []

def save_checkpoint(checkpoint_file, processed_indices, results):
    """
    Save checkpoint.

    Args:
        checkpoint_file: Path to checkpoint file
        processed_indices: Set of processed example indices
        results: List of results
    """
    checkpoint_data = {
        'processed_indices': processed_indices,
        'results': results
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

def process_single_example(args):
    """
    Process a single example (for parallel execution).

    Args:
        args: Tuple of (client, dataset, idx, num_video_frames, model_name)

    Returns:
        tuple: (idx, result_dict) or (idx, None) if error
    """
    client, dataset, idx, num_video_frames, model_name = args

    try:
        # Try to access the example
        try:
            example = dataset[idx]
        except RuntimeError as e:
            error_msg = str(e)
            return (idx, {'error': f"Problematic file: {error_msg}", 'skipped': True})

        # Run inference
        model_answer, explanation, full_response = run_inference_single(
            client, example, num_video_frames=num_video_frames, model_name=model_name
        )

        # Evaluate
        is_correct = evaluate_response(
            model_answer,
            example['answer'],
            example['answer_choices']
        )

        # Store result with all original metadata
        result = {
            'file_name': example['file_name'],
            'source_file': example['source_file'],
            'question': example['question'],
            'question_type': example['question_type'],
            'question_id': example['question_id'],
            'answer': example['answer'],  # Ground truth
            'answer_choices': str(example['answer_choices']),
            'correct_choice_idx': example['correct_choice_idx'],
            'model': model_name,
            'model_answer': model_answer if model_answer is not None else 'None',
            'explanation': explanation,
            'correct': is_correct,
            'media_type': example['media_type'],
            'skipped': False
        }

        return (idx, result)

    except Exception as e:
        error_msg = str(e)
        return (idx, {'error': f"Processing error: {error_msg}", 'skipped': True})

def run_benchmark(dataset_name, output_file='results.csv', checkpoint_file='checkpoint.pkl',
                  model_name="gemma-3-27b-it", num_video_frames=32, media_type='all',
                  num_workers=4, api_key=None):
    """
    Run benchmark evaluation on the entire dataset with parallelization.

    Args:
        dataset_name: Name of the HuggingFace dataset
        output_file: CSV file to save results
        checkpoint_file: Pickle file to save checkpoints
        model_name: Name of the model to use with API
        num_video_frames: Number of frames to extract from videos (default: 32)
        media_type: Only process examples of this media type ('video', 'image', or 'all')
        num_workers: Number of parallel workers (default: 4)
        api_key: Scaleway API key (if None, reads from environment)

    Returns:
        Dictionary with evaluation metrics
    """
    # Load dataset
    dataset = load_benchmark_dataset(dataset_name, media_type=media_type)

    if len(dataset) == 0:
        print("No examples to process. Exiting.")
        return None

    # Initialize client
    client = initialize_client(api_key)

    # Load checkpoint if exists
    processed_indices, results = load_checkpoint(checkpoint_file)

    # Simple log file for problematic files
    problematic_log = os.path.splitext(checkpoint_file)[0] + '_problematic.log'

    # Statistics
    correct = sum(1 for r in results if r.get('correct', False))
    total = len(results)
    failed_answers = sum(1 for r in results if r.get('model_answer') == 'None')
    skipped_count = sum(1 for r in results if r.get('skipped', False))

    # Statistics by question type
    stats_by_type = {}
    for r in results:
        if r.get('skipped', False):
            continue
        q_type = r.get('question_type')
        if q_type:
            if q_type not in stats_by_type:
                stats_by_type[q_type] = {'correct': 0, 'total': 0}
            stats_by_type[q_type]['total'] += 1
            if r.get('correct', False):
                stats_by_type[q_type]['correct'] += 1

    # Process remaining examples
    print(f"\nStarting processing with {num_workers} workers...")
    print(f"   Total examples: {len(dataset)}")
    print(f"   Already processed: {len(processed_indices)}")
    print(f"   Remaining: {len(dataset) - len(processed_indices)}")

    # Get unprocessed indices
    unprocessed_indices = [i for i in range(len(dataset)) if i not in processed_indices]

    # Prepare arguments for parallel processing
    task_args = [(client, dataset, idx, num_video_frames, model_name) for idx in unprocessed_indices]

    # Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_single_example, args): args[2] for args in task_args}

        # Process completed tasks with progress bar
        with tqdm(total=len(unprocessed_indices), desc=f"Processing {media_type or 'examples'}") as pbar:
            for future in as_completed(future_to_idx):
                idx, result = future.result()

                if result is not None:
                    if result.get('skipped', False):
                        # Log problematic file
                        with open(problematic_log, 'a') as log:
                            log.write(f"{idx}: {result.get('error', 'Unknown error')}\n")
                        skipped_count += 1
                    else:
                        # Update statistics
                        if result['model_answer'] == 'None':
                            failed_answers += 1

                        if result['correct']:
                            correct += 1
                        total += 1

                        # Update statistics by question type
                        q_type = result['question_type']
                        if q_type not in stats_by_type:
                            stats_by_type[q_type] = {'correct': 0, 'total': 0}
                        stats_by_type[q_type]['total'] += 1
                        if result['correct']:
                            stats_by_type[q_type]['correct'] += 1

                    results.append(result)
                    processed_indices.add(idx)

                pbar.update(1)

                # Save checkpoint periodically
                if len(processed_indices) % 10 == 0:
                    save_checkpoint(checkpoint_file, processed_indices, results)

                # Save intermediate CSV
                if len(processed_indices) % 50 == 0:
                    df = pd.DataFrame([r for r in results if not r.get('skipped', False)])
                    df.to_csv(output_file, index=False)

    # Calculate accuracy
    overall_accuracy = correct / total if total > 0 else 0

    # Calculate accuracy by question type
    accuracy_by_type = {}
    for q_type, stats in stats_by_type.items():
        accuracy_by_type[q_type] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    # Save final results as CSV (excluding skipped examples)
    df = pd.DataFrame([r for r in results if not r.get('skipped', False)])
    df.to_csv(output_file, index=False)

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Overall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
    print(f"Failed to provide valid answer: {failed_answers}/{total} ({failed_answers/total*100:.1f}%)" if total > 0 else "Failed to provide valid answer: 0")
    print(f"Skipped problematic files: {skipped_count}")
    print(f"\nAccuracy by Question Type:")
    for q_type, acc in accuracy_by_type.items():
        stats = stats_by_type[q_type]
        print(f"  {q_type}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    print(f"\nResults saved to: {output_file}")
    print("="*80)

    # Prepare summary
    summary = {
        'model': model_name,
        'overall_accuracy': overall_accuracy,
        'correct': correct,
        'total': total,
        'failed_answers': failed_answers,
        'skipped_problematic': skipped_count,
        'accuracy_by_type': accuracy_by_type,
        'stats_by_type': stats_by_type
    }

    return summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 3 API Benchmark - Uses Scaleway API for gemma-3-27b-it")
    parser.add_argument("--dataset", type=str, default="JessicaE/OpenSeeSimE-Structural", help="HuggingFace dataset name")
    parser.add_argument("--model", type=str, default="gemma-3-27b-it",
                        help="Model name for Scaleway API (default: gemma-3-27b-it)")
    parser.add_argument("--output", type=str, default="gemma3_api_benchmark_results.csv", help="Output CSV file path")
    parser.add_argument("--checkpoint", type=str, default="gemma3_api_checkpoint.pkl", help="Checkpoint file path")
    parser.add_argument("--num_frames", type=int, default=32, help="Number of frames to extract from videos")
    parser.add_argument("--media_type", type=str, choices=['video', 'image', 'all'], default='all',
                        help="Only process examples of this media type (video, image, or all)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--api_key", type=str, default=None, help="Scaleway API key (default: read from SCW_SECRET_KEY env)")

    args = parser.parse_args()

    # Run benchmark
    summary = run_benchmark(
        args.dataset,
        args.output,
        checkpoint_file=args.checkpoint,
        model_name=args.model,
        num_video_frames=args.num_frames,
        media_type=args.media_type,
        num_workers=args.num_workers,
        api_key=args.api_key
    )
