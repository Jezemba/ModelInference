#!/usr/bin/env python3
"""
Llama Medium VQA Benchmark - Groq API Version
----------------------------------------------
This version uses Groq's API for inference instead of loading models locally.
Supports both images and videos.
"""

import os
import io
import base64
import pandas as pd
import numpy as np
from datasets import load_dataset
from PIL import Image
import cv2
from tqdm import tqdm
import pickle
from pathlib import Path
from groq import Groq

def load_benchmark_dataset(dataset_name, split='test'):
    """
    Load the private benchmark dataset from HuggingFace.

    Args:
        dataset_name: Name of the dataset (e.g., 'JessicaE/OpenSeeSimE-Structural')
        split: Dataset split to load (default: 'test')

    Returns:
        Dataset object
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split, token=True)
    print(f"Loaded {len(dataset)} examples")
    return dataset

def initialize_groq_client():
    """
    Initialize Groq API client.
    Requires GROQ_API_KEY environment variable to be set.

    Returns:
        Groq client
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set. Please set it with your Groq API key.")

    print("Initializing Groq API client...")
    client = Groq(api_key=api_key)
    print("Groq client initialized successfully")
    return client

def image_to_base64(image):
    """
    Convert PIL Image to base64 string.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded string with data URI prefix
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def extract_video_frames(video_path, num_frames=8):
    """
    Extract uniformly sampled frames from video.
    Note: Using fewer frames (8) for API efficiency compared to local inference (32).

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 8)

    Returns:
        List of PIL Image objects
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Could not read video: {video_path}")

    # Sample frames uniformly across the video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

    cap.release()
    return frames

def prepare_groq_messages(example, media_type, frames=None):
    """
    Prepare message format for Groq API with structured output.

    Args:
        example: Single example from dataset
        media_type: Either 'image' or 'video'
        frames: List of PIL Images (for video)

    Returns:
        messages list in Groq API format
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

    # Format message based on media type
    content = []

    if media_type == 'image':
        # Single image
        img_base64 = image_to_base64(example['image'])
        content.append({
            "type": "image_url",
            "image_url": {
                "url": img_base64
            }
        })
        content.append({
            "type": "text",
            "text": prompt
        })
    else:  # video
        # Multiple frames from video
        content.append({
            "type": "text",
            "text": f"These are {len(frames)} frames from a video showing a sequence. {prompt}"
        })
        for frame in frames:
            img_base64 = image_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_base64
                }
            })

    messages = [
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

def run_inference_single(client, example, model_name, num_video_frames=8):
    """
    Run inference on a single example using Groq API.

    Args:
        client: Groq API client
        example: Single example from dataset
        model_name: Name of the Groq model to use
        num_video_frames: Number of frames to extract from video (default: 8)

    Returns:
        tuple: (answer, explanation, full_response)
    """
    media_type = example['media_type']

    try:
        # Prepare media
        frames = None
        if media_type == 'video':
            # Extract frames from video
            frames = extract_video_frames(example['video'], num_frames=num_video_frames)
            if len(frames) == 0:
                raise ValueError("No frames extracted from video")

        # Prepare messages
        messages = prepare_groq_messages(example, media_type, frames)

        # Call Groq API
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=0.0,  # Deterministic for evaluation
            max_tokens=512,
        )

        # Extract response
        decoded = chat_completion.choices[0].message.content

        # Parse structured response with validation
        answer, explanation = parse_model_response(decoded, example['answer_choices'])

        return (answer, explanation, decoded)

    except Exception as e:
        print(f"\nError processing example: {e}")
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

def run_benchmark(dataset_name, output_file='results_groq.csv', checkpoint_file='checkpoint_groq.pkl',
                  model_name="llama-3.2-90b-vision-preview", num_video_frames=8):
    """
    Run benchmark evaluation on the entire dataset with checkpointing using Groq API.

    Args:
        dataset_name: Name of the HuggingFace dataset
        output_file: CSV file to save results
        checkpoint_file: Pickle file to save checkpoints
        model_name: Name of the Groq model to use
        num_video_frames: Number of frames to extract from videos (default: 8)

    Returns:
        Dictionary with evaluation metrics
    """
    # Load dataset
    dataset = load_benchmark_dataset(dataset_name)

    # Initialize Groq client
    client = initialize_groq_client()

    # Load checkpoint if exists
    processed_indices, results = load_checkpoint(checkpoint_file)

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

    # Process remaining examples one by one
    print(f"\n🚀 Starting processing with Groq API...")
    print(f"   Model: {model_name}")
    print(f"   Total examples: {len(dataset)}")
    print(f"   Already processed: {len(processed_indices)}")
    print(f"   Remaining: {len(dataset) - len(processed_indices)}")

    # Get unprocessed indices
    unprocessed_indices = [i for i in range(len(dataset)) if i not in processed_indices]

    # Process one by one
    for idx in tqdm(unprocessed_indices, desc="Processing examples"):
        example = dataset[idx]

        try:
            # Run inference via Groq API
            model_answer, explanation, full_response = run_inference_single(
                client, example, model_name, num_video_frames=num_video_frames
            )

            # Check if model failed to provide valid answer
            if model_answer is None:
                failed_answers += 1

            # Evaluate
            is_correct = evaluate_response(
                model_answer,
                example['answer'],
                example['answer_choices']
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

            # Store result with all original metadata
            result = {
                'file_name': example['file_name'],
                'source_file': example['source_file'],
                'question': example['question'],
                'question_type': example['question_type'],
                'question_id': example['question_id'],
                'answer': example['answer'],  # Ground truth
                'answer_choices': str(example['answer_choices']),  # Convert list to string for CSV
                'correct_choice_idx': example['correct_choice_idx'],
                'model': model_name,
                'model_answer': model_answer if model_answer is not None else 'None',
                'explanation': explanation,
                'correct': is_correct,
                'media_type': example['media_type']
            }
            results.append(result)
            processed_indices.add(idx)

            # Save checkpoint after each example
            if idx % 10 == 0 or idx == unprocessed_indices[-1]:  # Save every 10 examples or at the end
                save_checkpoint(checkpoint_file, processed_indices, results)

                # Save intermediate CSV
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)

        except Exception as e:
            print(f"\n❌ Error processing example {idx}: {e}")
            # Save checkpoint even on error
            save_checkpoint(checkpoint_file, processed_indices, results)
            continue

    # Calculate accuracy
    overall_accuracy = correct / total if total > 0 else 0

    # Calculate accuracy by question type
    accuracy_by_type = {}
    for q_type, stats in stats_by_type.items():
        accuracy_by_type[q_type] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    # Save final results as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    # Remove checkpoint file on successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\n✅ Checkpoint file removed (processing complete)")

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS (Groq API)")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Overall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
    print(f"Failed to provide valid answer: {failed_answers}/{total} ({failed_answers/total*100:.1f}%)")
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
        'accuracy_by_type': accuracy_by_type,
        'stats_by_type': stats_by_type
    }

    return summary

if __name__ == "__main__":
    # Configuration
    DATASET_NAME = "JessicaE/OpenSeeSimE-Structural"
    OUTPUT_FILE = "llama_groq_benchmark_results.csv"
    CHECKPOINT_FILE = "llama_groq_checkpoint.pkl"

    # Groq Vision Models:
    # - "llama-3.2-11b-vision-preview" (smaller, faster)
    # - "llama-3.2-90b-vision-preview" (larger, more capable - recommended)
    MODEL_NAME = "llama-3.2-90b-vision-preview"

    # Extract 8 frames from videos (fewer than local inference for API efficiency)
    NUM_VIDEO_FRAMES = 8

    print("\n" + "="*80)
    print("IMPORTANT: Set your Groq API key before running:")
    print("  export GROQ_API_KEY='your_api_key_here'")
    print("="*80 + "\n")

    # Run benchmark
    summary = run_benchmark(
        DATASET_NAME,
        OUTPUT_FILE,
        checkpoint_file=CHECKPOINT_FILE,
        model_name=MODEL_NAME,
        num_video_frames=NUM_VIDEO_FRAMES
    )
