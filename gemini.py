#!/usr/bin/env python3
"""
Llama Medium VQA Benchmark - Gemini API Version
------------------------------------------------
This version uses Google's Gemini API for inference.
Supports both images and videos.
"""

import os
import io
import pandas as pd
import numpy as np
from datasets import load_dataset
from PIL import Image
import cv2
from tqdm import tqdm
import pickle
from pathlib import Path
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

def initialize_gemini_client(model_name):
    """
    Initialize Google Gemini API client.
    Requires GOOGLE_API_KEY environment variable to be set.

    Args:
        model_name: Name of the Gemini model to use

    Returns:
        Gemini GenerativeModel instance
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it with your Google API key.")

    print("Initializing Google Gemini API client...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    print(f"Gemini client initialized successfully with model: {model_name}")
    return model

def build_system_prompt():
    """Build a system prompt to guide Gemini's behavior"""
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

def prepare_gemini_content(example, media_type, frames=None):
    """
    Prepare content format for Gemini API with structured output.

    Args:
        example: Single example from dataset
        media_type: Either 'image' or 'video'
        frames: List of PIL Images (for video)

    Returns:
        content list in Gemini API format
    """
    question = example['question']
    answer_choices = example['answer_choices']

    # Build structured prompt
    system_prompt = build_system_prompt()
    prompt = f"{system_prompt}\n\nQuestion: {question}\n\n"

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

    # Format content based on media type
    # Gemini accepts PIL images directly in the content list
    content = []

    if media_type == 'image':
        # Single image
        content.append(example['image'])
        content.append(prompt)
    else:  # video
        # Multiple frames from video
        content.append(f"These are {len(frames)} frames from a video showing a sequence. {prompt}")
        for frame in frames:
            content.append(frame)

    return content

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

def run_inference_single(model, example, num_video_frames=8):
    """
    Run inference on a single example using Gemini API.

    Args:
        model: Gemini GenerativeModel instance
        example: Single example from dataset
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

        # Prepare content
        content = prepare_gemini_content(example, media_type, frames)

        # Call Gemini API with generation config for deterministic output
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,  # Deterministic for evaluation
            max_output_tokens=4096,  # Increased to handle longer explanations
        )

        # Set safety settings to be less restrictive for benchmark evaluation
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]

        response = model.generate_content(
            content,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Check if response was blocked or had issues
        if not response.candidates:
            error_msg = "No candidates returned by Gemini"
            print(f"\nWarning: {error_msg}")
            return (None, error_msg, "")

        candidate = response.candidates[0]

        # Check finish reason
        # finish_reason: 1=STOP (success), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
        if candidate.finish_reason != 1:  # Not STOP
            finish_reason_map = {
                2: "MAX_TOKENS",
                3: "SAFETY (content blocked by safety filters)",
                4: "RECITATION (blocked due to recitation)",
                5: "OTHER"
            }
            reason = finish_reason_map.get(candidate.finish_reason, f"UNKNOWN ({candidate.finish_reason})")

            # For MAX_TOKENS, we can still try to use the partial response
            # since the answer should be on the first line
            if candidate.finish_reason == 2:  # MAX_TOKENS
                print(f"\nInfo: Response reached MAX_TOKENS, attempting to extract answer from partial text...")
            else:
                print(f"\nWarning: Response blocked or incomplete. Finish reason: {reason}")

            # Try to get any partial text if available
            try:
                if candidate.content and candidate.content.parts:
                    partial_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                    if partial_text:
                        decoded = partial_text
                        answer, explanation = parse_model_response(decoded, example['answer_choices'])
                        # For MAX_TOKENS, accept the answer if we got it
                        if candidate.finish_reason == 2 and answer is not None:
                            return (answer, explanation, decoded)
                        elif answer is not None:
                            return (answer, explanation, decoded)
            except Exception as inner_e:
                print(f"\nFailed to extract partial text: {inner_e}")

            error_msg = f"Response blocked or incomplete. Finish reason: {reason}"
            return (None, error_msg, "")

        # Extract response text (only if finish_reason is STOP)
        try:
            decoded = response.text
        except Exception as e:
            error_msg = f"Failed to extract text: {str(e)}"
            print(f"\nWarning: {error_msg}")
            return (None, error_msg, "")

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

def run_benchmark(dataset_name, output_file='results_gemini.csv', checkpoint_file='checkpoint_gemini.pkl',
                  model_name="gemini-2.0-flash-exp", num_video_frames=8, media_type="all", max_workers=5):
    """
    Run benchmark evaluation on the entire dataset with checkpointing using Gemini API.

    Args:
        dataset_name: Name of the HuggingFace dataset
        output_file: CSV file to save results
        checkpoint_file: Pickle file to save checkpoints
        model_name: Name of the Gemini model to use
        num_video_frames: Number of frames to extract from videos (default: 8)
        media_type: Filter dataset by media type (image, video, or all)
        max_workers: Number of concurrent threads for parallel API calls (default: 5)

    Returns:
        Dictionary with evaluation metrics
    """
    # Load dataset
    dataset = load_benchmark_dataset(dataset_name)

    # Optional filtering by media type
    if media_type != "all":
        print(f"Filtering dataset for {media_type} examples only...")
        media_types = dataset["media_type"]
        filtered_indices = [i for i, mt in enumerate(media_types) if mt == media_type]
        dataset = dataset.select(filtered_indices)
        print(f"Remaining examples after filter: {len(dataset)}")

    # Initialize Gemini model
    model = initialize_gemini_client(model_name)

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

    # Process remaining examples with parallel threads
    print(f"\n🚀 Starting processing with Gemini API (parallel mode)...")
    print(f"   Model: {model_name}")
    print(f"   Total examples: {len(dataset)}")
    print(f"   Already processed: {len(processed_indices)}")
    print(f"   Remaining: {len(dataset) - len(processed_indices)}")
    print(f"   Concurrent workers: {max_workers}")

    # Get unprocessed indices
    unprocessed_indices = [i for i in range(len(dataset)) if i not in processed_indices]

    # Thread-safe lock for updating shared data
    lock = threading.Lock()

    # Helper function to process a single example
    def process_example(idx):
        try:
            # Try to load the example - may fail if image is corrupted
            example = dataset[idx]
        except (OSError, Exception) as e:
            print(f"\n⚠️  Skipping example {idx}: Corrupted or invalid image data ({str(e)[:50]})")
            with lock:
                processed_indices.add(idx)  # Mark as processed to skip in future runs
                # Save checkpoint to avoid retrying this example
                save_checkpoint(checkpoint_file, processed_indices, results)
            return idx, None, None, False, None

        try:
            # Run inference via Gemini API
            model_answer, explanation, full_response = run_inference_single(
                model, example, num_video_frames=num_video_frames
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
                'answer_choices': str(example['answer_choices']),  # Convert list to string for CSV
                'correct_choice_idx': example['correct_choice_idx'],
                'model': model_name,
                'model_answer': model_answer if model_answer is not None else 'None',
                'explanation': explanation,
                'correct': is_correct,
                'media_type': example['media_type']
            }

            return idx, result, model_answer, is_correct, example['question_type']

        except Exception as e:
            print(f"\n❌ Error processing example {idx}: {e}")
            return idx, None, None, False, None

    # Process examples in parallel using ThreadPoolExecutor
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_example, idx): idx for idx in unprocessed_indices}

        # Process completed futures with progress bar
        with tqdm(total=len(unprocessed_indices), desc="Processing examples") as pbar:
            for future in as_completed(future_to_idx):
                idx, result, model_answer, is_correct, q_type = future.result()

                if result is not None:
                    # Thread-safe update of shared state
                    with lock:
                        # Check if model failed to provide valid answer
                        if model_answer is None:
                            failed_answers += 1

                        # Update counters
                        if is_correct:
                            correct += 1
                        total += 1

                        # Update statistics by question type
                        if q_type:
                            if q_type not in stats_by_type:
                                stats_by_type[q_type] = {'correct': 0, 'total': 0}
                            stats_by_type[q_type]['total'] += 1
                            if is_correct:
                                stats_by_type[q_type]['correct'] += 1

                        # Store result
                        results.append(result)
                        processed_indices.add(idx)
                        completed_count += 1

                        # Save checkpoint periodically
                        if completed_count % 10 == 0 or completed_count == len(unprocessed_indices):
                            save_checkpoint(checkpoint_file, processed_indices, results)
                            # Save intermediate CSV
                            df = pd.DataFrame(results)
                            df.to_csv(output_file, index=False)

                pbar.update(1)

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
    print("BENCHMARK RESULTS (Gemini API)")
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
    import argparse

    parser = argparse.ArgumentParser(description="Run Gemini VQA Benchmark")
    parser.add_argument("--dataset_name", type=str, default="JessicaE/OpenSeeSimE-Structural",
                        help="HuggingFace dataset name")
    parser.add_argument("--output_file", type=str, default="gemini_benchmark_results.csv",
                        help="CSV output file")
    parser.add_argument("--checkpoint_file", type=str, default="gemini_checkpoint.pkl",
                        help="Checkpoint pickle file")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash",
                        help="Gemini model to use")
    parser.add_argument("--num_video_frames", type=int, default=8,
                        help="Number of frames to extract for video examples")
    parser.add_argument("--media_type", type=str, default="image", choices=["all", "image", "video"],
                        help="Filter dataset by media type")
    parser.add_argument("--max_workers", type=int, default=5,
                        help="Number of concurrent threads for parallel API calls (default: 5)")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("IMPORTANT: Set your Google API key before running:")
    print("  export GOOGLE_API_KEY='your_api_key_here'")
    print("=" * 80 + "\n")

    summary = run_benchmark(
        dataset_name=args.dataset_name,
        output_file=args.output_file,
        checkpoint_file=args.checkpoint_file,
        model_name=args.model_name,
        num_video_frames=args.num_video_frames,
        media_type=args.media_type,
        max_workers=args.max_workers
    )