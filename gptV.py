#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPT-5 VQA Video Benchmark (Fair Comparison with gemma.py)
----------------------------------------------------------

✅ Uses: OpenAI Responses API (GPT-5-2025-08-07)
✅ Supports: Videos from HuggingFace dataset (no AWS)
✅ Fair benchmark: 32 frames, 512 max tokens (matching gemma.py)
✅ Output: CSV with results + timing info
✅ Checkpointing: Resume from previous runs
"""

import os
import io
import json
import base64
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import pickle
import time

from openai import OpenAI
import cv2
import numpy as np

# -------------------------------------------------------------------
# CONFIGURATION (Fair Benchmark Settings)
# -------------------------------------------------------------------
MODEL_NAME = "gpt-5-2025-08-07"
MAX_OUTPUT_TOKENS = 512  # Match gemma.py (was 4096 in original)
REASONING_EFFORT = "minimal"
TEXT_VERBOSITY = "medium"

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def image_to_base64_data_uri(image):
    """
    Convert PIL Image to base64 data URI for GPT-5 API.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded data URI string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def extract_frames_from_videodecoder(video_decoder, num_frames=32):
    """
    Extract frames from a VideoDecoder object with middle frame guaranteed.

    Args:
        video_decoder: torchcodec VideoDecoder object
        num_frames: Number of frames to extract

    Returns:
        List of PIL Image objects
    """
    total_frames = len(video_decoder)

    # Calculate middle frame
    middle_frame = total_frames // 2

    # Generate frame indices with middle frame guaranteed (matching gemma.py)
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
    Extract uniformly sampled frames from video with middle frame guaranteed.

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

    # Generate frame indices with middle frame guaranteed (matching gemma.py)
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

    # Extract frames
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


def build_prompt(example):
    """Structured prompt identical to gemma.py logic."""
    question = example["question"]
    choices = example.get("answer_choices", [])
    prompt = f"{question}\n\n"
    if choices:
        prompt += "Answer options:\n"
        for c in choices:
            prompt += f"- {c}\n"
        prompt += "\n"
    prompt += (
        "Instructions:\n"
        "1. First line: Provide ONLY your answer exactly as it appears in the options above.\n"
        "2. Second line onwards: Provide a brief summary explaining your reasoning.\n\n"
        "Answer:"
    )
    return prompt


def parse_model_response(text, answer_choices):
    """Extract answer + explanation."""
    lines = text.strip().split("\n")
    if not lines:
        return None, ""
    first = lines[0].strip()
    explanation = "\n".join(lines[1:]).strip()
    if answer_choices:
        for c in answer_choices:
            if first.lower() == c.strip().lower():
                return c.strip(), explanation
        return None, explanation
    return first, explanation


def evaluate(model_answer, ground_truth, answer_choices):
    """Boolean correctness."""
    if model_answer is None:
        return False
    if model_answer.strip().lower() == ground_truth.strip().lower():
        return True
    if answer_choices:
        for c in answer_choices:
            if (
                model_answer.strip().lower() == c.strip().lower()
                and c.strip().lower() == ground_truth.strip().lower()
            ):
                return True
    return False


def openai_respond(user_content, system_prompt=None):
    """Wrapper for GPT-5 Responses API call with optional system prompt."""
    client = OpenAI()
    input_messages = []

    # Add system message if provided (OpenAI recommendation)
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})

    # Add user message
    input_messages.append({"role": "user", "content": user_content})

    resp = client.responses.create(
        model=MODEL_NAME,
        input=input_messages,
        reasoning={"effort": REASONING_EFFORT},
        text={"verbosity": TEXT_VERBOSITY},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return resp.output_text


# -------------------------------------------------------------------
# SINGLE EXAMPLE INFERENCE
# -------------------------------------------------------------------
def run_single_example(example, num_video_frames=32):
    """Handles video examples using HuggingFace dataset directly."""
    media_type = example["media_type"]

    if media_type != "video":
        raise ValueError(f"This script is for video benchmarking only. Got: {media_type}")

    prompt = build_prompt(example)
    system_prompt = build_system_prompt()

    # Get video from dataset
    video = example["video"]

    # Handle VideoDecoder objects or paths
    try:
        from torchcodec.decoders import VideoDecoder
        if isinstance(video, VideoDecoder):
            # Extract frames directly from VideoDecoder
            frames = extract_frames_from_videodecoder(video, num_frames=num_video_frames)
        else:
            # Extract frames from video path
            frames = extract_video_frames(video, num_frames=num_video_frames)
    except (ImportError, Exception):
        # Fallback to standard video extraction
        frames = extract_video_frames(video, num_frames=num_video_frames)

    if not frames:
        raise ValueError(f"No frames extracted from video {example.get('file_name', 'unknown')}")

    # Convert frames to base64 data URIs
    frames_base64 = []
    for frame in frames:
        frame_data_uri = image_to_base64_data_uri(frame)
        frames_base64.append(frame_data_uri)

    # Build GPT-5 multi-image input
    user_content = [
        {"type": "input_text", "text": f"These are {len(frames)} frames from a video. {prompt}"}
    ]
    for frame_b64 in frames_base64:
        user_content.append({"type": "input_image", "image_url": frame_b64})

    # Send to GPT-5
    text = openai_respond(user_content, system_prompt=system_prompt)
    ans, expl = parse_model_response(text, example.get("answer_choices"))
    return ans, expl, text


# -------------------------------------------------------------------
# CHECKPOINT HELPERS
# -------------------------------------------------------------------
def load_checkpoint(checkpoint_file):
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_file):
        print(f"\n📂 Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)

        processed_indices = checkpoint_data['processed_indices']
        results = checkpoint_data['results']
        problematic_indices = checkpoint_data.get('problematic_indices', set())

        print(f"   Resuming from checkpoint: {len(processed_indices)} examples already processed")
        if problematic_indices:
            print(f"   Skipping {len(problematic_indices)} problematic files")

        return processed_indices, results, problematic_indices
    else:
        return set(), [], set()


def save_checkpoint(checkpoint_file, processed_indices, results, problematic_indices):
    """Save checkpoint."""
    checkpoint_data = {
        'processed_indices': processed_indices,
        'results': results,
        'problematic_indices': problematic_indices
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)


# -------------------------------------------------------------------
# MAIN DRIVER
# -------------------------------------------------------------------
def run_benchmark(dataset_name, split="test", output_csv="gptV_results.csv",
                  checkpoint="gptV_checkpoint.pkl", max_examples=None,
                  num_video_frames=32):

    print(f"🔹 Loading dataset: {dataset_name}")
    dset = load_dataset(dataset_name, split=split, token=True)

    # Filter for video examples only
    print(f"Filtering dataset for video examples only...")
    video_indices = [i for i, m in enumerate(dset["media_type"]) if m == "video"]
    dset = dset.select(video_indices)
    print(f"✅ Total videos to process: {len(dset)}")

    total = len(dset)
    if max_examples:
        total = min(total, max_examples)
        dset = dset.select(range(total))

    # Load checkpoint
    processed_indices, results, problematic_indices = load_checkpoint(checkpoint)

    # Simple log file for problematic files
    problematic_log = os.path.splitext(checkpoint)[0] + '_problematic.log'

    # Statistics
    correct = sum(1 for r in results if r.get('correct', False))
    total_processed = len(results)
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

    # Get unprocessed indices
    unprocessed_indices = [i for i in range(total)
                          if i not in processed_indices and i not in problematic_indices]

    print(f"\n🚀 Starting GPT-5 Video Benchmark (Fair Comparison)...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Max Output Tokens: {MAX_OUTPUT_TOKENS} (matching gemma.py)")
    print(f"   Video Frames: {num_video_frames} (matching gemma.py)")
    print(f"   Total examples: {total}")
    print(f"   Already processed: {len(processed_indices)}")
    print(f"   Problematic examples to skip: {len(problematic_indices)}")
    print(f"   Remaining: {len(unprocessed_indices)}")

    # Process examples one by one
    for idx in tqdm(unprocessed_indices, desc="Processing videos"):
        try:
            start_time = time.time()

            # Try to access the example
            try:
                ex = dset[idx]
            except Exception as e:
                error_msg = str(e)
                print(f"Skipped problematic file at index {idx}: {error_msg}")

                with open(problematic_log, 'a') as log:
                    log.write(f"{idx}: {error_msg}\n")

                problematic_indices.add(idx)
                save_checkpoint(checkpoint, processed_indices, results, problematic_indices)
                continue

            # Run inference
            model_answer, explanation, full_text = run_single_example(ex, num_video_frames=num_video_frames)
            is_correct = evaluate(model_answer, ex["answer"], ex.get("answer_choices"))

            # Update counters
            if model_answer is None:
                failed_answers += 1
            if is_correct:
                correct += 1
            total_processed += 1

            # Calculate processing time
            processing_time = time.time() - start_time

            # Update statistics by question type
            q_type = ex.get('question_type')
            if q_type not in stats_by_type:
                stats_by_type[q_type] = {'correct': 0, 'total': 0}
            stats_by_type[q_type]['total'] += 1
            if is_correct:
                stats_by_type[q_type]['correct'] += 1

            # Store result
            res = {
                "file_name": ex.get("file_name"),
                "source_file": ex.get("source_file"),
                "question": ex.get("question"),
                "question_type": ex.get("question_type"),
                "question_id": ex.get("question_id"),
                "answer": ex.get("answer"),
                "answer_choices": str(ex.get("answer_choices")),
                "correct_choice_idx": ex.get("correct_choice_idx"),
                "model": MODEL_NAME,
                "model_answer": model_answer if model_answer else "None",
                "explanation": explanation,
                "correct": is_correct,
                "media_type": ex.get("media_type"),
                "processing_time": processing_time
            }
            results.append(res)
            processed_indices.add(idx)

            # Print timing every 10 examples
            if len(processed_indices) % 10 == 0:
                avg_time = sum(r.get("processing_time", 0) for r in results[-10:]) / min(10, len(results))
                print(f"\n⏱️  Avg time (last 10): {avg_time:.1f}s | Accuracy: {correct}/{total_processed} = {correct/total_processed:.1%}")

            # Save checkpoint periodically
            if len(processed_indices) % 5 == 0:
                save_checkpoint(checkpoint, processed_indices, results, problematic_indices)
                pd.DataFrame(results).to_csv(output_csv, index=False)

        except Exception as e:
            error_msg = str(e)
            print(f"Skipped problematic file at index {idx}: {error_msg}")

            with open(problematic_log, 'a') as log:
                log.write(f"{idx}: {error_msg}\n")

            problematic_indices.add(idx)
            save_checkpoint(checkpoint, processed_indices, results, problematic_indices)
            continue

    # Save final results
    all_df = pd.DataFrame(results)
    all_df.to_csv(output_csv, index=False)

    # Calculate final statistics
    acc = correct / total_processed if total_processed else 0
    accuracy_by_type = {}
    for q_type, stats in stats_by_type.items():
        accuracy_by_type[q_type] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    print("\n" + "=" * 80)
    print("GPT-5 VIDEO BENCHMARK RESULTS (Fair Comparison)")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Max Output Tokens: {MAX_OUTPUT_TOKENS}")
    print(f"Video Frames: {num_video_frames}")
    print(f"Overall Accuracy: {acc:.2%} ({correct}/{total_processed})")
    print(f"Failed to provide valid answer: {failed_answers}/{total_processed} ({failed_answers/total_processed*100:.1f}%)")
    print(f"Skipped problematic files: {len(problematic_indices)}")

    if stats_by_type:
        print(f"\nAccuracy by Question Type:")
        for q_type in sorted(stats_by_type.keys()):
            stats = stats_by_type[q_type]
            type_acc = accuracy_by_type[q_type]
            print(f"  {q_type}: {type_acc:.2%} ({stats['correct']}/{stats['total']})")

    print(f"\nResults saved to: {output_csv}")
    print("=" * 80)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GPT-5 Video Benchmark (Fair Comparison with gemma.py)")
    parser.add_argument("--dataset", type=str, default="JessicaE/OpenSeeSimE-Structural")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="gptV_results.csv")
    parser.add_argument("--checkpoint", type=str, default="gptV_checkpoint.pkl")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=32,
                       help="Number of frames to extract from videos (default: 32, matching gemma.py)")
    args = parser.parse_args()

    max_examples = None if args.max_examples <= 0 else args.max_examples

    print("\n" + "=" * 80)
    print("GPT-5 VIDEO BENCHMARK (Fair Comparison)")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {args.dataset}")
    print(f"Video Frames: {args.num_frames} (matching gemma.py baseline)")
    print(f"Max Output Tokens: {MAX_OUTPUT_TOKENS} (matching gemma.py baseline)")
    print(f"Output File: {args.output}")
    print(f"Checkpoint File: {args.checkpoint}")
    print("=" * 80 + "\n")

    run_benchmark(
        dataset_name=args.dataset,
        split=args.split,
        output_csv=args.output,
        checkpoint=args.checkpoint,
        max_examples=max_examples,
        num_video_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
