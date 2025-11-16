import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch
from tqdm import tqdm
import pickle
from pathlib import Path
import math
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
import tempfile
import cv2
import shutil
import argparse
import sys
import time

# Image preprocessing utilities
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_from_pil(image, input_size=448, max_num=12):
    """Load image from PIL Image object"""
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image")

    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32, include_middle=True):
    """
    Sample frame indices uniformly across the video.

    Args:
        bound: Optional time bounds [start, end] in seconds
        fps: Frames per second
        max_frame: Maximum frame index
        first_idx: First frame index (default 0)
        num_segments: Number of frames to sample
        include_middle: If True, ensures the middle frame is always sampled

    Returns:
        Array of frame indices
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)

    if include_middle and num_segments > 1:
        # Calculate the middle frame
        middle_frame = (start_idx + end_idx) // 2

        # For odd num_segments, middle frame will be naturally included at center position
        # For even num_segments, we force the middle frame to be included
        if num_segments % 2 == 1:
            # Odd number: sample symmetrically around middle
            half_segments = num_segments // 2
            # Sample frames before middle
            seg_size_before = (middle_frame - start_idx) / (half_segments + 0.5)
            indices_before = [
                int(start_idx + seg_size_before * (idx + 0.5))
                for idx in range(half_segments)
            ]
            # Sample frames after middle
            seg_size_after = (end_idx - middle_frame) / (half_segments + 0.5)
            indices_after = [
                int(middle_frame + seg_size_after * (idx + 0.5))
                for idx in range(1, half_segments + 1)
            ]
            # Combine: before + middle + after
            frame_indices = np.array(indices_before + [middle_frame] + indices_after)
        else:
            # Even number: force middle frame at position num_segments//2
            half_segments = num_segments // 2
            # Sample frames before middle (half_segments - 1 frames)
            if half_segments > 1:
                seg_size_before = (middle_frame - start_idx) / (half_segments - 0.5)
                indices_before = [
                    int(start_idx + seg_size_before * (idx + 0.5))
                    for idx in range(half_segments - 1)
                ]
            else:
                indices_before = []

            # Sample frames after middle (half_segments frames)
            seg_size_after = (end_idx - middle_frame) / (half_segments + 0.5)
            indices_after = [
                int(middle_frame + seg_size_after * (idx + 0.5))
                for idx in range(1, half_segments + 1)
            ]

            # Combine: before + middle + after
            frame_indices = np.array(indices_before + [middle_frame] + indices_after)
    else:
        # Standard uniform sampling without forcing middle frame
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])

    return frame_indices

def load_video_from_path(video_path, bound=None, input_size=448, max_num=1, num_segments=32, include_middle_frame=True):
    """
    Load video from file path.

    Args:
        video_path: Path to video file
        bound: Optional time bounds [start, end] in seconds
        input_size: Size to resize frames to (default 448)
        max_num: Max tiles per frame (default 1 for videos - each frame is single tile)
        num_segments: Number of frames to sample from video (default 32)
        include_middle_frame: If True, ensures middle frame is always sampled (default True)

    Note: For videos, max_num=1 is standard. Each frame becomes a single 448x448 tile.
          With 32 frames sampled uniformly (including the middle frame), this produces
          ~8,192 visual tokens, well within the 32K context window.
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments, include_middle=include_middle_frame)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def load_benchmark_dataset(dataset_name, split='test', media_type='all'):
    """
    Load the private benchmark dataset from HuggingFace, optionally filtered by media type.

    Args:
        dataset_name: Name of the dataset (e.g., 'JessicaE/OpenSeeSimE-Structural')
        split: Dataset split to load (default: 'test')
        media_type: 'image', 'video', or 'all' (default: 'all')

    Returns:
        Dataset object
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split, token=True)

    print(f"Total examples before filtering: {len(dataset)}")

    # Filter by media type if not 'all'
    if media_type != "all":
        print(f"Filtering dataset for {media_type} examples only...")
        media_types = dataset["media_type"]
        filtered_indices = [i for i, mt in enumerate(media_types) if mt == media_type]
        dataset = dataset.select(filtered_indices)
        print(f"Remaining examples after filter: {len(dataset)}")

    return dataset

def initialize_model(model_name="OpenGVLab/InternVL3_5-1B-Instruct"):
    """
    Initialize InternVL3.5 model and tokenizer.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name}")

    # Load model
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

    print("Model loaded successfully")
    return model, tokenizer

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

def prepare_prompt(example):
    """
    Prepare prompt for InternVL3.5 model with the new system prompt format.

    Args:
        example: Single example from dataset

    Returns:
        question string
    """
    question = example['question']
    answer_choices = example['answer_choices']

    # Build prompt with system instructions
    prompt = build_system_prompt() + "\n\n"

    # Add the question
    prompt += f"Question: {question}\n\n"

    # Add answer choices
    if answer_choices and len(answer_choices) > 0:
        prompt += "Options:\n"
        for choice in answer_choices:
            prompt += f"- {choice}\n"
        prompt += "\n"

    # Add instruction for structured response
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

def run_inference_single(model, tokenizer, example, num_video_frames=32):
    """
    Run inference on a single example.

    Args:
        model: InternVL3.5 model
        tokenizer: Tokenizer for the model
        example: Single example from dataset
        num_video_frames: Number of frames to extract from videos (default: 32)

    Returns:
        tuple: (answer, explanation, full_response)
    """
    media_type = example['media_type']
    prompt = prepare_prompt(example)

    try:
        # Prepare media (image or video)
        if media_type == 'image':
            # Load image from PIL Image
            pixel_values = load_image_from_pil(example['image'], max_num=12).to(torch.bfloat16).cuda()
            num_patches_list = None
        else:  # video
            # Handle video (including VideoDecoder objects)
            try:
                # First, check if the video is a VideoDecoder object
                from torchcodec.decoders import VideoDecoder
                if isinstance(example['video'], VideoDecoder):
                    # Extract frames directly from VideoDecoder
                    frames = extract_frames_from_videodecoder(example['video'], num_frames=num_video_frames)

                    # Process frames into pixel values
                    transform = build_transform(input_size=448)
                    pixel_values_list = []
                    num_patches_list = []

                    for frame in frames:
                        img = dynamic_preprocess(frame, image_size=448, use_thumbnail=True, max_num=1)
                        pixel_values = [transform(tile) for tile in img]
                        pixel_values = torch.stack(pixel_values)
                        num_patches_list.append(pixel_values.shape[0])
                        pixel_values_list.append(pixel_values)

                    pixel_values = torch.cat(pixel_values_list).to(torch.bfloat16).cuda()
                else:
                    # Use standard load_video_from_path for regular video path
                    pixel_values, num_patches_list = load_video_from_path(
                        example['video'],
                        num_segments=num_video_frames,
                        max_num=1,
                        include_middle_frame=True
                    )
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
            except (ImportError, Exception) as e:
                # Fallback to standard video path loading
                pixel_values, num_patches_list = load_video_from_path(
                    example['video'],
                    num_segments=num_video_frames,
                    max_num=1,
                    include_middle_frame=True
                )
                pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # Generation config - match gemma.py baseline (512 tokens, greedy)
        generation_config = dict(max_new_tokens=512, do_sample=False)

        # Generate response
        with torch.no_grad():
            if num_patches_list is not None:
                # Video case - need to format with frame prefixes
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                full_question = video_prefix + prompt
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    full_question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False
                )
            else:
                # Image case
                full_question = '<image>\n' + prompt
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    full_question,
                    generation_config,
                    history=None,
                    return_history=False
                )

        # Parse structured response with validation
        answer, explanation = parse_model_response(response, example['answer_choices'])

        return (answer, explanation, response)

    except Exception as e:
        import traceback
        traceback.print_exc()
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
    """
    checkpoint_data = {
        'processed_indices': processed_indices,
        'results': results,
        'problematic_indices': problematic_indices
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

def run_benchmark(dataset_name, output_file='results.csv', checkpoint_file='checkpoint.pkl',
                  model_name="OpenGVLab/InternVL3_5-1B-Instruct", max_examples=None,
                  media_type='all', num_video_frames=32):
    """
    Run benchmark evaluation on the entire dataset with checkpointing.

    Args:
        dataset_name: Name of the HuggingFace dataset
        output_file: CSV file to save results
        checkpoint_file: Pickle file to save checkpoints
        model_name: Name of the model to evaluate
        max_examples: Maximum number of examples to process (None = all)
        media_type: Filter by media type ('image', 'video', or 'all')
        num_video_frames: Number of frames to extract from videos (default: 32)

    Returns:
        Dictionary with evaluation metrics
    """
    # Load dataset (with media type filtering)
    dataset = load_benchmark_dataset(dataset_name, media_type=media_type)

    # Check if we have any examples to process
    if len(dataset) == 0:
        print(f"No {media_type} examples found in the dataset. Exiting.")
        return None

    # Limit examples if requested
    if max_examples is not None and max_examples > 0 and max_examples < len(dataset):
        print(f"Limiting to first {max_examples} examples")
        dataset = dataset.select(range(max_examples))

    # Initialize model
    model, tokenizer = initialize_model(model_name)

    # Load checkpoint if exists
    processed_indices, results, problematic_indices = load_checkpoint(checkpoint_file)

    # Simple log file for problematic files
    problematic_log = os.path.splitext(checkpoint_file)[0] + '_problematic.log'

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
    print(f"\n🚀 Starting processing...")
    print(f"   Total examples: {len(dataset)}")
    print(f"   Already processed: {len(processed_indices)}")
    print(f"   Problematic examples to skip: {len(problematic_indices)}")
    print(f"   Remaining: {len(dataset) - len(processed_indices) - len(problematic_indices)}")

    # Get unprocessed indices (excluding known problematic ones)
    unprocessed_indices = [i for i in range(len(dataset))
                          if i not in processed_indices and i not in problematic_indices]

    # Process one by one
    for idx in tqdm(unprocessed_indices, desc="Processing examples"):
        try:
            start_time = time.time()

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
                save_checkpoint(checkpoint_file, processed_indices, results, problematic_indices)

                # Skip to next example
                continue

            # Run inference
            model_answer, explanation, full_response = run_inference_single(
                model, tokenizer, example, num_video_frames=num_video_frames
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

            # Calculate processing time
            processing_time = time.time() - start_time

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
                'media_type': example['media_type'],
                'processing_time': processing_time
            }
            results.append(result)
            processed_indices.add(idx)

            # Print timing every 10 examples
            if len(processed_indices) % 10 == 0:
                avg_time = sum(r.get("processing_time", 0) for r in results[-10:]) / min(10, len(results))
                print(f"\n⏱️  Avg time (last 10): {avg_time:.1f}s | Accuracy: {correct}/{total} = {correct/total:.1%}")

            # Save checkpoint periodically
            if idx % 10 == 0:
                save_checkpoint(checkpoint_file, processed_indices, results, problematic_indices)

            # Save intermediate CSV
            if idx % 50 == 0:
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)

        except Exception as e:
            # General error during processing
            error_msg = str(e)
            print(f"Skipped problematic file at index {idx}: {error_msg}")

            # Log to file
            with open(problematic_log, 'a') as log:
                log.write(f"{idx}: {error_msg}\n")

            # Add to problematic indices
            problematic_indices.add(idx)

            # Save checkpoint even on error
            save_checkpoint(checkpoint_file, processed_indices, results, problematic_indices)
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

    # Remove checkpoint file on successful completion if requested
    if os.path.exists(checkpoint_file) and len(unprocessed_indices) == 0:
        os.remove(checkpoint_file)
        print(f"\n✅ Checkpoint file removed (processing complete)")

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Overall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
    print(f"Failed to provide valid answer: {failed_answers}/{total} ({failed_answers/total*100:.1f}%)")
    print(f"Skipped problematic files: {len(problematic_indices)}")
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
        'skipped_problematic': len(problematic_indices),
        'accuracy_by_type': accuracy_by_type,
        'stats_by_type': stats_by_type
    }

    return summary

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="InternVL Video Benchmark (Fair comparison with gemma.py)")
    parser.add_argument("--dataset", type=str, default="JessicaE/OpenSeeSimE-Structural",
                      help="HuggingFace dataset name")
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3_5-1B-Instruct",
                      help="Model name (e.g., OpenGVLab/InternVL3_5-1B-Instruct or OpenGVLab/InternVL3_5-4B-Instruct)")
    parser.add_argument("--media_type", type=str, default="all", choices=["image", "video", "all"],
                      help="Filter by media type: 'image', 'video', or 'all'")
    parser.add_argument("--output", type=str, default=None,
                      help="Output CSV file path (auto-generated if not specified)")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Checkpoint file path (auto-generated if not specified)")
    parser.add_argument("--max_examples", type=int, default=0,
                      help="Maximum number of examples to process (0 = all)")
    parser.add_argument("--num_frames", type=int, default=32,
                      help="Number of frames to extract from videos (default: 32, matching gemma.py)")

    args = parser.parse_args()

    # Auto-generate output and checkpoint filenames based on model and media_type
    if args.output is None:
        model_short = args.model.split('/')[-1].lower().replace('-', '_')
        args.output = f"{model_short}_{args.media_type}_results.csv"

    if args.checkpoint is None:
        model_short = args.model.split('/')[-1].lower().replace('-', '_')
        args.checkpoint = f"{model_short}_{args.media_type}_checkpoint.pkl"

    # Convert max_examples to None if 0
    max_examples = args.max_examples if args.max_examples > 0 else None

    print("\n" + "="*80)
    print("INTERNVL VIDEO BENCHMARK (Fair Comparison)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Media Type Filter: {args.media_type}")
    print(f"Num Video Frames: {args.num_frames} (matching gemma.py baseline)")
    print(f"Max New Tokens: 512 (matching gemma.py baseline)")
    print(f"Output File: {args.output}")
    print(f"Checkpoint File: {args.checkpoint}")
    print("="*80 + "\n")

    # Run benchmark
    summary = run_benchmark(
        args.dataset,
        args.output,
        checkpoint_file=args.checkpoint,
        model_name=args.model,
        max_examples=max_examples,
        media_type=args.media_type,
        num_video_frames=args.num_frames
    )

    return 0 if summary is not None else 1

if __name__ == "__main__":
    exit(main())
