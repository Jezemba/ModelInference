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

def initialize_model(model_name="OpenGVLab/InternVL3_5-14B-Instruct"):
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

def prepare_prompt(example):
    """
    Prepare prompt for InternVL3.5 model with structured output.
    
    Args:
        example: Single example from dataset
    
    Returns:
        question string
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

def run_inference_single(model, tokenizer, example):
    """
    Run inference on a single example.
    
    Args:
        model: InternVL3.5 model
        tokenizer: Tokenizer for the model
        example: Single example from dataset
    
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
            # Load video from path
            # Note: For videos, use max_num=1 (standard) and num_segments=32 (recommended)
            # Samples 32 frames uniformly across the video, ALWAYS including the middle frame
            # This produces ~8,192 visual tokens, well within the 32K context window
            pixel_values, num_patches_list = load_video_from_path(
                example['video'], 
                num_segments=32,  # Sample 32 frames uniformly (Video-MME standard)
                max_num=1,  # Each frame is single 448x448 tile (standard for videos)
                include_middle_frame=True  # ALWAYS include the exact middle frame
            )
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
        # Generation config
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

def run_benchmark(dataset_name, output_file='results.csv', checkpoint_file='checkpoint.pkl',
                  model_name="OpenGVLab/InternVL3_5-14B-Instruct"):
    """
    Run benchmark evaluation on the entire dataset with checkpointing.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        output_file: CSV file to save results
        checkpoint_file: Pickle file to save checkpoints
        model_name: Name of the model to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load dataset
    dataset = load_benchmark_dataset(dataset_name)
    
    # Initialize model
    model, tokenizer = initialize_model(model_name)
    
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
    print(f"\n🚀 Starting processing...")
    print(f"   Total examples: {len(dataset)}")
    print(f"   Already processed: {len(processed_indices)}")
    print(f"   Remaining: {len(dataset) - len(processed_indices)}")
    
    # Get unprocessed indices
    unprocessed_indices = [i for i in range(len(dataset)) if i not in processed_indices]
    
    # Process one by one
    for idx in tqdm(unprocessed_indices, desc="Processing examples"):
        example = dataset[idx]
        
        try:
            # Run inference
            model_answer, explanation, full_response = run_inference_single(model, tokenizer, example)
            
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
    print("BENCHMARK RESULTS")
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
    OUTPUT_FILE = "internvl3_5_14b_benchmark_results.csv"
    CHECKPOINT_FILE = "internvl3_5_14b_checkpoint.pkl"
    MODEL_NAME = "OpenGVLab/InternVL3_5-14B-Instruct"
    
    # Run benchmark
    summary = run_benchmark(
        DATASET_NAME, 
        OUTPUT_FILE, 
        checkpoint_file=CHECKPOINT_FILE,
        model_name=MODEL_NAME
    )