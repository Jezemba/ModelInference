import argparse
import os
import time
import json
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
import tempfile
import pickle
import shutil
import base64
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import subprocess
import sys

### IMPORTANT: This script can either launch a vLLM server or process a dataset using an existing server.
### Use --mode=server to launch a server, or --mode=process to process a dataset.


def extract_frames_from_video(video_path_or_decoder, num_frames=32, temp_dir=None):
    """
    Extract frames from video path or VideoDecoder object.

    Args:
        video_path_or_decoder: Path to video file or VideoDecoder object
        num_frames: Number of frames to extract
        temp_dir: Temporary directory for processing

    Returns:
        List of base64-encoded frames
    """
    try:
        # Check if input is a VideoDecoder object
        from torchcodec.decoders import VideoDecoder
        if isinstance(video_path_or_decoder, VideoDecoder):
            frames = extract_frames_from_videodecoder(video_path_or_decoder, num_frames)
        else:
            # Regular video file path
            frames = extract_video_frames(video_path_or_decoder, num_frames)

        # Convert frames to base64
        base64_frames = [encode_image_to_base64(frame) for frame in frames]
        return base64_frames
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
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"

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

def prepare_prompt(question, answer_choices):
    """
    Prepare prompt with the system prompt format.

    Args:
        question: Question text
        answer_choices: List of answer choices

    Returns:
        Formatted prompt string
    """
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

def process_dataset(args):
    """
    Process the dataset using vLLM's API server.
    """
    # Connect to vLLM server
    client = OpenAI(
        api_key="EMPTY",  # Not needed for local vLLM
        base_url=f"http://{args.server_host}:{args.server_port}/v1",
        timeout=args.timeout
    )

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

    # Create temporary directory for processing if needed
    temp_dir = tempfile.mkdtemp(prefix="qwen_vllm_tmp_")

    try:
        # Process examples in order
        for idx in tqdm(unprocessed_indices, desc="Processing examples"):
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

                # Prepare prompt with new system prompt format
                question = example['question']
                answer_choices = example['answer_choices']
                prompt = prepare_prompt(question, answer_choices)

                # Create content array based on media type
                if media_type == 'image':
                    # Convert PIL Image to base64
                    base64_image = encode_image_to_base64(example['image'])

                    content = [
                        {"type": "image_url",
                        "image_url": {"url": base64_image}},
                        {"type": "text", "text": prompt}
                    ]
                else:  # video
                    # Extract frames from video
                    try:
                        # Extract frames from video
                        base64_frames = extract_frames_from_video(
                            example['video'],
                            num_frames=args.num_frames,
                            temp_dir=temp_dir
                        )

                        if not base64_frames or len(base64_frames) == 0:
                            raise ValueError("Failed to extract frames from video")

                        # Create content with multiple frames
                        content = []
                        for base64_frame in base64_frames:
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": base64_frame}
                            })

                        # Add text prompt at the end
                        content.append({
                            "type": "text",
                            "text": f"These are {len(base64_frames)} frames from a video. {prompt}"
                        })
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

                # Create messages for API
                messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]

                # Start timing
                start_time = time.time()

                # Make API request
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature
                )

                # Calculate inference time
                inference_time = time.time() - start_time

                # Get response text
                output_text = response.choices[0].message.content

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
                    'model': args.model,
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
        print("BENCHMARK RESULTS")
        print("="*80)
        print(f"\nModel: {args.model}")
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

def launch_server(args):
    """
    Launch a vLLM server for Qwen multimodal models.
    """
    # JSON format for limit-mm-per-prompt
    limit_mm_json = json.dumps({"image": 1, "video": 0})

    # Set CUDA_VISIBLE_DEVICES to use only the specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Using GPU: {args.gpu_id}")

    # Construct the command
    cmd = [
        "vllm", "serve", args.model,
        "--tensor-parallel-size", "1",  # Use only 1 GPU
        "--limit-mm-per-prompt", limit_mm_json,
        "--max-model-len", str(args.max_model_len),
        "--enable-chunked-prefill",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-seqs", str(args.max_num_seqs),
        "--host", args.server_host,
        "--port", str(args.server_port),
        "--dtype", "bfloat16"
    ]

    # Print the command being executed
    print("Launching vLLM server with command:")
    print(" ".join(cmd))

    # Execute the command
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="Qwen VL Unified Script - Launch server or process dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch vLLM server for 2B model
  python qwen.py --mode server --model Qwen/Qwen3-VL-2B-Instruct --gpu-id 0

  # Launch vLLM server for 8B model
  python qwen.py --mode server --model Qwen/Qwen3-VL-8B-Instruct --gpu-id 1

  # Process dataset with 2B model (images only)
  python qwen.py --mode process --model Qwen/Qwen3-VL-2B-Instruct --media-type image

  # Process dataset with 8B model (videos only)
  python qwen.py --mode process --model Qwen/Qwen3-VL-8B-Instruct --media-type video

  # Process all media types
  python qwen.py --mode process --model Qwen/Qwen3-VL-2B-Instruct --media-type all
        """
    )

    # Mode selection
    parser.add_argument("--mode", type=str, required=True, choices=['server', 'process'],
                       help="Mode: 'server' to launch vLLM server, 'process' to process dataset")

    # Model selection
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                       help="Model name (e.g., Qwen/Qwen3-VL-2B-Instruct or Qwen/Qwen3-VL-8B-Instruct)")

    # Server parameters (for both modes)
    parser.add_argument("--server-host", type=str, default="localhost",
                       help="Host of the vLLM server")
    parser.add_argument("--server-port", type=int, default=8000,
                       help="Port of the vLLM server")

    # Server-specific parameters
    parser.add_argument("--gpu-id", type=int, default=1,
                       help="Which GPU to use for server (default: 1)")
    parser.add_argument("--max-model-len", type=int, default=1024,
                       help="Maximum model context length for server")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                       help="Fraction of GPU memory to use for server")
    parser.add_argument("--max-num-seqs", type=int, default=8,
                       help="Maximum number of sequences to process in parallel")

    # Dataset processing parameters
    parser.add_argument("--dataset", type=str, default="JessicaE/OpenSeeSimE-Structural",
                       help="Dataset name on Hugging Face")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to use")
    parser.add_argument("--media-type", type=str, choices=['video', 'image', 'all'], default='all',
                       help="Filter by media type: 'image', 'video', or 'all'")
    parser.add_argument("--max-examples", type=int, default=0,
                       help="Maximum number of examples to process (0 = all)")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Timeout for API requests in seconds")

    # Model inference parameters
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for sampling")
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

    # Auto-generate output and checkpoint filenames based on model and media_type
    if args.mode == 'process':
        if args.output is None:
            model_short = args.model.split('/')[-1].lower().replace('-', '_')
            args.output = f"{model_short}_{args.media_type}_results.csv"

        if args.checkpoint is None:
            model_short = args.model.split('/')[-1].lower().replace('-', '_')
            args.checkpoint = f"{model_short}_{args.media_type}_checkpoint.pkl"

        # Convert max_examples to None if 0
        if args.max_examples == 0:
            args.max_examples = None

    # Execute based on mode
    if args.mode == 'server':
        print("\n" + "="*80)
        print("QWEN VL - LAUNCHING vLLM SERVER")
        print("="*80)
        print(f"Model: {args.model}")
        print(f"GPU: {args.gpu_id}")
        print(f"Host: {args.server_host}")
        print(f"Port: {args.server_port}")
        print(f"Max Model Length: {args.max_model_len}")
        print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
        print("="*80 + "\n")
        launch_server(args)
    else:  # process mode
        print("\n" + "="*80)
        print("QWEN VL - DATASET PROCESSING")
        print("="*80)
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Media Type Filter: {args.media_type}")
        print(f"Server: {args.server_host}:{args.server_port}")
        print(f"Output File: {args.output}")
        print(f"Checkpoint File: {args.checkpoint}")
        print("="*80 + "\n")
        process_dataset(args)

if __name__ == "__main__":
    main()
