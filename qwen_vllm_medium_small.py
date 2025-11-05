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

### IMPORTANT: This script requires a running vLLM API server.
### Please refer to VLM/ModelInference/run_vllm_server.py for instructions.


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
    print(f"Loaded {len(dataset)} examples")
    
    # Filter by media type if specified
    if args.media_type in ['video', 'image']:
        print(f"Filtering for {args.media_type} examples only...")
        # Use select with indices to avoid decoding during filter
        media_indices = []
        for idx in tqdm(range(len(dataset)), desc=f"Finding {args.media_type}s"):
            try:
                # Access the raw string field without decoding media
                media_type = dataset._data.table['media_type'][idx].as_py()
                if media_type == args.media_type:
                    media_indices.append(idx)
            except:
                # If we can't determine type, skip it
                continue
        dataset = dataset.select(media_indices)
        print(f"{args.media_type.capitalize()} examples after filtering: {len(dataset)}")
    
    # Load checkpoint
    processed_indices, results = load_checkpoint(args.checkpoint)
    
    # Statistics
    correct = sum(1 for r in results if r.get('correct', False))
    total = len(results)
    failed_answers = sum(1 for r in results if r.get('model_answer') == 'None')
    
    # Get unprocessed indices
    unprocessed_indices = [i for i in range(len(dataset)) if i not in processed_indices]
    
    # Process examples
    print(f"\n🚀 Starting processing...")
    print(f"   Total examples: {len(dataset)}")
    print(f"   Already processed: {len(processed_indices)}")
    print(f"   Remaining: {len(dataset) - len(processed_indices)}")
    
    # Create temporary directory for processing if needed
    temp_dir = tempfile.mkdtemp(prefix="qwen_vllm_tmp_")
    
    try:
        # Process examples in order
        for idx in tqdm(unprocessed_indices, desc="Processing examples"):
            example = dataset[idx]
            media_type = example['media_type']
            
            try:
                # Prepare prompt
                question = example['question']
                answer_choices = example['answer_choices']
                
                prompt = f"{question}\n\n"
                if answer_choices and len(answer_choices) > 0:
                    prompt += "Answer options:\n"
                    for choice in answer_choices:
                        prompt += f"- {choice}\n"
                    prompt += "\n"
                
                prompt += "Instructions:\n"
                prompt += "1. First line: Provide ONLY your answer exactly as it appears in the options above.\n"
                prompt += "2. Second line onwards: Provide a brief summary explaining your reasoning.\n\n"
                prompt += "Answer:"
                
                # Create content array based on media type
                # In the process_dataset function where it handles different media types
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
                        continue  # Skip this example if video processing fails
                
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
                if idx % args.save_interval == 0:
                    save_checkpoint(args.checkpoint, processed_indices, results)
                    
                    # Save intermediate CSV
                    df = pd.DataFrame(results)
                    df.to_csv(args.output, index=False)
                    print(f"\nCheckpoint saved after processing {len(processed_indices)} examples")
                    print(f"Current accuracy: {correct/total:.2%} ({correct}/{total})")
                    print(f"Average inference time: {sum(r['inference_time'] for r in results)/len(results):.2f}s per example")
                
            except Exception as e:
                print(f"\n❌ Error processing example {idx}: {e}")
                import traceback
                traceback.print_exc()
                # Save checkpoint on error
                save_checkpoint(args.checkpoint, processed_indices, results)
                continue
        
        # Calculate final statistics
        overall_accuracy = correct / total if total > 0 else 0
        
        # Save final results
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        
        # Remove checkpoint file on successful completion
        if os.path.exists(args.checkpoint) and len(unprocessed_indices) == len(processed_indices):
            os.remove(args.checkpoint)
            print(f"\n✅ Checkpoint file removed (processing complete)")
        
        # Print summary
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        print(f"\nModel: {args.model}")
        print(f"Overall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
        print(f"Failed to provide valid answer: {failed_answers}/{total} ({failed_answers/total*100:.1f}%)")
        print(f"Average inference time: {sum(r['inference_time'] for r in results)/len(results):.2f}s per example")
        print(f"\nResults saved to: {args.output}")
        print("="*80)
    
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Process dataset with vLLM API server")
    
    # Server connection parameters
    parser.add_argument("--server-host", type=str, default="localhost", 
                       help="Host of the vLLM server")
    parser.add_argument("--server-port", type=int, default=8000, 
                       help="Port of the vLLM server")
    parser.add_argument("--timeout", type=int, default=3600, 
                       help="Timeout for API requests in seconds")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="JessicaE/OpenSeeSimE-Structural", 
                       help="Dataset name on Hugging Face")
    parser.add_argument("--split", type=str, default="test", 
                       help="Dataset split to use")
    parser.add_argument("--media-type", type=str, choices=['video', 'image', 'all'], default='all',
                       help="Only process examples of this media type")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct", 
                       help="Model name for API requests")
    parser.add_argument("--max-tokens", type=int, default=512, 
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, 
                       help="Temperature for sampling")
    parser.add_argument("--num_frames", type=int, default=32, 
                   help="Number of frames to extract from videos")
    
    # Output parameters
    parser.add_argument("--output", type=str, default="qwen_vllm_benchmark_results.csv", 
                       help="Path to save results CSV")
    parser.add_argument("--checkpoint", type=str, default="qwen_vllm_checkpoint.pkl", 
                       help="Path to save checkpoint file")
    parser.add_argument("--save-interval", type=int, default=10, 
                       help="Save checkpoint every N examples")
    
    args = parser.parse_args()
    
    # Convert 'all' to None for media_type
    if args.media_type == 'all':
        args.media_type = None
    
    process_dataset(args)

if __name__ == "__main__":
    main()