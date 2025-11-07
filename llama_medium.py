import os
import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import pickle
from pathlib import Path
import cv2

# llama-cpp-python imports
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

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

def initialize_model(model_path, clip_model_path=None, n_ctx=4096, n_gpu_layers=-1):
    """
    Initialize Llama model using llama-cpp-python.

    Args:
        model_path: Path to GGUF model file
        clip_model_path: Path to CLIP model for vision (optional, for LLaVA models)
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU (-1 for all)

    Returns:
        Llama model instance
    """
    print(f"Loading model: {model_path}")

    # Initialize chat handler for vision if clip model is provided
    if clip_model_path and os.path.exists(clip_model_path):
        print(f"Loading vision support with CLIP model: {clip_model_path}")
        chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path)
        model = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
    else:
        # Text-only model
        print("Loading text-only model (no vision support)")
        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

    print("Model loaded successfully")
    return model

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

def create_grammar(answer_choices):
    """
    Create GBNF grammar for structured output with answer choices.

    The grammar enforces:
    - First line: Must be exactly one of the answer choices
    - Second line onward: Free-form explanation

    Args:
        answer_choices: List of valid answer choices

    Returns:
        GBNF grammar string
    """
    if not answer_choices or len(answer_choices) == 0:
        # Fallback grammar for free-form answers
        grammar = r'''
root ::= answer "\n" explanation
answer ::= [^\n]+
explanation ::= [^\n]+ ("\n" [^\n]+)*
'''
        return grammar

    # Escape special characters and create choice alternatives
    escaped_choices = []
    for choice in answer_choices:
        # Escape special GBNF characters
        escaped = choice.replace('\\', '\\\\').replace('"', '\\"')
        escaped_choices.append(f'"{escaped}"')

    # Join choices with | for alternatives
    choices_str = ' | '.join(escaped_choices)

    # Create grammar with strict answer matching
    grammar = f'''
root ::= answer "\\n" explanation
answer ::= {choices_str}
explanation ::= text-line ("\\n" text-line)*
text-line ::= [^\\n]+
'''

    return grammar

def prepare_prompt(example, media_type, vision_supported=False):
    """
    Prepare prompt text for the model.

    Args:
        example: Single example from dataset
        media_type: Either 'image' or 'video'
        vision_supported: Whether the model supports vision

    Returns:
        prompt string
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
    prompt += "1. First line: Provide ONLY your answer exactly as it appears in the options above. Do NOT add any other text on this line.\n"
    prompt += "2. Second line onwards: Provide a brief explanation (1-2 sentences) of your reasoning.\n\n"

    if not vision_supported:
        prompt += "Note: Vision is not supported. Please answer based on the question text only.\n\n"

    prompt += "Answer:"

    return prompt

def prepare_messages(example, media_type, vision_supported=False):
    """
    Prepare message format for llama-cpp-python.

    Args:
        example: Single example from dataset
        media_type: Either 'image' or 'video'
        vision_supported: Whether model supports vision

    Returns:
        messages list for chat completion
    """
    prompt = prepare_prompt(example, media_type, vision_supported)

    if vision_supported and media_type == 'image':
        # Vision model with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(example['image'])}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    else:
        # Text-only
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

    return messages

def encode_image(pil_image):
    """
    Encode PIL image to base64 string.

    Args:
        pil_image: PIL Image object

    Returns:
        Base64 encoded string
    """
    import base64
    from io import BytesIO

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

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
    explanation = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""

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

def run_inference_single(model, example, num_video_frames=32, vision_supported=False):
    """
    Run inference on a single example using llama-cpp-python with grammar.

    Args:
        model: Llama model instance
        example: Single example from dataset
        num_video_frames: Number of frames to extract from video (default: 32)
        vision_supported: Whether the model supports vision

    Returns:
        tuple: (answer, explanation, full_response)
    """
    media_type = example['media_type']

    try:
        # Create grammar based on answer choices
        grammar = create_grammar(example['answer_choices'])

        # Prepare prompt
        prompt = prepare_prompt(example, media_type, vision_supported)

        # For vision models, prepare with image
        if vision_supported and media_type == 'image':
            # Use chat completion with image
            messages = prepare_messages(example, media_type, vision_supported)
            response = model.create_chat_completion(
                messages=messages,
                grammar=grammar,
                max_tokens=512,
                temperature=0.0
            )
            decoded = response['choices'][0]['message']['content']
        else:
            # Text-only completion
            if media_type == 'video':
                # For video without vision support, just note it in prompt
                prompt = f"[VIDEO CONTENT - Vision not supported]\n{prompt}"

            # Use grammar-constrained generation
            response = model.create_completion(
                prompt=prompt,
                grammar=grammar,
                max_tokens=512,
                temperature=0.0,
                echo=False
            )
            decoded = response['choices'][0]['text']

        # Parse structured response with validation
        answer, explanation = parse_model_response(decoded, example['answer_choices'])

        return (answer, explanation, decoded)

    except Exception as e:
        print(f"\nError processing example: {e}")
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

def run_benchmark(dataset_name, model_path, output_file='results.csv', checkpoint_file='checkpoint.pkl',
                  clip_model_path=None, num_video_frames=32, media_type="all", n_ctx=4096, n_gpu_layers=-1):
    """
    Run benchmark evaluation on the entire dataset with checkpointing.

    Args:
        dataset_name: Name of the HuggingFace dataset
        model_path: Path to GGUF model file
        output_file: CSV file to save results
        checkpoint_file: Pickle file to save checkpoints
        clip_model_path: Path to CLIP model for vision support (optional)
        num_video_frames: Number of frames to extract from videos (default: 32)
        media_type: Filter by media type - 'image', 'video', or 'all' (default: 'all')
        n_ctx: Context window size (default: 4096)
        n_gpu_layers: Number of layers to offload to GPU (default: -1 for all)

    Returns:
        Dictionary with evaluation metrics
    """
    # Load dataset
    dataset = load_benchmark_dataset(dataset_name)

    # Optional filtering by media type
    if media_type != "all":
        print(f"Filtering dataset for {media_type} examples only...")
        filtered_indices = [
            i for i in range(len(dataset))
            if dataset[i]['media_type'] == media_type
        ]
        dataset = dataset.select(filtered_indices)
        print(f"Remaining examples after filter: {len(dataset)}")

    # Initialize model
    model = initialize_model(model_path, clip_model_path, n_ctx, n_gpu_layers)
    vision_supported = clip_model_path is not None and os.path.exists(clip_model_path)

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
    print(f"   Vision supported: {vision_supported}")

    # Get unprocessed indices
    unprocessed_indices = [i for i in range(len(dataset)) if i not in processed_indices]

    # Process one by one
    for idx in tqdm(unprocessed_indices, desc="Processing examples"):
        example = dataset[idx]

        try:
            # Run inference with grammar
            model_answer, explanation, full_response = run_inference_single(
                model, example, num_video_frames=num_video_frames, vision_supported=vision_supported
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
                'model': model_path,
                'model_answer': model_answer if model_answer is not None else 'None',
                'explanation': explanation,
                'full_response': full_response,
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
            import traceback
            traceback.print_exc()
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
    print(f"\nModel: {model_path}")
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
        'model': model_path,
        'overall_accuracy': overall_accuracy,
        'correct': correct,
        'total': total,
        'failed_answers': failed_answers,
        'accuracy_by_type': accuracy_by_type,
        'stats_by_type': stats_by_type
    }

    return summary

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Llama VQA Benchmark with llama-cpp-python Grammar Support"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="JessicaE/OpenSeeSimE-Structural",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llama_cpp_grammar_benchmark_results.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="llama_cpp_grammar_checkpoint.pkl",
        help="Checkpoint file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default=None,
        help="Path to CLIP model for vision support (for LLaVA models)"
    )
    parser.add_argument(
        "--num_video_frames",
        type=int,
        default=32,
        help="Number of frames to extract from videos"
    )
    parser.add_argument(
        "--media_type",
        type=str,
        choices=["image", "video", "all"],
        default="all",
        help="Filter dataset by media type (image, video, or all)"
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=4096,
        help="Context window size"
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 for all)"
    )

    args = parser.parse_args()

    # Validate model file exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        print("\nTo use this script, you need a GGUF format model.")
        print("You can download one from HuggingFace, for example:")
        print("  - Text models: https://huggingface.co/TheBloke")
        print("  - LLaVA models: https://huggingface.co/mys/ggml_llava-v1.5-7b")
        return

    # Run benchmark
    summary = run_benchmark(
        dataset_name=args.dataset,
        model_path=args.model,
        output_file=args.output,
        checkpoint_file=args.checkpoint,
        clip_model_path=args.clip_model,
        num_video_frames=args.num_video_frames,
        media_type=args.media_type,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers
    )


if __name__ == "__main__":
    main()
