import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import pickle
import argparse
import sys
import base64
from io import BytesIO
import time
from openai import OpenAI

# API Configuration
API_BASE_URL = "https://chat.intern-ai.org.cn/api/v1/"
API_MODEL = "internvl3.5-241b-a28b"  # OpenGVLab/InternVL3_5-241B-A28B non-reasoning full 16fp

def get_api_key():
    """Get API key from environment variable"""
    api_key = os.environ.get("INTERNLM_API_KEY")
    if not api_key:
        raise ValueError(
            "INTERNLM_API_KEY environment variable not set. "
            "Please set it with: export INTERNLM_API_KEY=sk-yourapikey"
        )
    return api_key

def image_to_base64(image):
    """
    Convert PIL Image to base64 string.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded string with data URI prefix
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Save to BytesIO buffer
    buffered = BytesIO()
    image.save(buffered, format="PNG")

    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Return with data URI prefix
    return f"data:image/png;base64,{img_str}"

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
    Prepare prompt for InternVL3.5 API.

    Args:
        example: Single example from dataset

    Returns:
        question string (text only, without system prompt - that goes separately)
    """
    question = example['question']
    answer_choices = example['answer_choices']

    # Build user prompt (system prompt will be sent separately in API)
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

def call_api(image, prompt, system_prompt, max_retries=3, retry_delay=2):
    """
    Call InternLM API with image and prompt using OpenAI SDK.

    Args:
        image: PIL Image object
        prompt: User prompt text
        system_prompt: System prompt text
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        API response text or None if failed
    """
    api_key = get_api_key()

    # Initialize OpenAI client with InternLM endpoint
    client = OpenAI(
        api_key=api_key,
        base_url=API_BASE_URL,
    )

    # Convert image to base64
    image_base64 = image_to_base64(image)

    # Make API call with retries
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=API_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_base64
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0,  # Deterministic for benchmarking
                max_tokens=512
            )

            # Extract response text
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                print(f"Unexpected response format: {response}")
                return None

        except Exception as e:
            error_msg = str(e)
            print(f"API call error (attempt {attempt + 1}/{max_retries}): {error_msg}")

            # Check for rate limiting
            if "429" in error_msg or "rate limit" in error_msg.lower():
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue

            return None

    return None

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
    if not response_text:
        return None, ""

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

def run_inference_single(example):
    """
    Run inference on a single example using API.

    Args:
        example: Single example from dataset

    Returns:
        tuple: (answer, explanation, full_response)
    """
    media_type = example['media_type']

    # Currently only support images via API
    if media_type != 'image':
        return (None, f"Media type '{media_type}' not supported via API", "")

    try:
        # Get image
        image = example['image']
        if not isinstance(image, Image.Image):
            return (None, "Invalid image format", "")

        # Prepare prompts
        system_prompt = build_system_prompt()
        user_prompt = prepare_prompt(example)

        # Call API
        response = call_api(image, user_prompt, system_prompt)

        if response is None:
            return (None, "API call failed", "")

        # Parse structured response with validation
        answer, explanation = parse_model_response(response, example['answer_choices'])

        return (answer, explanation, response)

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
                  max_examples=None, media_type='image'):
    """
    Run benchmark evaluation on the entire dataset with checkpointing.

    Args:
        dataset_name: Name of the HuggingFace dataset
        output_file: CSV file to save results
        checkpoint_file: Pickle file to save checkpoints
        max_examples: Maximum number of examples to process (None = all)
        media_type: Filter by media type ('image' only supported via API)

    Returns:
        Dictionary with evaluation metrics
    """
    # Verify API key is available
    try:
        get_api_key()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return None

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

    # Rate limiting: 30 req/min = 1 request every 2 seconds
    min_time_between_requests = 2.0
    last_request_time = None

    # CSV saving optimization: save every N examples instead of every time
    csv_save_interval = 10
    examples_since_last_save = 0

    # Process one by one
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
                save_checkpoint(checkpoint_file, processed_indices, results, problematic_indices)

                # Skip to next example
                continue

            # Smart rate limiting: wait only if needed
            if last_request_time is not None:
                elapsed = time.time() - last_request_time
                if elapsed < min_time_between_requests:
                    wait_time = min_time_between_requests - elapsed
                    time.sleep(wait_time)

            # Record request start time
            last_request_time = time.time()

            # Run inference via API
            model_answer, explanation, full_response = run_inference_single(example)

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
                'model': API_MODEL,
                'model_answer': model_answer if model_answer is not None else 'None',
                'explanation': explanation,
                'correct': is_correct,
                'media_type': example['media_type']
            }
            results.append(result)
            processed_indices.add(idx)
            examples_since_last_save += 1

            # Save checkpoint after each example
            save_checkpoint(checkpoint_file, processed_indices, results, problematic_indices)

            # Save CSV periodically (every N examples) instead of every time
            if examples_since_last_save >= csv_save_interval:
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
                examples_since_last_save = 0

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
    print("BENCHMARK RESULTS - InternVL3.5-241B-A28B API")
    print("="*80)
    print(f"\nModel: {API_MODEL} (via API)")
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
        'model': API_MODEL,
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
    parser = argparse.ArgumentParser(description="InternVL3.5-241B-A28B API Benchmark")
    parser.add_argument("--dataset", type=str, default="JessicaE/OpenSeeSimE-Structural",
                      help="HuggingFace dataset name")
    parser.add_argument("--media_type", type=str, default="image", choices=["image"],
                      help="Media type: only 'image' supported via API")
    parser.add_argument("--output", type=str, default=None,
                      help="Output CSV file path (auto-generated if not specified)")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Checkpoint file path (auto-generated if not specified)")
    parser.add_argument("--max_examples", type=int, default=0,
                      help="Maximum number of examples to process (0 = all)")

    args = parser.parse_args()

    # Auto-generate output and checkpoint filenames
    if args.output is None:
        args.output = f"internvl3_5_241b_api_{args.media_type}_results.csv"

    if args.checkpoint is None:
        args.checkpoint = f"internvl3_5_241b_api_{args.media_type}_checkpoint.pkl"

    # Convert max_examples to None if 0
    max_examples = args.max_examples if args.max_examples > 0 else None

    print("\n" + "="*80)
    print("INTERNVL3.5-241B-A28B API BENCHMARK")
    print("="*80)
    print(f"Model: {API_MODEL} (via API)")
    print(f"Dataset: {args.dataset}")
    print(f"Media Type Filter: {args.media_type}")
    print(f"Output File: {args.output}")
    print(f"Checkpoint File: {args.checkpoint}")
    print("="*80 + "\n")

    # Run benchmark
    summary = run_benchmark(
        args.dataset,
        args.output,
        checkpoint_file=args.checkpoint,
        max_examples=max_examples,
        media_type=args.media_type
    )

    return 0 if summary is not None else 1

if __name__ == "__main__":
    # Configuration for direct script execution (without args)
    if len(sys.argv) > 1:
        # Use argparse if arguments provided
        exit(main())
    else:
        # Default configuration
        DATASET_NAME = "JessicaE/OpenSeeSimE-Structural"
        MEDIA_TYPE = "image"  # Only images supported via API
        OUTPUT_FILE = f"internvl3_5_241b_api_{MEDIA_TYPE}_results.csv"
        CHECKPOINT_FILE = f"internvl3_5_241b_api_{MEDIA_TYPE}_checkpoint.pkl"
        MAX_EXAMPLES = 0  # Set to a positive number to limit examples, 0 for all

        print("\n" + "="*80)
        print("INTERNVL3.5-241B-A28B API BENCHMARK (Default Configuration)")
        print("="*80)
        print(f"Model: {API_MODEL} (via API)")
        print(f"Dataset: {DATASET_NAME}")
        print(f"Media Type Filter: {MEDIA_TYPE}")
        print(f"Output File: {OUTPUT_FILE}")
        print(f"Checkpoint File: {CHECKPOINT_FILE}")
        print("="*80 + "\n")

        # Run benchmark
        summary = run_benchmark(
            DATASET_NAME,
            OUTPUT_FILE,
            checkpoint_file=CHECKPOINT_FILE,
            max_examples=MAX_EXAMPLES if MAX_EXAMPLES > 0 else None,
            media_type=MEDIA_TYPE
        )
