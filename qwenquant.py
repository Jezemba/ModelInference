import argparse
import os
import time
import json
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
import pickle
from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer

### IMPORTANT: This script processes image datasets using quantized Qwen3-VL models via unsloth.
### Use --quant to select quantization level: 4bit, 8bit, or 16bit
### Only processes images (videos are not supported)

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

def prepare_user_prompt(question, answer_choices):
    """
    Prepare user prompt (without system prompt).

    Args:
        question: Question text
        answer_choices: List of answer choices

    Returns:
        Formatted user prompt string
    """
    prompt = ""

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

def load_quantized_model(quant_level):
    """
    Load quantized Qwen3-VL model using unsloth.

    Args:
        quant_level: Quantization level (4bit, 8bit, 16bit)

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"\n🔧 Loading Qwen3-VL-8B-Instruct with {quant_level} quantization...")

    # Map quantization level to unsloth parameters
    if quant_level == "4bit":
        load_in_4bit = True
        load_in_8bit = False
    elif quant_level == "8bit":
        load_in_4bit = False
        load_in_8bit = True
    elif quant_level == "16bit":
        load_in_4bit = False
        load_in_8bit = False
    else:
        raise ValueError(f"Invalid quantization level: {quant_level}. Choose from: 4bit, 8bit, 16bit")

    # Load model using unsloth
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3-VL-8B-Instruct",
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    # Prepare for inference
    FastVisionModel.for_inference(model)

    print("   ✅ Model loaded successfully")
    return model, tokenizer

def process_dataset(args):
    """
    Process the image dataset using quantized Qwen3-VL model.
    """
    # Load the quantized model
    model, tokenizer = load_quantized_model(args.quant)

    # Load dataset
    print(f"\n📚 Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split, token=True)
    print(f"Total examples before filtering: {len(dataset)}")

    # Filter for images only
    print(f"Filtering dataset for image examples only...")
    media_types = dataset["media_type"]
    filtered_indices = [i for i, mt in enumerate(media_types) if mt == 'image']
    dataset = dataset.select(filtered_indices)
    print(f"Remaining image examples: {len(dataset)}")

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

    try:
        # Process examples in order
        for idx in tqdm(unprocessed_indices, desc="Processing examples"):
            try:
                # Try to access the example
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

                # Get image (PIL Image object)
                image = example['image']

                # Prepare prompts
                question = example['question']
                answer_choices = example['answer_choices']
                user_prompt = prepare_user_prompt(question, answer_choices)
                system_prompt = build_system_prompt()

                # Build message in the format expected by Qwen3-VL
                # Combine system prompt with user prompt
                full_text_prompt = f"{system_prompt}\n\n{user_prompt}"

                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": full_text_prompt}
                    ]}
                ]

                # Apply chat template
                input_text = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True
                )

                # Tokenize image + text
                inputs = tokenizer(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(model.device)

                # Start timing
                start_time = time.time()

                # Generate response
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        use_cache=True,
                        temperature=args.temperature,
                        min_p=0.1,
                    )

                # Calculate inference time
                inference_time = time.time() - start_time

                # Decode output
                # Remove input tokens to get only the generated text
                generated_ids = output_ids[0][len(inputs.input_ids[0]):]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

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
                    'model': f"Qwen3-VL-8B-Instruct-{args.quant}",
                    'model_answer': model_answer if model_answer is not None else 'None',
                    'explanation': explanation,
                    'correct': is_correct,
                    'media_type': 'image',
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
        print(f"\nModel: Qwen3-VL-8B-Instruct ({args.quant} quantization)")
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
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Quantized Model Processing (Images Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process dataset with 4-bit quantization (recommended)
  python qwenquant.py --quant 4bit

  # Process dataset with 8-bit quantization (higher quality)
  python qwenquant.py --quant 8bit

  # Process with 16-bit (highest quality, FP16)
  python qwenquant.py --quant 16bit

  # Limit to 100 examples
  python qwenquant.py --quant 4bit --max-examples 100
        """
    )

    # Quantization level selection (REQUIRED)
    parser.add_argument("--quant", type=str, required=True,
                       choices=['4bit', '8bit', '16bit'],
                       help="Quantization level: 4bit, 8bit, or 16bit")

    # Dataset processing parameters
    parser.add_argument("--dataset", type=str, default="JessicaE/OpenSeeSimE-Structural",
                       help="Dataset name on Hugging Face")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to use")
    parser.add_argument("--max-examples", type=int, default=0,
                       help="Maximum number of examples to process (0 = all)")

    # Model inference parameters
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for sampling")

    # Output parameters
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save results CSV (auto-generated if not specified)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to save checkpoint file (auto-generated if not specified)")
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Save checkpoint every N examples")

    args = parser.parse_args()

    # Auto-generate output and checkpoint filenames
    if args.output is None:
        args.output = f"qwen3_vl_8b_{args.quant}_image_results.csv"

    if args.checkpoint is None:
        args.checkpoint = f"qwen3_vl_8b_{args.quant}_image_checkpoint.pkl"

    # Convert max_examples to None if 0
    if args.max_examples == 0:
        args.max_examples = None

    # Print configuration
    print("\n" + "="*80)
    print("QWEN3-VL QUANTIZED MODEL - IMAGE DATASET PROCESSING")
    print("="*80)
    print(f"Model: Qwen3-VL-8B-Instruct (via unsloth)")
    print(f"Quantization: {args.quant}")
    print(f"Dataset: {args.dataset}")
    print(f"Media Type: Images only")
    print(f"Output File: {args.output}")
    print(f"Checkpoint File: {args.checkpoint}")
    print("="*80 + "\n")

    # Process dataset
    process_dataset(args)

if __name__ == "__main__":
    main()
