import os
import pandas as pd
from datasets import load_dataset
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from tqdm import tqdm
import json
import pickle
from pathlib import Path

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

def initialize_model(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct"):
    """
    Initialize Qwen3-VL model and processor.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        model, processor
    """
    print(f"Loading model: {model_name}")
    
    # Load model with flash_attention_2 for better performance
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("Model loaded successfully")
    return model, processor

def prepare_messages(example, media_type):
    """
    Prepare message format for Qwen3-VL model with structured output.
    
    Args:
        example: Single example from dataset
        media_type: Either 'image' or 'video'
    
    Returns:
        messages list in proper format
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
    if media_type == 'image':
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example['image']
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    else:  # video
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": example['video']
                    },
                    {"type": "text", "text": prompt}
                ]
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

def run_inference_batch(model, processor, batch_examples):
    """
    Run inference on a batch of examples.
    
    Args:
        model: Qwen3-VL model
        processor: Processor for the model
        batch_examples: List of examples from dataset
    
    Returns:
        List of tuples: [(answer, explanation, full_response), ...]
    """
    torch.cuda.empty_cache()  # Clear GPU cache before batch processing
    batch_messages = []
    batch_answer_choices = []
    
    # Prepare all messages in the batch
    for example in batch_examples:
        media_type = example['media_type']
        messages = prepare_messages(example, media_type)
        batch_messages.append(messages)
        batch_answer_choices.append(example['answer_choices'])
    
    # Process batch
    all_inputs = []
    for messages in batch_messages:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        all_inputs.append(inputs)
    
    # Generate responses for batch
    results = []
    for i, inputs in enumerate(all_inputs):
        try:
            # Move inputs to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512
                )
            
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Parse structured response with validation
            answer, explanation = parse_model_response(output_text, batch_answer_choices[i])
            
            results.append((answer, explanation, output_text))
            
        except Exception as e:
            print(f"\nError processing example in batch: {e}")
            results.append((None, str(e), ""))
    
    return results

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

def determine_batch_size(model, processor):
    """
    Automatically determine optimal batch size based on available GPU memory.
    
    Args:
        model: The model
        processor: The processor
    
    Returns:
        int: Recommended batch size
    """
    # Get available GPU memory
    if torch.cuda.is_available():
        # Get total memory of first GPU
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        
        # Conservative batch size estimation
        # Qwen3-VL-30B-A3B is large, so we use conservative estimates
        if total_memory >= 80:  # A100/H100 80GB
            return 4
        elif total_memory >= 40:  # A100 40GB
            return 2
        else:  # Smaller GPUs
            return 1
    else:
        return 1

def run_benchmark(dataset_name, output_file='results.csv', checkpoint_file='checkpoint.pkl',
                  batch_size=None, model_name="Qwen/Qwen3-VL-30B-A3B-Instruct"):
    """
    Run benchmark evaluation on the entire dataset with batch processing and checkpointing.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        output_file: CSV file to save results
        checkpoint_file: Pickle file to save checkpoints
        batch_size: Batch size for processing (None for auto-detect)
        model_name: Name of the model to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load dataset
    dataset = load_benchmark_dataset(dataset_name)
    
    # Initialize model
    model, processor = initialize_model(model_name)
    
    # Determine batch size if not specified
    if batch_size is None:
        batch_size = determine_batch_size(model, processor)
        print(f"\n🔧 Auto-detected batch size: {batch_size}")
    else:
        print(f"\n🔧 Using specified batch size: {batch_size}")
    
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
    
    # Process remaining examples in batches
    print(f"\n🚀 Starting batch processing...")
    print(f"   Total examples: {len(dataset)}")
    print(f"   Already processed: {len(processed_indices)}")
    print(f"   Remaining: {len(dataset) - len(processed_indices)}")
    
    # Get unprocessed indices
    unprocessed_indices = [i for i in range(len(dataset)) if i not in processed_indices]
    
    # Process in batches
    for batch_start in tqdm(range(0, len(unprocessed_indices), batch_size), desc="Processing batches"):
        batch_indices = unprocessed_indices[batch_start:batch_start + batch_size]
        batch_examples = [dataset[idx] for idx in batch_indices]
        
        try:
            # Run inference on batch
            batch_results = run_inference_batch(model, processor, batch_examples)
            
            # Process results
            for idx, (model_answer, explanation, full_response) in zip(batch_indices, batch_results):
                example = dataset[idx]
                
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
            
            # Save checkpoint after each batch
            save_checkpoint(checkpoint_file, processed_indices, results)
            
            # Save intermediate CSV
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
        
        except Exception as e:
            print(f"\n❌ Error processing batch starting at index {batch_start}: {e}")
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
    OUTPUT_FILE = "qwen3_vl_benchmark_results.csv"
    CHECKPOINT_FILE = "qwen3_vl_checkpoint.pkl"
    MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    BATCH_SIZE = 1  # Auto-detect based on GPU memory (or set manually, e.g., 2)
    
    # Run benchmark
    summary = run_benchmark(
        DATASET_NAME, 
        OUTPUT_FILE, 
        checkpoint_file=CHECKPOINT_FILE,
        batch_size=BATCH_SIZE,
        model_name=MODEL_NAME
    )