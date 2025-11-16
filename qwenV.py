import os
import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
import pickle
import shutil
import tempfile
import argparse

# ================================================================
#                MODEL HELPER FUNCTIONS
# ================================================================
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


def parse_model_response(response_text, answer_choices):
    """Parse structured answer from model output."""
    lines = response_text.strip().split("\n")
    if len(lines) == 0:
        return None, ""
    potential_answer = lines[0].strip()
    explanation = "\n".join(lines[1:]).strip()

    if answer_choices and len(answer_choices) > 0:
        for choice in answer_choices:
            if potential_answer.lower() == choice.strip().lower():
                return choice.strip(), explanation
        return None, explanation
    else:
        return potential_answer, explanation


def evaluate_response(model_answer, ground_truth, answer_choices):
    """Exact match comparison."""
    if model_answer is None:
        return False
    if model_answer.strip().lower() == ground_truth.strip().lower():
        return True
    if answer_choices:
        for choice in answer_choices:
            if model_answer.strip().lower() == choice.strip().lower() \
               and choice.strip().lower() == ground_truth.strip().lower():
                return True
    return False


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        print(f"\n📂 Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, "rb") as f:
            data = pickle.load(f)
        return data["processed_indices"], data["results"]
    else:
        return set(), []


def save_checkpoint(checkpoint_file, processed_indices, results):
    data = {"processed_indices": processed_indices, "results": results}
    with open(checkpoint_file, "wb") as f:
        pickle.dump(data, f)


# ================================================================
#                BENCHMARK CORE
# ================================================================
def run_benchmark(
    dataset_name="JessicaE/OpenSeeSimE-Structural",
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    output_file="qwenV_results.csv",
    checkpoint_file="qwenV_checkpoint.pkl",
    max_examples=None,
    start_index=0,
    end_index=None
):
    print(f"🔹 Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test", token=True)

    # Filter for video examples
    video_indices = [i for i, m in enumerate(dataset["media_type"]) if m == "video"]
    dataset = dataset.select(video_indices)
    total_videos = len(dataset)
    print(f"✅ Total videos to process: {total_videos}")

    # Apply slicing for distributed run
    if end_index is None or end_index > total_videos:
        end_index = total_videos

    if start_index < 0 or start_index >= end_index:
        raise ValueError(f"Invalid index range: start={start_index}, end={end_index}")

    dataset = dataset.select(range(start_index, end_index))
    print(f"✅ Processing subset: indices [{start_index}:{end_index}] "
          f"({len(dataset)} examples on this machine)")

    if max_examples and max_examples < len(dataset):
        dataset = dataset.select(range(max_examples))

    # Load model and processor
    print(f"🔹 Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    print("✅ Model loaded successfully.")

    processed_indices, results = load_checkpoint(checkpoint_file)

    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    failed = sum(1 for r in results if r.get("model_answer") == "None")

    temp_dir = tempfile.mkdtemp(prefix="qwenV_")

    try:
        for idx in tqdm(range(len(dataset)), desc="Processing video examples"):
            if idx in processed_indices:
                continue

            example = dataset[idx]
            file_name = example["file_name"]
            question = example["question"]
            answer_choices = example["answer_choices"]
            ground_truth = example["answer"]

            # Build structured prompt with new format
            prompt = f"{question}\n\n"
            if answer_choices:
                prompt += "Answer options:\n"
                for c in answer_choices:
                    prompt += f"- {c}\n"
                prompt += "\n"

            # Add explicit instructions
            prompt += "Instructions:\n"
            prompt += "1. First line: Provide ONLY your answer exactly as it appears in the options above.\n"
            prompt += "2. Second line onwards: Provide a brief summary explaining your reasoning.\n\n"
            prompt += "Answer:"

            try:
                # Get video directly from HuggingFace dataset
                video = example['video']

                # Debug: print video type for first example
                if idx == 0:
                    print(f"🔍 Video object type: {type(video)}")
                    if hasattr(video, '__dict__'):
                        print(f"🔍 Video attributes: {list(vars(video).keys())}")

                # Extract video path from various possible formats
                if isinstance(video, str):
                    # Already a string path (most common case)
                    video_path = video
                elif hasattr(video, 'path'):
                    # VideoDecoder or similar with path attribute
                    video_path = video.path
                elif hasattr(video, 'filename'):
                    # Some video objects use filename
                    video_path = video.filename
                else:
                    # Unknown format - print info and raise error
                    print(f"❌ Unexpected video type: {type(video)}")
                    print(f"   Available attributes: {dir(video)}")
                    raise ValueError(
                        f"Cannot extract video path from type {type(video)}. "
                        "Please check the dataset video format."
                    )

                # === Build messages ===
                # Match the format from qwen_video_model.py (no "type" field for video)
                messages = [
                    {"role": "user", "content": [
                        {"video": video_path,
                         "total_pixels": 20480 * 32 * 32,
                         "min_pixels": 64 * 32 * 32,
                         "max_frames": 2048,
                         "sample_fps": 29},
                        {"type": "text", "text": prompt}
                    ]}
                ]

                # === Preprocess ===
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    [messages],
                    return_video_kwargs=True,
                    image_patch_size=16,
                    return_video_metadata=True
                )

                if video_inputs is not None:
                    video_inputs, video_metadatas = zip(*video_inputs)
                    video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
                else:
                    video_metadatas = None

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    **video_kwargs,
                    do_resize=False,
                    return_tensors="pt"
                ).to(model.device)

                # === Generate ===
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=512)

                generated_ids = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # === Parse & Evaluate ===
                model_answer, explanation = parse_model_response(response, answer_choices)
                is_correct = evaluate_response(model_answer, ground_truth, answer_choices)

                if model_answer is None:
                    failed += 1
                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "file_name": file_name,
                    "source_file": example.get("source_file", ""),
                    "question": question,
                    "question_type": example.get("question_type", ""),
                    "question_id": example.get("question_id", ""),
                    "answer": ground_truth,
                    "answer_choices": str(answer_choices),
                    "correct_choice_idx": example.get("correct_choice_idx", ""),
                    "model": model_name,
                    "model_answer": model_answer if model_answer else "None",
                    "explanation": explanation,
                    "correct": is_correct,
                    "media_type": example["media_type"]
                }

                results.append(result)
                processed_indices.add(idx)

                # Save checkpoint periodically
                if len(processed_indices) % 5 == 0:
                    save_checkpoint(checkpoint_file, processed_indices, results)
                    pd.DataFrame(results).to_csv(output_file, index=False)

            except Exception as e:
                import traceback
                print(f"\n❌ Error processing index {idx}: {type(e).__name__}: {e}")
                traceback.print_exc()
                save_checkpoint(checkpoint_file, processed_indices, results)
                continue

        # Final save
        pd.DataFrame(results).to_csv(output_file, index=False)
        save_checkpoint(checkpoint_file, processed_indices, results)

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(f"Model: {model_name}")
        if total == 0:
            print("⚠️ No successful predictions — check previous errors.")
        else:
            print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
            print(f"Failed answers: {failed}/{total}")
        print(f"Results saved to: {output_file}")
        print("=" * 80)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ================================================================
#                ENTRY POINT
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-VL Video Benchmark Runner (HuggingFace Dataset)")
    parser.add_argument("--dataset_name", type=str, default="JessicaE/OpenSeeSimE-Structural")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output_file", type=str, default="qwenV_results.csv")
    parser.add_argument("--checkpoint_file", type=str, default="qwenV_checkpoint.pkl")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    args = parser.parse_args()

    run_benchmark(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        output_file=args.output_file,
        checkpoint_file=args.checkpoint_file,
        max_examples=args.max_examples,
        start_index=args.start_index,
        end_index=args.end_index
    )
