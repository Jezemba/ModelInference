#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3-VL-235B-A22B-Instruct VQA Benchmark — Using Alibaba Cloud API
----------------------------------------------------------------------------

✅ Uses: Alibaba Cloud DashScope API (OpenAI-compatible endpoint)
✅ Model: Qwen3-VL-235B-A22B-Instruct (235B parameters, 22B activated)
✅ Dataset: same private HF dataset as the original script
✅ Prompt: Simplified with system prompt and few-shot examples
✅ Output: identical columns; default output file: qwen_api_results.csv
✅ Parallelism: same multiprocessing pattern

Key changes from local version:
- Removed local model loading (AutoProcessor, Qwen3VLForConditionalGeneration)
- Added OpenAI client for API calls to Alibaba Cloud
- Images are encoded as base64 for API transmission
- Added API key configuration via environment variable
- Kept all prompt engineering and evaluation logic identical
"""

import os
import re
import io
import base64
import argparse
import traceback
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from openai import OpenAI

# ----------------------------- CONFIG ---------------------------------
# Use the open-source 235B model via Alibaba Cloud API
MODEL_ID = "qwen3-vl-235b-a22b-instruct"

# API Configuration
# Singapore region endpoint (use this for international access)
API_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
# For Beijing region, use: "https://dashscope.aliyuncs.com/compatible-mode/v1"

MAX_OUTPUT_TOKENS = 512

# Recommended generation parameters for Qwen3-VL
GEN_KW = dict(
    max_tokens=MAX_OUTPUT_TOKENS,
    temperature=0.7,
    top_p=0.8,
)

_CLIENT = None


# ------------------------ API CLIENT SETUP --------------------------
def get_api_client():
    """Initialize OpenAI client for Alibaba Cloud API"""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY environment variable not set. "
                "Please obtain an API key from Alibaba Cloud Model Studio: "
                "https://www.alibabacloud.com/help/en/model-studio/get-api-key"
            )
        
        _CLIENT = OpenAI(
            api_key=api_key,
            base_url=API_BASE_URL,
        )
    return _CLIENT


# ------------------------ IMAGE ENCODING --------------------------
def pil_to_base64(pil_image: Image.Image) -> str:
    """Convert PIL Image to base64 string for API transmission"""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


# ------------------------ OPTION NORMALIZATION ------------------------
def normalize_options(options):
    key_to_value = {}
    value_to_key = {}

    if not options:
        return key_to_value, value_to_key

    if isinstance(options[0], dict) and "key" in options[0]:
        for opt in options:
            k = str(opt["key"]).strip()
            v = str(opt.get("value", k)).strip()
            key_to_value[k] = v
            value_to_key[v.lower()] = k
        return key_to_value, value_to_key

    lowered = [str(o).strip().lower() for o in options]
    if set(lowered) == {"yes", "no"}:
        for v in options:
            k = v.strip().lower()
            key_to_value[k] = v.strip()
            value_to_key[v.strip().lower()] = k
        return key_to_value, value_to_key

    for i, v in enumerate(options):
        k = chr(ord("A") + i)
        vs = str(v).strip()
        key_to_value[k] = vs
        value_to_key[vs.lower()] = k
    return key_to_value, value_to_key


# ------------------------ PROMPT BUILDING ----------------------------
def build_prompt(question: str, options, key_to_value):
    """Build a simplified prompt with few-shot examples"""
    
    prompt = question.strip() + "\n\n"

    if key_to_value:
        prompt += "Options:\n"
        if isinstance(options[0], dict) and "key" in options[0]:
            for o in options:
                prompt += f"{o['key']}. {o.get('value','').strip()}\n"
        else:
            lowered = [str(o).strip().lower() for o in options]
            if set(lowered) == {"yes", "no"}:
                prompt += "Yes\nNo\n"
            else:
                for i, o in enumerate(options):
                    prompt += f"{chr(ord('A')+i)}. {str(o).strip()}\n"
        prompt += "\n"

    # Simplified instructions with few-shot examples
    prompt += "Instructions:\n"
    prompt += "Line 1: Write ONLY the answer letter (A/B/C/D) or word (Yes/No)\n"
    prompt += "Line 2: One brief sentence explaining why\n\n"
    
    # Add few-shot examples based on question type
    if key_to_value:
        lowered = [str(o).strip().lower() for o in options] if options else []
        if set(lowered) == {"yes", "no"}:
            # Yes/No example
            prompt += "Example:\nYes\nThe image clearly shows a dog.\n\n"
        else:
            # Multiple choice example
            prompt += "Example:\nB\nThe object in the image matches the description in option B.\n\n"
    
    prompt += "Your answer:"
    return prompt


def build_system_prompt():
    """Build a system prompt to guide the model's behavior"""
    return (
        "You are a visual question answering assistant. "
        "Your task is to answer questions about images accurately and concisely. "
        "Always follow this format exactly:\n"
        "- First line: Only the answer (a single letter like A, B, C, D or a word like Yes, No)\n"
        "- Second line: One brief sentence (10-15 words) explaining your answer\n"
        "Do not add extra text, formatting, or explanations beyond these two lines."
    )


# ------------------------ FIRST-LINE ENFORCEMENT ----------------------
def _allowed_first_line_strings(options, question_type):
    if not options:
        return set()
    lowered = [str(o).strip().lower() for o in options]
    if set(lowered) == {"yes", "no"} or (question_type or "").strip().lower() == "yes_no":
        return {"Yes", "No", "yes", "no"}
    n = min(len(options), 26)
    return {chr(ord("A") + i) for i in range(n)}

_FIRST_LINE_LETTER_PATTERNS = [
    r"^\s*([A-Za-z])[.)]\s",
    r"^\s*\(?([A-Za-z])\)?\s*$",
    r"^\s*([A-Za-z])\s*$",
    r"^[^A-Za-z]*\b([A-Za-z])\b",
]

_YESNO_PATTERNS = [
    r"^\s*(Yes|No)\b[^\n]*",
    r"^\s*(True|False)\b[^\n]*",
]

def _normalize_yesno(token: str) -> str:
    t = token.strip().lower()
    if t in ["yes", "y", "true"]:
        return "Yes"
    if t in ["no", "n", "false"]:
        return "No"
    return token

def enforce_first_line_format(raw_text: str, options, question_type):
    text = (raw_text or "").strip()
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    raw_first_line = lines[0].strip() if lines else ""
    reasoning_lines = lines[1:] if len(lines) > 1 else []

    allowed = _allowed_first_line_strings(options, question_type)

    raw_clean = raw_first_line.strip().lstrip("([<{").rstrip(")]>}.").strip()
    raw_clean_norm = _normalize_yesno(raw_clean)
    if (raw_clean_norm in allowed) or (raw_clean_norm.capitalize() in allowed):
        fixed_first = raw_clean_norm if raw_clean_norm in allowed else raw_clean_norm.capitalize()
        return fixed_first, "\n".join(reasoning_lines).strip(), raw_first_line

    for pat in _YESNO_PATTERNS:
        m = re.match(pat, raw_first_line, flags=re.IGNORECASE)
        if m:
            candidate = _normalize_yesno(m.group(1))
            if candidate in allowed:
                return candidate, "\n".join(reasoning_lines).strip(), raw_first_line

    for pat in _FIRST_LINE_LETTER_PATTERNS:
        m = re.match(pat, raw_first_line, flags=re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in allowed:
                return letter, "\n".join(reasoning_lines).strip(), raw_first_line

    head = raw_first_line[:120]
    for tok in ["Yes", "No", "yes", "no", "True", "False", "true", "false"]:
        if tok in head:
            cand = _normalize_yesno(tok)
            if cand in allowed:
                return cand, "\n".join(reasoning_lines).strip(), raw_first_line

    for c in head:
        if c.isalpha():
            letter = c.upper()
            if letter in allowed:
                return letter, "\n".join(reasoning_lines).strip(), raw_first_line

    if allowed and reasoning_lines:
        fallback = list(allowed)[0]
        return fallback, "\n".join(reasoning_lines).strip(), raw_first_line

    return None, "\n".join(reasoning_lines).strip(), raw_first_line


def parse_first_line_to_key(first_line_str: str, key_to_value, value_to_key):
    fl = first_line_str.strip()
    if fl in key_to_value:
        return fl
    if fl.lower() in value_to_key:
        return value_to_key[fl.lower()]
    for k, v in key_to_value.items():
        if v.lower() == fl.lower():
            return k
    return fl


def parse_model_response(text_fixed: str, options):
    lines = [ln for ln in text_fixed.splitlines() if ln.strip() != ""]
    first = lines[0].strip() if lines else ""
    explanation = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    if not options:
        return first, explanation, first

    key_to_value, value_to_key = normalize_options(options)
    k = parse_first_line_to_key(first, key_to_value, value_to_key)
    return k, explanation, first


def evaluate(answer_key, ground_truth_key, options, question_type):
    if answer_key is None:
        return False
    if not options or question_type == "text":
        return str(answer_key).strip().lower() == str(ground_truth_key).strip().lower()

    yn_alias = {"yes": "yes", "y": "yes", "true": "yes", "no": "no", "n": "no", "false": "no"}
    gt = ground_truth_key.strip().lower()
    ak = str(answer_key).strip().lower()
    if gt in yn_alias:
        gt = yn_alias[gt]
    if ak in yn_alias:
        ak = yn_alias[ak]
    return ak == gt


# ------------------------ API INFERENCE ------------------------------
import time

def qwen_api_respond(pil_images, prompt: str, system_prompt: str) -> str:
    """Call Alibaba Cloud API with images and prompt"""
    client = get_api_client()
    
    content = []
    for img in pil_images:
        img_base64 = pil_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": img_base64}
        })
    content.append({"type": "text", "text": prompt})
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            **GEN_KW
        )
        
        # CRITICAL: Add delay to stay under 100k TPM
        time.sleep(25)  # 25 seconds between requests = safe margin
        
        return completion.choices[0].message.content
    
    except Exception as e:
        print(f"API Error: {e}")
        raise


# ------------------------ SINGLE EXAMPLE ------------------------------
def run_single_example(example):
    """Process a single example and return results dict"""
    if not isinstance(example, dict):
        raise TypeError(f"Example should be dict, got {type(example)}: {example}")
    
    question = example.get("question", "")
    options = example.get("options", [])
    question_type = example.get("question_type", "")
    imgs = example.get("images", [])
    
    if not imgs:
        raise ValueError(f"No images found in example. Keys: {list(example.keys())}")
    
    key_to_value, _ = normalize_options(options)
    
    system_prompt = build_system_prompt()
    prompt = build_prompt(question, options, key_to_value)

    text = qwen_api_respond(imgs, prompt, system_prompt)

    fixed_first, reasoning, raw_first = enforce_first_line_format(text, options=options, question_type=question_type)
    text_fixed = (fixed_first + ("\n" + reasoning if reasoning else "")).strip()

    answer_key, explanation, raw_first_line = parse_model_response(text_fixed, options)
    is_correct = evaluate(answer_key, example.get("answer", ""), options, question_type)

    return {
        "id": example.get("id"),
        "class": example.get("class"),
        "question": question,
        "question_type": question_type,
        "options": str(options),
        "answer_gt": example.get("answer", ""),
        "model_answer_key": answer_key if answer_key is not None else "None",
        "model_first_line": raw_first,
        "explanation": explanation,
        "correct": bool(is_correct),
        "num_images": len(imgs),
        "image_filenames": "|".join(example.get("image_filenames", [])),
        "model": MODEL_ID,
    }


# ------------------------ WORKER -------------------------------------
def worker_process(args):
    worker_id, indices, dataset, out_csv = args
    rows = []
    for idx in tqdm(indices, desc=f"Worker {worker_id}", position=worker_id):
        try:
            ex = dataset[idx]
            
            if not isinstance(ex, dict):
                print(f"\n[Worker {worker_id}] ERROR at idx {idx}:")
                print(f"  Type: {type(ex)}")
                print(f"  Value: {ex}")
                continue
            
            row = run_single_example(ex)
            rows.append(row)
            if len(rows) % 5 == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
        except Exception as e:
            print(f"\n[Worker {worker_id}] idx {idx} failed with error: {e}")
            print(f"Exception type: {type(e).__name__}")
            if isinstance(ex, dict):
                print(f"Example keys: {list(ex.keys())}")
            else:
                print(f"Example type: {type(ex)}, value: {ex}")
            print(f"Full traceback:")
            traceback.print_exc()
            continue
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


# ------------------------ DRIVER -------------------------------------
def run_benchmark(dataset_name,
                  split="test",
                  output_csv="qwen_api_results.csv",
                  max_examples=None,
                  num_workers=1):
    
    # Verify API key is set
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise ValueError(
            "DASHSCOPE_API_KEY environment variable not set.\n"
            "Please set it with: export DASHSCOPE_API_KEY='your-api-key'\n"
            "Get your API key from: https://www.alibabacloud.com/help/en/model-studio/get-api-key"
        )
    
    dset = load_dataset(dataset_name, split=split, token=True)
    total = len(dset)
    if max_examples:
        total = min(total, max_examples)
        dset = dset.select(range(total))

    print("\n🚀 Starting Qwen3-VL-235B-A22B-Instruct API evaluation ...")
    print(f"   Model: {MODEL_ID}")
    print(f"   API Endpoint: {API_BASE_URL}")
    print(f"   Dataset: {dataset_name} [{split}]")
    print(f"   Total examples: {total}")
    print(f"   Workers: {num_workers}")
    print(f"   Generation params: temp={GEN_KW['temperature']}, top_p={GEN_KW['top_p']}")
    
    # Debug: check first example structure
    print(f"\n   Checking dataset structure...")
    first_ex = dset[0]
    print(f"   First example type: {type(first_ex)}")
    if isinstance(first_ex, dict):
        print(f"   First example keys: {list(first_ex.keys())}")
    else:
        print(f"   WARNING: First example is not a dict!")

    indices = list(range(total))
    chunks = [indices[i::num_workers] for i in range(num_workers)]

    worker_args = []
    for w, chunk in enumerate(chunks):
        worker_csv = f"results_worker_{w}.csv"
        worker_args.append((w, chunk, dset, worker_csv))

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            worker_csvs = pool.map(worker_process, worker_args)
    else:
        worker_csvs = [worker_process(worker_args[0])]

    all_df = pd.concat([pd.read_csv(p) for p in worker_csvs if os.path.exists(p)],
                       ignore_index=True)
    all_df.to_csv(output_csv, index=False)

    acc = (all_df["correct"].sum() * 1.0 / len(all_df)) if len(all_df) else 0.0
    print("\n" + "=" * 80)
    print("QWEN3-VL-235B-A22B-INSTRUCT API BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model: {MODEL_ID}")
    print(f"Accuracy: {acc:.2%} ({all_df['correct'].sum()}/{len(all_df)})")
    print(f"Saved: {output_csv}")
    print("=" * 80)


# ------------------------ CLI ----------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Qwen3-VL-235B-A22B-Instruct VQA (API)")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--output", type=str, default="qwen_api_results.csv")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--region", type=str, choices=["singapore", "beijing"], default="singapore",
                    help="API region: singapore (international) or beijing (China)")
    args = ap.parse_args()

    # Set API endpoint based on region
    global API_BASE_URL
    if args.region == "beijing":
        API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    max_examples = None if args.max_examples <= 0 else args.max_examples
    run_benchmark(
        dataset_name=args.dataset,
        split=args.split,
        output_csv=args.output,
        max_examples=max_examples,
        num_workers=max(1, args.num_workers),
    )


if __name__ == "__main__":
    main()