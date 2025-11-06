#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPT-5 VQA Benchmark (Parallel Version, Final)
---------------------------------------------

✅ Uses: OpenAI Responses API (GPT-5-2025-08-07)
✅ Supports: Images + Videos (via S3 presigned URLs)
✅ Output: Per-example CSV + checkpoint (merged after run)
✅ Parallel: --num_workers N (default = 1)
✅ Filtering: --media_type image|video|all
"""

import os
import io
import json
import base64
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import boto3
from botocore.exceptions import ClientError
from multiprocessing import Pool

from openai import OpenAI
import cv2
import numpy as np

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
MODEL_NAME = "gpt-5-2025-08-07"
MAX_OUTPUT_TOKENS = 4096
REASONING_EFFORT = "minimal"
TEXT_VERBOSITY = "medium"

S3_BUCKET = os.environ.get("S3_BUCKET", "gpt5-vqa-temp-aipexws3-1760793625")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def get_s3_client():
    """Each worker gets its own boto3 client."""
    return boto3.client("s3", region_name=AWS_REGION)


def get_presigned_s3_url(file_name, media_type, s3_client, expiry_seconds=3600):
    """Build presigned HTTPS URL for an image/video stored in S3."""
    if media_type == "image":
        key = f"Images/{file_name}"
    elif media_type == "video":
        key = f"Videos/{file_name}"
    else:
        raise ValueError(f"Unknown media_type: {media_type}")

    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expiry_seconds,
        )
        return url
    except ClientError as e:
        print(f"[Worker] Failed to presign {key}: {e}")
        return None


def build_prompt(example):
    """Structured prompt identical to InternVL logic."""
    question = example["question"]
    choices = example.get("answer_choices", [])
    prompt = f"{question}\n\n"
    if choices:
        prompt += "Answer options:\n"
        for c in choices:
            prompt += f"- {c}\n"
        prompt += "\n"
    prompt += (
        "Instructions:\n"
        "1. First line: ONLY the correct answer exactly as in options above.\n"
        "2. Second line: Brief reasoning (1-2 sentences).\n\n"
        "Answer:"
    )
    return prompt


def parse_model_response(text, answer_choices):
    """Extract answer + explanation."""
    lines = text.strip().split("\n")
    if not lines:
        return None, ""
    first = lines[0].strip()
    explanation = "\n".join(lines[1:]).strip()
    if answer_choices:
        for c in answer_choices:
            if first.lower() == c.strip().lower():
                return c.strip(), explanation
        return None, explanation
    return first, explanation


def evaluate(model_answer, ground_truth, answer_choices):
    """Boolean correctness."""
    if model_answer is None:
        return False
    if model_answer.strip().lower() == ground_truth.strip().lower():
        return True
    if answer_choices:
        for c in answer_choices:
            if (
                model_answer.strip().lower() == c.strip().lower()
                and c.strip().lower() == ground_truth.strip().lower()
            ):
                return True
    return False


def openai_respond(user_content):
    """Wrapper for GPT-5 Responses API call."""
    client = OpenAI()
    resp = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": user_content}],
        reasoning={"effort": REASONING_EFFORT},
        text={"verbosity": TEXT_VERBOSITY},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return resp.output_text


# -------------------------------------------------------------------
# SINGLE EXAMPLE INFERENCE
# -------------------------------------------------------------------
def run_single_example(example):
    """Handles image or video examples."""
    s3_client = get_s3_client()
    media_type = example["media_type"]
    prompt = build_prompt(example)

    # ---------------- IMAGE ----------------
    if media_type == "image":
        image_url = get_presigned_s3_url(example["file_name"], "image", s3_client)
        if not image_url:
            raise ValueError(f"Failed to generate URL for {example['file_name']}")
        user_content = [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": image_url},
        ]
        text = openai_respond(user_content)
        ans, expl = parse_model_response(text, example.get("answer_choices"))
        return ans, expl, text

    # ---------------- VIDEO ----------------
    elif media_type == "video":
        # 1️⃣ Download the video from S3
        local_video = f"/tmp/{example['file_name']}"
        s3_client.download_file(S3_BUCKET, f"Videos/{example['file_name']}", local_video)

        # 2️⃣ Extract ~8 frames as PNG base64
        cap = cv2.VideoCapture(local_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"Could not read video: {example['file_name']}")

        num_frames = 8
        frame_indices = np.linspace(0, total - 1, num_frames, dtype=int)

        frames_base64 = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            frames_base64.append(f"data:image/png;base64,{encoded}")

        cap.release()

        if not frames_base64:
            raise ValueError(f"No frames extracted from video {example['file_name']}")

        # 3️⃣ Build GPT-5 multi-image input
        user_content = [
            {"type": "input_text", "text": f"These are frames from a video.\n{prompt}"}
        ]
        for frame_b64 in frames_base64:
            user_content.append({"type": "input_image", "image_url": frame_b64})

        # 4️⃣ Send to GPT-5
        text = openai_respond(user_content)
        ans, expl = parse_model_response(text, example.get("answer_choices"))
        return ans, expl, text

    else:
        raise ValueError(f"Unsupported media_type: {media_type}")


# -------------------------------------------------------------------
# WORKER FUNCTION
# -------------------------------------------------------------------
def worker_process(args):
    """Executed in each process."""
    (worker_id, indices, dataset, out_csv) = args
    worker_results = []
    for idx in tqdm(indices, desc=f"Worker {worker_id}", position=worker_id):
        try:
            ex = dataset[idx]
            model_answer, explanation, full_text = run_single_example(ex)
            is_correct = evaluate(model_answer, ex["answer"], ex.get("answer_choices"))
            res = {
                "idx": idx,
                "file_name": ex.get("file_name"),
                "source_file": ex.get("source_file"),
                "question": ex.get("question"),
                "question_type": ex.get("question_type"),
                "question_id": ex.get("question_id"),
                "answer": ex.get("answer"),
                "answer_choices": str(ex.get("answer_choices")),
                "correct_choice_idx": ex.get("correct_choice_idx"),
                "model": MODEL_NAME,
                "model_answer": model_answer if model_answer else "None",
                "explanation": explanation,
                "correct": is_correct,
                "media_type": ex.get("media_type"),
            }
            worker_results.append(res)
            # Write incremental CSV for this worker
            if len(worker_results) % 5 == 0:
                pd.DataFrame(worker_results).to_csv(out_csv, index=False)
        except Exception as e:
            print(f"[Worker {worker_id}] Example {idx} failed: {e}")
            continue

    pd.DataFrame(worker_results).to_csv(out_csv, index=False)
    return out_csv


# -------------------------------------------------------------------
# MAIN DRIVER
# -------------------------------------------------------------------
def run_benchmark(dataset_name, split="test", output_csv="gpt5_vqa_results.csv",
                  checkpoint="gpt5_vqa_checkpoint.pkl", max_examples=None,
                  num_workers=1, media_type="all"):

    dset = load_dataset(dataset_name, split=split, token=True)

    # Optional filtering by media type
    if media_type != "all":
        print(f"Filtering dataset for {media_type} examples only...")
        keep_idx = [
            i for i in range(len(dset))
            if dset._data.table["media_type"][i].as_py() == media_type
        ]
        dset = dset.select(keep_idx)
        print(f"Remaining examples after filter: {len(dset)}")

    total = len(dset)
    if max_examples:
        total = min(total, max_examples)
        dset = dset.select(range(total))
    print(f"\n🚀 Starting GPT-5 evaluation …")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Total examples: {total}")
    print(f"   Using {num_workers} workers")

    # Split indices
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

    # Merge results
    all_df = pd.concat(
        [pd.read_csv(f) for f in worker_csvs if os.path.exists(f)],
        ignore_index=True
    )
    all_df.to_csv(output_csv, index=False)

    correct = all_df["correct"].sum()
    total = len(all_df)
    acc = correct / total if total else 0

    # Calculate accuracy by question type
    stats_by_type = {}
    accuracy_by_type = {}

    if "question_type" in all_df.columns:
        for q_type in all_df["question_type"].unique():
            if pd.notna(q_type):
                type_df = all_df[all_df["question_type"] == q_type]
                type_correct = type_df["correct"].sum()
                type_total = len(type_df)
                stats_by_type[q_type] = {'correct': type_correct, 'total': type_total}
                accuracy_by_type[q_type] = type_correct / type_total if type_total > 0 else 0

    print("\n" + "=" * 80)
    print("GPT-5 BENCHMARK RESULTS (Merged)")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Accuracy: {acc:.2%} ({correct}/{total})")

    if stats_by_type:
        print(f"\nAccuracy by Question Type:")
        for q_type in sorted(stats_by_type.keys()):
            stats = stats_by_type[q_type]
            type_acc = accuracy_by_type[q_type]
            print(f"  {q_type}: {type_acc:.2%} ({stats['correct']}/{stats['total']})")

    print(f"\nResults saved to: {output_csv}")
    print("=" * 80)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GPT-5 VQA Benchmark (Parallel)")
    parser.add_argument("--dataset", type=str, default="JessicaE/OpenSeeSimE-Structural")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="gpt5_vqa_results.csv")
    parser.add_argument("--checkpoint", type=str, default="gpt5_vqa_checkpoint.pkl")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--media_type",
        type=str,
        choices=["image", "video", "all"],
        default="all",
        help="Filter dataset by media type (image, video, or all)"
    )
    args = parser.parse_args()

    max_examples = None if args.max_examples <= 0 else args.max_examples
    run_benchmark(
        dataset_name=args.dataset,
        split=args.split,
        output_csv=args.output,
        checkpoint=args.checkpoint,
        max_examples=max_examples,
        num_workers=max(1, args.num_workers),
        media_type=args.media_type,
    )


if __name__ == "__main__":
    main()
