import argparse
import os
import subprocess
import json

def main():
    """
    Script to launch a vLLM server for Qwen multimodal models with very conservative memory settings.
    """
    parser = argparse.ArgumentParser(description="Launch vLLM server for Qwen multimodal models")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct", 
                        help="Model name or path")
    parser.add_argument("--port", type=int, default=8000, 
                        help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host to run the server on")
    parser.add_argument("--gpu-id", type=int, default=1, 
                        help="Which GPU to use (default: 1, the second GPU)")
    parser.add_argument("--max-model-len", type=int, default=1024, 
                        help="Maximum model context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7, 
                        help="Fraction of GPU memory to use")
    parser.add_argument("--max-num-seqs", type=int, default=8, 
                        help="Maximum number of sequences to process in parallel")
    args = parser.parse_args()
    
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
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", "bfloat16"
    ]
    
    # Print the command being executed
    print("Launching vLLM server with command:")
    print(" ".join(cmd))
    
    # Execute the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main()