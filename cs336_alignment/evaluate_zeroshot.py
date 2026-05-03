import json
import argparse
from typing import Callable, List, Dict, Any
import pandas as pd
from vllm import LLM, SamplingParams
from pathlib import Path
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]
        prompt = prompts[i]
        
        reward_dict = reward_fn(generated_text, ground_truth)
        
        result = {
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "reward": reward_dict.get("reward", 0.0),
            "format_reward": reward_dict.get("format_reward", 0.0),
            "answer_reward": reward_dict.get("answer_reward", 0.0),
        }
        results.append(result)
        
    print(f"Serializing results to {output_path}...")
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    # Calculate aggregate metrics
    avg_reward = sum(r["reward"] for r in results) / len(results) if results else 0
    avg_format_reward = sum(r["format_reward"] for r in results) / len(results) if results else 0
    avg_answer_reward = sum(r["answer_reward"] for r in results) / len(results) if results else 0
    
    print(f"Evaluation Complete.")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Average Format Reward: {avg_format_reward:.4f}")
    print(f"Average Answer Reward: {avg_answer_reward:.4f}")

    # Category counts
    # (1) correct with both format and answer reward 1
    # (2) format reward 1 and answer reward 0
    # (3) format reward 0 and answer reward 0
    cat_1 = sum(1 for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0)
    cat_2 = sum(1 for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 0.0)
    cat_3 = sum(1 for r in results if r["format_reward"] == 0.0 and r["answer_reward"] == 0.0)
    
    print(f"Counts per category:")
    print(f"Category 1 (Format 1, Answer 1): {cat_1}")
    print(f"Category 2 (Format 1, Answer 0): {cat_2}")
    print(f"Category 3 (Format 0, Answer 0): {cat_3}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../../Qwen/Qwen2.5-0.5B/")
    parser.add_argument("--output_path", type=str, default="zeroshot_qwen_math_eval.jsonl")
    args = parser.parse_args()
    
    # 1) load the MATH validation examples
    print("Loading data...")
    df = pd.read_parquet("hf://datasets/qwedsacf/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet")
    
    # 2) format them as string prompts using the r1_zero prompt
    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()
    
    prompts = [prompt_template.replace("{question}", problem) for problem in df["problem"]]
    ground_truths = list(df["solution"])
    
    # Initialize vLLM
    print(f"Loading model {args.model_path}...")
    # Load vllm Model
    model = LLM(
        model=args.model_path, 
        trust_remote_code=True, 
        tensor_parallel_size=1,
        max_num_seqs=64, # Adjust this to tune concurrency
        max_model_len=4096 # Adjust based on needed context window
    )    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop_token_ids=[],
    )
    
    # Evaluate
    evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
