import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel # type: ignore
from evaluate import load as load_metric # type: ignore
import os

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "../model"
dataset_path="../dataset/preprocess/dolly_clean.jsonl"

def load_finetuned_model(base_model_name, adapter_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer

def load_baseline_model(base_model_name):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text="", max_new_tokens=200):
    prompt = f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{input_text}\\n\\n### Response:\\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate(dataset_path=dataset_path, base_model_name=base_model_name, adapter_path=adapter_path, output_csv="evaluation_results.csv", sample_size=50):
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    eval_data = raw_data[:sample_size]

    # Load models
    print("Loading models...")
    finetuned_model, finetuned_tokenizer = load_finetuned_model(base_model_name, adapter_path)
    baseline_model, baseline_tokenizer = load_baseline_model(base_model_name)

    rouge = load_metric("rouge")
    results = []

    for i, item in enumerate(eval_data):
        instruction = item["instruction"]
        input_text = item["input"]
        expected_output = item["output"]

        print(f"Evaluating sample {i+1}/{sample_size}...")

        base_output = generate_response(baseline_model, baseline_tokenizer, instruction, input_text)
        finetuned_output = generate_response(finetuned_model, finetuned_tokenizer, instruction, input_text)

        # ROUGE for both
        rouge_base = load_metric("rouge")
        rouge_finetuned = load_metric("rouge")
        rouge_base.add(prediction=base_output, reference=expected_output)
        rouge_finetuned.add(prediction=finetuned_output, reference=expected_output)

        score_base = rouge_base.compute()["rougeL"]
        score_finetuned = rouge_finetuned.compute()["rougeL"]

        results.append({
            "instruction": instruction,
            "input": input_text,
            "expected_output": expected_output,
            "baseline_output": base_output,
            "finetuned_output": finetuned_output,
            "rougeL_baseline": round(score_base, 4),
            "rougeL_finetuned": round(score_finetuned, 4)
        })

    df = pd.DataFrame(results)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Print average improvement
    avg_base = df["rougeL_baseline"].mean()
    avg_finetuned = df["rougeL_finetuned"].mean()
    print("\\n ROUGE-L Summary:")
    print(f"Baseline Avg: {avg_base:.4f}")
    print(f"Finetuned Avg: {avg_finetuned:.4f}")
    print(f"Improvement  : {avg_finetuned - avg_base:.4f}")

    print(f" Evaluation complete. Results saved to: {output_csv}")

if __name__ == "__main__":
    evaluate(
        output_csv="evaluation_results.csv",
        sample_size=3
    )