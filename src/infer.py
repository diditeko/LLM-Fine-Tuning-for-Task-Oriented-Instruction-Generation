import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # type: ignore

# Model dan adapter path
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "../model"

# Instruction tetap
instruction = (
    "Generate clear and concise step-by-step instructions in numbered list format "
    "based on the user request. Only include essential steps. Avoid redundant information. "
    "Use the format: 1. ..., 2. ..., 3. ..., and so on."
)

def load_model(base_model_name, adapter_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text="", max_new_tokens=150):
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
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

if __name__ == "__main__":
    input_text = input("Masukkan intent yang ingin diberikan: ")

    model, tokenizer = load_model(base_model_name, adapter_path)
    response = generate_response(model, tokenizer, instruction, input_text)

    print("\nOutput Model:\n", response)

    # Simpan hasil ke file .jsonl
    os.makedirs("log", exist_ok=True)
    result = {
        "instruction": instruction,
        "input": input_text,
        "response": response
    }

    with open("../log/hasil.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Hasil berhasil disimpan ke log/hasil.jsonl")
