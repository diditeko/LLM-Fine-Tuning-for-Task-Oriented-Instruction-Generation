import json
import argparse
import os
from datasets import load_dataset # type: ignore

raw_data = load_dataset("databricks/databricks-dolly-15k", split="train")

def preprocess (save_path= '../dataset/preprocess/dolly_clean.jsonl'):
    data = raw_data

    processed_data=[]
    for item in data:
        instruction=item.get('instruction','').strip()
        response=item.get('response','').strip()

        if instruction and response and len(response.split()) > 5:
            formatted = {
                "instruction": "Generate step-by-step instructions for the user request.",
                "input": instruction,
                "output": response
            }
            processed_data.append(formatted)
    

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f_out:
        for item in processed_data:
            json.dump(item, f_out, ensure_ascii=False)
            f_out.write("\n")

    return processed_data

if __name__ == "__main__":
    preprocess()

