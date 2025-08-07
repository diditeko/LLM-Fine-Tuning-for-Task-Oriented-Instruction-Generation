import os
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType # type: ignore
from datasets import load_dataset, Dataset # type: ignore
import json

Model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_json(path):
    with open(path,'r',encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    return Dataset.from_list(lines)

#  Format instruction-style prompt
def formatting(example):
    prompt = f"### Instruction:\\n{example['instruction']}\\n\\n### Input:\\n{example['input']}\\n\\n### Response:\\n{example['output']}"
    return { "text": prompt}

# Tokenization function
def tokenize(example, tokenizer):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    model_name = Model
    data_path = "../dataset/preprocess/dolly_clean.jsonl"
    output_dir = "../model/"

    dataset = load_json(data_path)
    dataset = dataset.map(formatting)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token 

    # Tokenize
    tokenized_dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    # Load base model 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA config (standard)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        learning_rate=2e-4,
        logging_dir="logs",
        output_dir=output_dir,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=1,
        fp16=True,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Training finished. Adapter saved to: {output_dir}")

    #log
    log_path = "../logs/training_log.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Training Log - Fine-tuning TinyLlama with LoRA\\n\\n")
        f.write("Base Model:\\n")
        f.write(f"{model_name}\\n\\n")

        f.write("Adapter Type:\\n")
        f.write(f"LoRA (r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout})\\n\\n")

        f.write("Training Configuration:\\n")
        f.write(f"Epochs: {training_args.num_train_epochs}\\n")
        f.write(f"Batch Size: {training_args.per_device_train_batch_size}\\n")
        f.write(f"Learning Rate: {training_args.learning_rate}\\n")

        f.write("Training Steps:\\n")
        f.write(f"Total steps: {trainer.state.max_steps}\\n")
        f.write(f"Final training loss: {trainer.state.log_history[-1].get('loss', 'N/A')}\\n\\n")

        f.write("Dataset:\\n")
        f.write(f"Preprocessed entries: {len(data_path)}\\n")
        f.write("Tokenizer: AutoTokenizer (pad_token = eos_token)\\n\\n")

        f.write("Output:\\n")
        f.write(f"Adapter saved in: {output_dir}\\n")

if __name__ == "__main__":
    main()


