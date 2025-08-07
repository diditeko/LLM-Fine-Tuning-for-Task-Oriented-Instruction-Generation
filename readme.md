# LLM Instruction Generation – Fine-tuning with LoRA



---

## 1. Problem Overview

The goal is to fine-tune a base LLM to generate **step-by-step instructions** based on user prompts (intents). The model is expected to:
- Understand user intent
- Output structured, clear, and actionable instructions
- Generalize across domains and tasks

---

## 2. Dataset Design and Preparation

### Dataset Used
 use the [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset because:
- It contains diverse instructions and responses
- Aligns with task-oriented outputs
- Higher quality and less hallucination than Alpaca
- Berisi instruksi dan respons yang beragam
- The data is annotated by humans, making it higher quality compared to the Alpaca dataset which was generated using GPT-3
- Many samples include procedural steps, such as SOPs or how-to guides, which are well-suited for instruction-tuned model training

### Preprocessing Steps
- Filtering rows with empty `instruction` or `response`
- Reformatting into Alpaca-style JSONL format with:
  - `"instruction"`: Static instruction for the model
  - `"input"`: Actual user intent
  - `"output"`: Ground-truth response

> See `scripts/preprocess_data.py`

---

## 3. Model Choice and Training

### Model Used
We fine-tune `TinyLlama/TinyLlama-1.1B-Chat` using **LoRA (Low-Rank Adaptation)** for efficient parameter-efficient training.
why this model :
-  Lightweight (1.1B parameters): Efficient for fine-tuning and inference on limited resources (e.g., Google Colab, low-VRAM GPUs).
-  Chat-optimized: Trained for instruction-following with structured prompts (e.g., Alpaca/Dolly-style).
-  Open-access: No gating or access restrictions like Mistral or LLaMA models.
-  Fast and low-latency: Quick to load and generate, useful for testing and iteration.

why LoRA :

- because i training using GPU google collab A100 and LoRA use 10GB Vram
- Proven performance: LoRA performs comparably to full fine-tuning on many instruction-following tasks.
- Faster training: Drastically reduces training time and GPU requirements.
- for stabilty LoRA more stable for training and infer than Qlora


### Training Highlights
- PEFT Adapter: LoRA (`r=8`, `alpha=16`, `dropout=0.05`)
- Max sequence length: 512
- Epochs: 10
- Learning rate: 2e-4

> See `src/train.py`

---

## 4. Evaluation Model

### Quantitative Metric
- ROUGE-L: To evaluate content similarity between predicted and ground-truth instructions.

WHY Choose ROUGE:
- Designed for Long and Structured Texts
    ROUGE is well-suited for evaluating outputs like multi-step instructions, summaries, or other long-form text generations.

- Captures Contextual Similarity via N-grams and LCS
    It measures overlap using unigram, bigram, and Longest Common Subsequence (ROUGE-L), allowing it to evaluate both content and sequence similarity.

- Aligned with Task-Oriented Generation Goals
    Since our model outputs structured instructions, ROUGE can assess whether the generated response covers the key points of the ground-truth answer.

- More Suitable Than Accuracy or BLEU
    Accuracy is too simplistic for text generation. BLEU is better for translation. ROUGE offers a better balance for evaluating instruction-style outputs.

- Widely Used in Industry and Academia
    ROUGE is a standard metric in NLP research (e.g., summarization, QA, generative tasks), making our evaluation results easy to benchmark or compare.

###  Baseline Comparison
We compare:
- Base model output (TinyLlama without fine-tuning)
- Fine-tuned output (with LoRA)

> See `src/evaluates.py`

### Results Summary
| Model         | ROUGE-L (Avg) |
|---------------|---------------|
| Baseline      | ~0.854        |
| Fine-tuned    | ~0.886        |

---

* Baseline (0.0854)
- This is the ROUGE-L score of the base model (TinyLlama before fine-tuning).
- The model hasn't been trained on your specific instruction dataset yet, so:
- Its outputs are likely generic, vague, or off-topic.
- It might hallucinate or ignore the instruction format.

* Fine-tuned (0.0886)
- This is the ROUGE-L score after fine-tuning with LoRA using the Dolly dataset.
- The slight increase shows that:
- The model has started learning how to follow the instruction format.
- Its output is a bit more similar to the expected response.

* Improvement (Δ = +0.0032)
- This is a small but positive improvement (~0.32%).
- It suggests that fine-tuning helped, but the effect is limited due to:
- A small model (1.1B TinyLlama)
- Possibly short training time or small dataset
- Limited instruction diversity or quality in the data


## For test 
- For Test model just Run `src/infer.py` and Check Result in log/hasil.josnl

