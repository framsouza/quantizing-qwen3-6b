from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
import os, gc, math, pathlib
import warnings
import logging
import torch

warnings.filterwarnings("ignore")

logging.getLogger("llmcompressor").setLevel(logging.WARNING)
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if hasattr(torch, "mps"):
    torch.mps.is_available = lambda: False
if hasattr(torch, "accelerator"):
    torch.accelerator.is_available = lambda *a, **k: False

MODEL_DIR = "models/Qwen3-0.6B"
OUTPUT_DIR = "models/Qwen3-0.6B-W4A16"

CALIBRATION_DATASET = "HuggingFaceH4/ultrachat_200k"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQ_LENGTH = 4096

BASE_MODEL_ID = "Qwen/Qwen3-0.6B"

print(f"Base model:      {MODEL_DIR}")
print(f"Quantized model: {OUTPUT_DIR}")

if not os.path.isdir(MODEL_DIR):
    print(f"Downloading {BASE_MODEL_ID} to {MODEL_DIR} ...")
    snapshot_download(repo_id=BASE_MODEL_ID, local_dir=MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

recipe = GPTQModifier(
    scheme="W4A16",
    targets="Linear",
    ignore=["lm_head"],
)

print(f"Recipe: {recipe}")


def build_chat_calibration_dataset(split, num_samples):
    ds = load_dataset(CALIBRATION_DATASET, split=split)
    ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

    def apply_template(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    return ds.map(apply_template, remove_columns=ds.column_names)


if not os.path.isdir(OUTPUT_DIR):
    calibration_dataset = build_chat_calibration_dataset(
        "train_sft", NUM_CALIBRATION_SAMPLES
    )
    oneshot(
        model=MODEL_DIR,
        dataset=calibration_dataset,
        recipe=recipe,
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )
    print(f"Quantization complete. Model saved to: {OUTPUT_DIR}")


def folder_size(path):
    p = pathlib.Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def format_size(nbytes):
    if nbytes < 1024**2:
        return f"{nbytes / 1024:.1f} KB"
    if nbytes < 1024**3:
        return f"{nbytes / 1024**2:.1f} MB"
    return f"{nbytes / 1024**3:.2f} GB"


size_orig = folder_size(MODEL_DIR)
size_q = folder_size(OUTPUT_DIR)
reduction = (1 - size_q / size_orig) * 100 if size_orig > 0 else 0

print("Model Size Comparison")
print("=" * 45)
print(f"Original (BF16):    {format_size(size_orig)}")
print(f"Quantized (W4A16):  {format_size(size_q)}")
print(f"Reduction:          {reduction:.0f}%")

prompt = "What is AI inference?"
chat_messages = [{"role": "user", "content": prompt}]


def build_chat_inputs():
    return tokenizer.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )


base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="cpu",
    dtype=torch.bfloat16,
)

inputs = build_chat_inputs()
outputs = base_model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
generated = outputs[0][inputs["input_ids"].shape[-1] :]

print(f"Base Model ({MODEL_DIR})")
print(f"Prompt: {prompt}")
print(f"Response: {tokenizer.decode(generated, skip_special_tokens=True)}")

quant_model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="cpu",
    dtype=torch.bfloat16,
)

inputs = build_chat_inputs()
outputs = quant_model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
generated = outputs[0][inputs["input_ids"].shape[-1] :]

print(f"Quantized Model ({OUTPUT_DIR})")
print(f"Prompt: {prompt}")
print(f"Response: {tokenizer.decode(generated, skip_special_tokens=True)}")


def calculate_perplexity(model, tokenizer, dataset, max_tokens=5000, stride=512):
    encodings = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = encodings.input_ids
    nlls, prev_end = [], 0

    for begin_loc in range(0, input_ids.size(1), stride):
        end_loc = min(begin_loc + stride, input_ids.size(1))
        trg_len = end_loc - prev_end
        input_slice = input_ids[:, begin_loc:end_loc]
        target_slice = input_slice.clone()
        target_slice[:, :-trg_len] = -100
        with torch.no_grad():
            loss = model(input_slice, labels=target_slice).loss
            nlls.append(loss * trg_len)
        prev_end = end_loc

    return math.exp(torch.stack(nlls).sum() / prev_end)


test_data = build_chat_calibration_dataset("test_sft", 256)
print(f"Loaded {len(test_data)} test samples")

quant_ppl = calculate_perplexity(quant_model, tokenizer, test_data)
print(f"Quantized perplexity: {quant_ppl:.2f}")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="cpu",
    dtype=torch.bfloat16,
)
base_ppl = calculate_perplexity(base_model, tokenizer, test_data)
print(f"Base perplexity: {base_ppl:.2f}")

print("Perplexity Comparison")
print("=" * 40)
print(f"Base (BF16):      {base_ppl:.2f}")
print(f"Quantized (W4A16): {quant_ppl:.2f}")
print(
    f"Difference:       {quant_ppl - base_ppl:+.2f} ({(quant_ppl / base_ppl - 1) * 100:+.1f}%)"
)
print(
    f"\nFYI: A small increase in perplexity is expected the quantized layers use 4-bit weights."
)
