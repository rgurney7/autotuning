"""Fine-tune Gemma 4 E2B on train.jsonl via Modal. Run: uv run modal run finetune_modal.py

Structured as an exact extension of debug_modal.py's probe — the only call
sequence that survives gemma4's per-layer-input machinery. SFTTrainer is out:
unsloth's trainer patches (padding-free collation, forced checkpointing)
break this architecture, so training is a bare loop.
"""

import modal

app = modal.App("autotune-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("unsloth")
    .add_local_file("train.jsonl", "/root/train.jsonl")
)

CHAT_SYSTEM_PROMPT = "You are a helpful assistant that can answer questions and help with tasks."


@app.function(image=image, gpu="L40S", timeout=1800, secrets=[modal.Secret.from_dotenv()])
def finetune():
    import json
    import os

    import torch
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        "unsloth/gemma-4-E2B-it",  # same base Ollama serves (gemma4:e2b)
        max_seq_length=2048,
        load_in_4bit=True,
    )

    examples = [json.loads(line)["messages"] for line in open("/root/train.jsonl")]

    def encode(msgs):
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        return tokenizer(text=text, return_tensors="pt").input_ids.to("cuda")

    # loss only on the FINAL assistant turn: examples are prefix-expanded, so
    # earlier turns reappear as context in many examples and would otherwise be
    # weighted by how early they fell in the conversation. Start all-masked
    # (-100), then unmask that last span, located by rendering the conversation
    # without the reply and diffing the token lengths.
    def encode_with_labels(msgs):
        ids = encode(msgs)
        labels = torch.full_like(ids, -100)
        start = encode(msgs[:-1]).shape[1]
        labels[:, start:] = ids[:, start:]
        return ids, labels

    # probe-verified warmup: full-length no_grad forward + a short generate
    # BEFORE attaching adapters. do not reorder — grad-enabled first forwards
    # crash in gemma4's per-layer-input path.
    ids = encode(examples[0])
    with torch.no_grad():
        model(input_ids=ids)
        model.generate(input_ids=ids[:, :16], max_new_tokens=8)
    torch.cuda.empty_cache()  # drop warmup KV cache before training allocations

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        use_gradient_checkpointing=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )
    for epoch in range(10):
        total = 0.0
        for msgs in examples:
            ids, labels = encode_with_labels(msgs)
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(loss)
        print(f"epoch {epoch}: mean loss {total / len(examples):.4f}")

    # smoke test: does it know the user now?
    FastModel.for_inference(model)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": "How old am I and what is my name?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=100)
    print("SMOKE TEST:", tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

    # push merged GGUF to HF (private), then: ollama run hf.co/ShallowLearning/autotune-gemma
    model.push_to_hub_gguf(
        "ShallowLearning/autotune-gemma",
        tokenizer,
        quantization_method="q4_k_m",
        token=os.environ["HF_TOKEN"],
        private=True,
    )


@app.local_entrypoint()
def main():
    finetune.remote()
