"""Isolate the gemma4 forward-pass crash: base model vs 4bit vs adapters. Run: uv run modal run debug_modal.py"""

import modal

app = modal.App("autotune-debug")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("unsloth")
    .add_local_file("train.jsonl", "/root/train.jsonl")
)


@app.function(image=image, gpu="L4", timeout=1200)
def probe():
    import json

    import torch
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        "unsloth/gemma-4-E2B-it", max_seq_length=2048, load_in_4bit=True
    )

    cfg = getattr(model.config, "text_config", model.config)
    print("CONFIG num_hidden_layers:", getattr(cfg, "num_hidden_layers", "?"))
    for k in dir(cfg):
        if "per_layer" in k or "altup" in k:
            print("CONFIG", k, "=", getattr(cfg, k))

    msgs = json.loads(open("/root/train.jsonl").readline())["messages"]
    text = tokenizer.apply_chat_template(msgs, tokenize=False)
    ids = tokenizer(text=text, return_tensors="pt").input_ids.to("cuda")
    print("SEQ LEN:", ids.shape)

    try:
        with torch.no_grad():
            out = model(input_ids=ids)
        print("BASE FORWARD OK:", out.logits.shape)
    except Exception as e:
        print("BASE FORWARD FAILED:", type(e).__name__, e)

    try:
        with torch.no_grad():
            gen = model.generate(input_ids=ids[:, :16], max_new_tokens=8)
        print("GENERATE OK:", tokenizer.decode(gen[0][-8:]))
    except Exception as e:
        print("GENERATE FAILED:", type(e).__name__, e)

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
    # training-style forward on EVERY example, mask and no-mask — is the crash length-dependent?
    for i, line in enumerate(open("/root/train.jsonl")):
        msgs = json.loads(line)["messages"]
        ids = tokenizer(
            text=tokenizer.apply_chat_template(msgs, tokenize=False), return_tensors="pt"
        ).input_ids.to("cuda")
        for label, kwargs in [
            ("no-mask", {}),
            ("with-mask", {"attention_mask": torch.ones_like(ids)}),
        ]:
            try:
                out = model(input_ids=ids, labels=ids.clone(), **kwargs)
                print(f"EX{i} len={ids.shape[1]} {label}: OK loss={float(out.loss):.3f}")
            except Exception as e:
                print(f"EX{i} len={ids.shape[1]} {label}: FAILED {type(e).__name__} {e}")


@app.local_entrypoint()
def main():
    probe.remote()
