"""Direct MiniCPM BF16 tool-call probe — transformers ONLY, no project code.

Loads openbmb/MiniCPM5-1B with stock HuggingFace eager (BF16), renders a
few tools through the model's own chat template, generates, prints the
whole raw output. Nothing from src/ is imported.

Run (stop llm_chat.py / server first — same 4GB GPU):
  .venv/bin/python -u test_toolcall.py
  .venv/bin/python -u test_toolcall.py --prompt "list the files"
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOLS = [
    {"name": "list",
     "description": "List directory entries below the working directory.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}}}},
    {"name": "spawn_terminal",
     "description": "Spawn a new OS terminal window. Use for opencode/htop/interactive TUIs.",
     "parameters": {"type": "object",
                    "properties": {"command": {"type": "string"},
                                   "cwd": {"type": "string"},
                                   "title": {"type": "string"}}}},
    {"name": "exec",
     "description": "Run a bash shell command.",
     "parameters": {"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]}},
]


def main() -> None:
    ap = argparse.ArgumentParser(description="direct MiniCPM BF16 probe")
    ap.add_argument("--model", default="openbmb/MiniCPM5-1B")
    ap.add_argument("--prompt", default="open a terminal running opencode")
    ap.add_argument("--max-tokens", type=int, default=320)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    try:
        import accelerate  # noqa: F401
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=False, low_cpu_mem_usage=True)
    except Exception:
        # No accelerate: BF16 on CUDA (~2.1GB), fp32 on CPU.
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=False,
            low_cpu_mem_usage=True)
        if torch.cuda.is_available():
            model = model.to("cuda")
    model.eval()
    dev = next(model.parameters()).device
    print(f"model on {dev}", flush=True)

    msgs = [{"role": "system", "content": "You control a terminal."},
            {"role": "user", "content": args.prompt}]
    ids = tok.apply_chat_template(
        msgs, tools=TOOLS, add_generation_prompt=True,
        enable_thinking=True, return_tensors="pt")["input_ids"].to(dev)
    print(f"prompt tokens: {ids.shape[1]}", flush=True)

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.max_tokens,
                             do_sample=False, use_cache=True)
    new_ids = out[0][ids.shape[1]:].tolist()
    print("=== WHOLE RAW OUTPUT ===", flush=True)
    print(tok.decode(new_ids, skip_special_tokens=True))
    print("=== END RAW ===", flush=True)


if __name__ == "__main__":
    main()
