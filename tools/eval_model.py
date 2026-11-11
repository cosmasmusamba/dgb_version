#!/usr/bin/env python3
r"""
tools/eval_model.py (batch3 / roadmap T0.4) - one repeatable report for a DGB checkpoint.

    python tools\eval_model.py                                   # latest checkpoint, cleaned data from config
    python tools\eval_model.py --checkpoint checkpoints\dgb1\models\best_model.pt
    python tools\eval_model.py --data datasets\dgb1\cleaned\wk_9.txt --max-lines 3000

Reports
  * loss per token and perplexity on the evaluation text
  * bits per byte (BPB): total negative log-likelihood divided by the UTF-8 bytes of the text.
    Unlike perplexity it does NOT depend on the tokenizer, so numbers stay comparable when the
    tokenizer changes. Lower is better. (Approximate: BOS is not predicted, EOS is.)
  * greedy continuations of a fixed prompt set (read them: numbers alone do not show broken text)
  * the same numbers for a "uniform guess" baseline, so you can see how much the model learned

The report is also saved as JSON in eval_reports\ so runs can be compared over time.
IMPORTANT: evaluation text must be data the model was NOT trained on, or the numbers mean nothing.
"""
import argparse
import datetime
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROMPTS = [
    "The capital of Uganda is",
    "In Python, a list comprehension is",
    "Mobile money allows people to",
    "Once upon a time,",
    "def add(a, b):",
]


def natural_key(p: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]


def read_lines(path: Path, max_lines: int):
    files = [path] if path.is_file() else sorted(path.glob("*.txt"), key=natural_key)
    if not files:
        raise SystemExit(f"ERROR: no text found at {path}")
    lines = []
    for f in reversed(files):                      # newest/last file first, keep the tail
        with f.open("r", encoding="utf-8", errors="replace") as fh:
            chunk = [l.strip() for l in fh if l.strip()]
        lines = chunk[-(max_lines - len(lines)):] + lines
        if len(lines) >= max_lines:
            break
    return lines[-max_lines:]


def main() -> int:
    import torch
    import torch.nn.functional as F
    from configs.loader import get_config
    from tokenizer.dgb_tokenizer import DGBTokenizer
    from transformer.utils.model_helpers import load_model, latest_checkpoint, resolve_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--data", help="file or folder of .txt (default: paths.cleaned_dir)")
    ap.add_argument("--max-lines", type=int, default=2000)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report-dir", default="eval_reports")
    a = ap.parse_args()

    cfg = get_config()
    device = resolve_device(a.device)
    tok = DGBTokenizer.from_pretrained(cfg.tokenizer_dir())
    ckpt = Path(a.checkpoint) if a.checkpoint else None
    if ckpt is None:
        mdir = cfg.models_dir()
        best = mdir / "best_model.pt"
        ckpt = best if best.exists() else latest_checkpoint(mdir)
    if ckpt is None or not Path(ckpt).exists():
        print("ERROR: no checkpoint found. Pass --checkpoint <file.pt>")
        return 2
    model = load_model(str(ckpt), device, vocab_size=tok.vocab_size).to(device).eval()
    arch = getattr(model, "architecture", "encoder_decoder")
    if arch != "decoder_only":
        print("ERROR: this harness evaluates decoder_only models only (legacy model leaks labels).")
        return 2

    data = Path(a.data) if a.data else cfg.cleaned_dir()
    lines = read_lines(data, a.max_lines)
    print(f"checkpoint : {ckpt}")
    print(f"tokenizer  : {type(tok).__name__}  vocab_size={tok.vocab_size}")
    print(f"eval text  : {data}  ({len(lines)} lines)")

    nll = n_tok = n_bytes = 0
    pad = cfg.transformer.pad_idx
    with torch.no_grad():
        for line in lines:
            ids = tok.encode(line, add_special_tokens=True, truncation=True,
                             max_length=cfg.transformer.max_seq_len)
            if len(ids) < 3:
                continue
            x = torch.tensor([ids[:-1]], device=device)
            y = torch.tensor([ids[1:]], device=device)
            logits = model(x).float()
            nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                   ignore_index=pad, reduction="sum").item()
            n_tok += int((y != pad).sum())
            n_bytes += len(line.encode("utf-8"))
    if n_tok == 0:
        print("ERROR: no usable evaluation lines")
        return 2
    loss = nll / n_tok
    uniform = math.log(tok.vocab_size)
    res = {
        "checkpoint": str(ckpt), "tokenizer": type(tok).__name__, "vocab_size": tok.vocab_size,
        "eval_lines": len(lines), "eval_tokens": n_tok, "eval_bytes": n_bytes,
        "loss_per_token": round(loss, 4), "perplexity": round(math.exp(min(loss, 50)), 2),
        "bits_per_byte": round(nll / math.log(2) / n_bytes, 4),
        "uniform_guess_loss": round(uniform, 4),
        "uniform_guess_bits_per_byte": round(n_tok * uniform / math.log(2) / n_bytes, 4),
    }
    print(f"\nloss/token {res['loss_per_token']}   perplexity {res['perplexity']}   "
          f"bits/byte {res['bits_per_byte']}")
    print(f"(uniform guess would be loss {res['uniform_guess_loss']}, bits/byte "
          f"{res['uniform_guess_bits_per_byte']})")

    samples = []
    print("\n-- greedy continuations --")
    for p in PROMPTS:
        ids = tok.encode(p, add_special_tokens=True)
        if len(ids) > 1 and ids[-1] == getattr(tok, "eos_id", 3):
            ids = ids[:-1]
        out = model.generate(torch.tensor([ids], device=device), max_new_tokens=a.max_new_tokens,
                             eos_id=3, temperature=0.0)[0].tolist()
        text = tok.decode(out, skip_special_tokens=True)
        samples.append({"prompt": p, "continuation": text, "new_tokens": len(out)})
        note = "   (model ended immediately with <EOS>; normal for an undertrained model)" if not text else ""
        print(f"  {p!r} -> {text!r}{note}")
    res["samples"] = samples

    rdir = Path(a.report_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    out_path = rdir / f"eval_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
