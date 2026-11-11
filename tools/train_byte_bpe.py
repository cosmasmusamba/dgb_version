#!/usr/bin/env python3
r"""
tools/train_byte_bpe.py (batch3) - train the byte-level BPE tokenizer on your cleaned text files.

    python tools\train_byte_bpe.py                      # corpus + output dir from runtime_config.json
    python tools\train_byte_bpe.py --corpus datasets\dgb1\cleaned --out checkpoints\dgb1\tokenizer
    python tools\train_byte_bpe.py --vocab-size 8000 --max-lines 200000

Writes <out>\byte_bpe.json. The old vocabulary.json / bpe_merges.json in that folder are left
untouched; DGBTokenizer.from_pretrained() prefers byte_bpe.json when it exists. To go back to the
old tokenizer, rename or delete byte_bpe.json.
"""
import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.byte_bpe import ByteBPETokenizer, pre_split  # noqa: E402


def natural_key(p: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", help="folder of .txt files (default: paths.cleaned_dir from config)")
    ap.add_argument("--out", help="output folder (default: paths.tokenizer_dir from config)")
    ap.add_argument("--vocab-size", type=int, help="default: tokenizer.vocab_size from config")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-lines", type=int, default=0, help="stop after N lines (0 = all)")
    a = ap.parse_args()

    from configs.loader import get_config
    cfg = get_config()
    corpus = Path(a.corpus) if a.corpus else cfg.cleaned_dir()
    out = Path(a.out) if a.out else cfg.tokenizer_dir()
    vocab_size = a.vocab_size or cfg.tokenizer.vocab_size
    files = sorted(corpus.glob("*.txt"), key=natural_key)
    if not files:
        print(f"ERROR: no .txt files in {corpus}")
        return 2
    print(f"corpus : {corpus}  ({len(files)} files)")
    print(f"output : {out}")
    print(f"vocab  : {vocab_size}")

    t0, n_lines, n_bytes, freq = time.time(), 0, 0, Counter()
    for f in files:
        with f.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                freq.update(pre_split(line))
                n_lines += 1
                n_bytes += len(line.encode("utf-8"))
                if a.max_lines and n_lines >= a.max_lines:
                    break
        if a.max_lines and n_lines >= a.max_lines:
            break
    print(f"read   : {n_lines} lines, {n_bytes/1e6:.1f} MB, {len(freq)} unique words  ({time.time()-t0:.0f}s)")

    t1 = time.time()
    tok = ByteBPETokenizer.train_from_word_freq(freq, vocab_size=vocab_size, min_freq=a.min_freq)
    print(f"trained: vocab_size={tok.vocab_size}  merges={len(tok.merges)}  ({time.time()-t1:.0f}s)")
    tok.save(out)

    # Report: round trip + compression on a sample of the corpus.
    sample, total_tok, total_b, bad = [], 0, 0, 0
    with files[0].open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                sample.append(line)
            if len(sample) >= 2000:
                break
    for s in sample:
        ids = tok.encode(s, add_special_tokens=False, truncation=False)
        total_tok += len(ids)
        total_b += len(s.encode("utf-8"))
        bad += tok.decode(ids) != s
    if sample:
        print(f"check  : {len(sample)} lines, exact round trip failures = {bad}, "
              f"{total_b/max(total_tok,1):.2f} bytes per token")
    print("example:", tok.tokenize(sample[0])[:12] if sample else "-")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
