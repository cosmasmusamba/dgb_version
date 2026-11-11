"""Tests for the byte-level BPE tokenizer (batch3 / T1.1)."""
import random
from collections import Counter

import pytest

from tokenizer.byte_bpe import (BOS_ID, EOS_ID, N_SPECIAL, PAD_ID, UNK_ID, ByteBPETokenizer,
                                pre_split)

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Mobile money lets people send money to each other in Kampala.",
    "def add(a, b):\n    return a + b\n",
    "It's a function; it doesn't return anything useful, but we're fine.",
    "Webale nnyo! Oli otya? Nze nsanyuse okukulaba.",
] * 40

HARD_TEXTS = [
    "", " ", "  leading and trailing  ", "tabs\tand\nnewlines\r\nmixed\n\n\n",
    "def f(x):\n    if x:\n        return 1\n    return 0\n",
    "emoji 😀🎉 and accents café naïve Zürich", "中文 日本語 한국어 العربية हिन्दी",
    "Luganda: Nze mbadde nkwagala nnyo, Omwana wange.", "under_score __init__ snake_case",
    "numbers 1234567890 and 3.14159", "'quoted' \"double\" <tags> {json: [1,2]}",
    "a" * 500, "\u0000\u0001 control chars \x7f", "mixed\u00a0nbsp and \u200bzero-width",
]


@pytest.fixture(scope="module")
def tok():
    return ByteBPETokenizer.train(CORPUS, vocab_size=400, min_freq=2)


def test_special_ids_match_legacy_layout(tok):
    assert (PAD_ID, UNK_ID, BOS_ID, EOS_ID) == (0, 1, 2, 3)
    assert tok.vocab_size == N_SPECIAL + 256 + len(tok.merges)
    assert tok.vocab_size <= 400


@pytest.mark.parametrize("text", HARD_TEXTS)
def test_exact_round_trip(tok, text):
    ids = tok.encode(text, add_special_tokens=False, truncation=False)
    assert tok.decode(ids) == text


def test_round_trip_random_unicode_never_unk(tok):
    rnd = random.Random(1234)
    for _ in range(300):
        s = "".join(chr(rnd.choice([rnd.randint(32, 126), rnd.randint(0xA0, 0x2FF),
                                    rnd.randint(0x4E00, 0x4E80), rnd.randint(0x1F300, 0x1F3FF), 10, 32]))
                    for _ in range(rnd.randint(0, 60)))
        ids = tok.encode(s, add_special_tokens=False, truncation=False)
        assert UNK_ID not in ids
        assert tok.decode(ids) == s


def test_pre_split_loses_nothing():
    rnd = random.Random(7)
    pool = "abc XYZ 019 _-.,;'\"\n\t\r é中😀"
    for _ in range(500):
        s = "".join(rnd.choice(pool) for _ in range(rnd.randint(0, 40)))
        assert "".join(pre_split(s)) == s


def test_spaces_survive_decode(tok):
    text = "The quick brown fox jumps over the lazy dog."
    out = tok.decode(tok.encode(text))
    assert out == text and " " in out          # the legacy tokenizer returned 'Thequickbrownfox...'


def test_training_learns_merges_and_compresses(tok):
    text = "Mobile money lets people send money to each other in Kampala."
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) < len(text.encode("utf-8")) * 0.6


def test_training_is_deterministic():
    a = ByteBPETokenizer.train(CORPUS, vocab_size=350)
    b = ByteBPETokenizer.train(list(reversed(CORPUS)), vocab_size=350)
    assert a.merges == b.merges


def test_train_from_word_freq_matches_train():
    wf = Counter()
    for t in CORPUS:
        wf.update(pre_split(t))
    assert ByteBPETokenizer.train_from_word_freq(wf, 350).merges == ByteBPETokenizer.train(CORPUS, 350).merges


def test_vocab_too_small_is_rejected():
    with pytest.raises(ValueError):
        ByteBPETokenizer.train(CORPUS, vocab_size=100)


def test_special_tokens_truncation_padding_batch(tok):
    r = tok.encode_plus("hello world", add_special_tokens=True)
    assert r.input_ids[0] == BOS_ID and r.input_ids[-1] == EOS_ID
    assert tok.decode(r.input_ids) == "hello world"
    assert "<BOS>" in tok.decode(r.input_ids, skip_special_tokens=False)
    t = tok.encode("word " * 200, max_length=16)
    assert len(t) == 16 and t[-1] == EOS_ID
    p = tok.encode_plus("hi", max_length=12, padding=True)
    assert len(p.input_ids) == 12 and p.input_ids[-1] == PAD_ID and sum(p.attention_mask) < 12
    b = tok.encode_batch(["a", "b c"], max_length=8, padding=True)
    assert [len(x) for x in b.input_ids] == [8, 8]


def test_save_load_round_trip_and_facade_dispatch(tok, tmp_path):
    tok.save(tmp_path)
    again = ByteBPETokenizer.from_pretrained(tmp_path, max_seq_len=64)
    text = "Round trip: café 😀 def f():\n    pass"
    assert again.encode(text) == tok.encode(text)
    assert again.vocab_size == tok.vocab_size and again.max_seq_len == 64
    from tokenizer.dgb_tokenizer import DGBTokenizer
    via_facade = DGBTokenizer.from_pretrained(tmp_path)
    assert isinstance(via_facade, ByteBPETokenizer)
    assert via_facade.decode(via_facade.encode(text)) == text


def test_corrupt_file_is_rejected(tmp_path):
    (tmp_path / "byte_bpe.json").write_text('{"format":"other"}', encoding="utf-8")
    with pytest.raises(ValueError):
        ByteBPETokenizer.from_pretrained(tmp_path)


def test_works_with_streaming_dataset_and_model(tok, tmp_path):
    torch = pytest.importorskip("torch")
    from trainer.core.dataset_loader import StreamingTextDataset, build_streaming_loader
    from transformer.core.gpt_model import DGBGPT
    d = tmp_path / "c"
    d.mkdir()
    (d / "a.txt").write_text("\n".join(CORPUS[:12]) + "\n", encoding="utf-8")
    ds = StreamingTextDataset(tok, d, max_seq_len=64)
    src, tgt = next(iter(build_streaming_loader(ds, batch_size=4)))
    m = DGBGPT(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=1, d_ff=64, dropout=0.0,
               max_seq_len=64)
    assert m(src).shape == (src.size(0), src.size(1), tok.vocab_size)
