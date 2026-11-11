"""
tests/unit/test_gpt_model.py   (batch2 / roadmap tasks T0.2 + T0.3)

Checks the decoder-only model DGBGPT: no label leakage, real learning, checkpoint safety,
the architecture-aware trainer helper and the inference engine path.
The legacy encoder-decoder leak stays documented in test_training_objective.py.

Run:  python -m pytest tests\\unit\\test_gpt_model.py -q -p no:cacheprovider
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from transformer.core.gpt_model import DGBGPT  # noqa: E402

VOCAB, PAD, FIRST_REAL = 50, 0, 6
LN_V = math.log(VOCAB - FIRST_REAL)           # best possible loss on unpredictable tokens


def _tiny(seed: int = 0) -> DGBGPT:
    torch.manual_seed(seed)
    return DGBGPT(vocab_size=VOCAB, d_model=32, n_heads=2, n_layers=2, d_ff=64,
                  dropout=0.0, max_seq_len=64, pad_idx=PAD)


def _random_batch(gen, batch=16, seq=32):
    ids = torch.randint(FIRST_REAL, VOCAB, (batch, seq + 1), generator=gen)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()      # same pairing as the data loader


def _alternating_batch(gen, batch=16, seq=32):
    """a b a b a b ...  The token at t equals the token at t-2: needs attention to learn."""
    a = torch.randint(FIRST_REAL, VOCAB, (batch, 1), generator=gen)
    b = torch.randint(FIRST_REAL, VOCAB, (batch, 1), generator=gen)
    ids = torch.cat([a, b] * ((seq + 1) // 2 + 1), dim=1)[:, : seq + 1]
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def _train(model, batch_fn, steps, lr=3e-3, seed=1):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    losses = []
    for _ in range(steps):
        src, tgt = batch_fn(gen)
        loss = crit(model(src).reshape(-1, VOCAB), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


# --------------------------------------------------------------------- leakage / causality
def test_future_tokens_do_not_change_past_logits():
    model = _tiny().eval()
    gen = torch.Generator().manual_seed(2)
    src, _ = _random_batch(gen)
    t = 10
    src2 = src.clone()
    src2[:, t + 1:] = torch.randint(FIRST_REAL, VOCAB, src2[:, t + 1:].shape, generator=gen)
    with torch.no_grad():
        diff = (model(src)[:, : t + 1] - model(src2)[:, : t + 1]).abs().max().item()
    assert diff < 1e-5, f"future tokens leak into earlier positions (max diff {diff})"


def test_unpredictable_tokens_stay_at_chance_loss():
    """The T0.1 property, now holding: random tokens cannot be learned."""
    losses = _train(_tiny(), _random_batch, steps=300)
    final = sum(losses[-50:]) / 50
    print(f"\n[T0.2] random-token loss, last 50 steps: {final:.4f}  (chance floor ln(44) = {LN_V:.4f})")
    assert final > 0.9 * LN_V, f"loss {final:.3f} is below the honest floor {LN_V:.3f}: leakage"
    assert final < LN_V + 0.3, f"loss {final:.3f} is far above chance: training is not working"


def test_learns_a_real_pattern_on_unseen_data():
    """Held-out sequences, so memorising is impossible; the model must use attention."""
    model = _tiny()
    losses = _train(model, _alternating_batch, steps=500)
    gen = torch.Generator().manual_seed(999)
    src, tgt = _alternating_batch(gen, batch=64)
    model.eval()
    with torch.no_grad():
        held_out = nn.functional.cross_entropy(model(src).reshape(-1, VOCAB), tgt.reshape(-1)).item()
    print(f"\n[T0.2] pattern task: train start {losses[0]:.3f} -> held-out {held_out:.4f} (chance {LN_V:.3f})")
    assert held_out < 0.5, f"model failed to learn a trivial pattern (held-out loss {held_out:.3f})"


def test_right_padding_does_not_change_real_tokens():
    model = _tiny().eval()
    gen = torch.Generator().manual_seed(3)
    src, _ = _random_batch(gen, batch=4, seq=20)
    padded = torch.cat([src, torch.full((4, 8), PAD, dtype=torch.long)], dim=1)
    with torch.no_grad():
        diff = (model(src) - model(padded)[:, :20]).abs().max().item()
    assert diff < 1e-5


# --------------------------------------------------------------------- generation
def test_generate_is_deterministic_and_pads_after_eos():
    model = _tiny().eval()
    prompt = torch.tensor([[2, 10, 11, 12]])
    a = model.generate(prompt, max_new_tokens=6)
    b = model.generate(prompt, max_new_tokens=6)
    assert torch.equal(a, b) and a.shape == (1, 6)
    # force EOS: make token 3 the argmax everywhere by biasing the tied output row
    model2 = _tiny().eval()
    with torch.no_grad():
        model2.embed.weight[3] += 50.0
    out = model2.generate(prompt, max_new_tokens=6, eos_id=3)
    assert out[0, 0].item() == 3 or out.shape[1] <= 6
    sampled = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=10, top_p=0.9,
                             generator=torch.Generator().manual_seed(0))
    assert sampled.shape == (1, 5) and int(sampled.max()) < VOCAB


def test_greedy_decode_drops_trailing_eos_and_returns_only_new_tokens():
    model = _tiny().eval()
    prompt = torch.tensor([[2, 10, 11, 3]])                       # BOS a b EOS (tokenizer style)
    out = model.greedy_decode(prompt, bos_id=2, eos_id=3, max_len=4)
    expect = model.generate(prompt[:, :-1], max_new_tokens=4, eos_id=3)
    assert torch.equal(out, expect)


# --------------------------------------------------------------------- factory / checkpoints
def _cfg(arch):
    return SimpleNamespace(architecture=arch, d_model=32, n_heads=2, n_layers=2,
                           n_encoder_layers=1, n_decoder_layers=1, d_ff=64, dropout=0.1,
                           max_seq_len=64, pad_idx=0, tie_embeddings=True,
                           layer_norm_eps=1e-6, rope_theta=10000.0)


def test_build_model_factory_selects_architecture():
    from transformer.core.transformer_model import DGBTransformer
    from transformer.utils.model_helpers import build_model
    assert isinstance(build_model(_cfg("decoder_only"), 50), DGBGPT)
    assert isinstance(build_model(_cfg("encoder_decoder"), 50), DGBTransformer)
    assert build_model(_cfg("decoder_only"), 50, dropout=0.0).config["dropout"] == 0.0


def test_checkpoint_roundtrip_and_architecture_guard(tmp_path):
    from modules.utils.error_handler import CheckpointError
    from transformer.utils.model_helpers import build_model, load_checkpoint, save_checkpoint
    model = _tiny().eval()
    path = save_checkpoint(tmp_path / "m.pt", model, epoch=1, loss=1.5)
    clone = _tiny(seed=123).eval()
    payload = load_checkpoint(path, clone)
    assert payload["architecture"] == "decoder_only"
    ids = torch.randint(FIRST_REAL, VOCAB, (2, 12))
    with torch.no_grad():
        assert torch.allclose(model(ids), clone(ids), atol=1e-6)
    legacy = build_model(_cfg("encoder_decoder"), 50)
    with pytest.raises(CheckpointError):
        load_checkpoint(path, legacy)


# --------------------------------------------------------------------- trainer helper
def test_forward_loss_helper_matches_manual_loss_for_both_architectures():
    from trainer.core.training_loop import _forward_loss
    from transformer.utils.model_helpers import build_model
    gen = torch.Generator().manual_seed(5)
    src, tgt = _random_batch(gen, batch=4, seq=16)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    gpt = _tiny().eval()
    with torch.no_grad():
        manual = crit(gpt(src).reshape(-1, VOCAB), tgt.reshape(-1))
        assert torch.allclose(_forward_loss(gpt, src, tgt, crit), manual)
    legacy = build_model(_cfg("encoder_decoder"), 50).eval()
    with torch.no_grad():
        assert torch.isfinite(_forward_loss(legacy, src, tgt, crit))


# --------------------------------------------------------------------- inference engine
class _FakeTokenizer:
    vocab_size = VOCAB

    def encode(self, text, add_special_tokens=True, **_):
        return [2] + [FIRST_REAL + (ord(c) % 40) for c in text] + [3]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if not (skip_special_tokens and i < FIRST_REAL))


def test_inference_engine_runs_a_decoder_only_model():
    try:
        from inference.engine.generator import InferenceEngine
        from inference.sampling.top_p_sampler import GenerationConfig
    except ImportError as exc:                       # optional deps of the API stack
        pytest.skip(f"inference stack not importable here: {exc}")
    model = _tiny().eval()
    engine = InferenceEngine(model, _FakeTokenizer(), torch.device("cpu"))
    engine.warmup()
    cfg = GenerationConfig(max_new_tokens=5)
    text = engine._complete_sync("hello", cfg, False, None)
    assert isinstance(text, str)
    text_beam = engine._complete_sync("hello", cfg, True, None)      # beam falls back to greedy
    assert isinstance(text_beam, str)
    src = torch.tensor([_FakeTokenizer().encode("hi")])
    pairs = list(engine._stream_sampler.generate(src, GenerationConfig(max_new_tokens=4, do_sample=True)))
    assert 1 <= len(pairs) <= 4
