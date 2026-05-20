"""Unit tests for core DGB modules."""
import pytest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_config_loads():
    from configs.loader import get_config, reload_config
    reload_config()
    cfg = get_config()
    assert cfg.project.model_id == "dgb1"
    assert cfg.training.learning_rate == 3e-4   # FIX B4 verified
    assert cfg.training.seed == 42               # FIX T4 verified
    assert cfg.training.num_workers == 0         # FIX T6 verified
    assert cfg.training.warmup_steps >= 1000
    assert cfg.transformer.max_seq_len == 512

def test_config_b4_lr_wired():
    from configs.loader import get_config, reload_config
    reload_config()
    cfg = get_config()
    from trainer.core.training_loop import TrainingConfig
    train_cfg = TrainingConfig.from_cfg(cfg)
    assert train_cfg.learning_rate == cfg.training.learning_rate
    assert train_cfg.learning_rate == 3e-4

def test_config_b1_max_seq_len():
    from configs.loader import get_config, reload_config
    reload_config()
    cfg = get_config()
    assert cfg.transformer.max_seq_len != cfg.tokenizer.vocab_size, \
        "B1: max_seq_len must NOT equal vocab_size"
    assert cfg.transformer.max_seq_len == 512

def test_transformer_heads_divisor():
    from configs.loader import get_config, reload_config
    reload_config()
    cfg = get_config()
    tf  = cfg.transformer
    assert tf.d_model % tf.n_heads == 0

def test_path_resolver():
    from configs.loader import get_config, reload_config
    reload_config()
    cfg = get_config()
    from modules.utils.path_resolver import init_path_resolver
    res = init_path_resolver(cfg.project.model_id, cfg)
    assert "dgb1" in str(res.tokenizer_dir())
    assert "dgb1" in str(res.models_dir())

def test_safe_writer_roundtrip(tmp_path):
    from modules.utils.safe_writer import atomic_write_json, compute_checksum, write_checksum, verify_checksum
    import json
    data = {"key": "value", "n": 42}
    p = tmp_path / "test.json"
    atomic_write_json(p, data)
    loaded = json.loads(p.read_text())
    assert loaded == data
    write_checksum(p)
    assert verify_checksum(p) is True

def test_run_context():
    from modules.utils.run_context import RunContext
    ctx = RunContext(model_id="dgb1", run_id="20260518130915")
    assert ctx.run_id == "20260518130915"
    assert ctx.checkpoint_name(5, 6.6502) == "20260518130915_epoch_005_loss_6.6502.pt"

def test_pre_tokenizer():
    from tokenizer.core.pre_tokenizer import PreTokenizer
    pre = PreTokenizer(add_eow=True)
    words = pre.split_words("Hello world!")
    assert len(words) > 0

def test_bpe_processor_train():
    from tokenizer.core.bpe_processor import BPEProcessor
    from collections import Counter
    freq = Counter({"hello</w>": 10, "world</w>": 8, "hell</w>": 5})
    bpe  = BPEProcessor(num_merges=5, min_freq=2)
    bpe.train(freq)
    assert isinstance(bpe.merges, list)

def test_memory_manager():
    from modules.utils.memory_manager import MemoryManager
    mm = MemoryManager(warn_gb=0.001, error_gb=0.0)
    mm.check(100)   # should not raise

def test_streaming_event():
    from modules.utils.streaming import StreamEvent
    ev = StreamEvent.metric(epoch=1, loss=2.5)
    sse = ev.to_sse()
    assert "event:" in sse
    assert "data:" in sse

def test_pipeline_state():
    from modules.utils.pipeline_state import PipelineState, StageStatus
    state = PipelineState(model_id="dgb1")
    rec   = state.get("dataset_clean")
    assert rec.status == StageStatus.PENDING
    rec.mark_running()
    assert rec.status == StageStatus.RUNNING
    rec.mark_completed(files=3)
    assert rec.is_done

def test_beam_search_config():
    from inference.sampling.beam_search import BeamSearchConfig, Hypothesis
    cfg = BeamSearchConfig(beam_size=4, max_length=50)
    assert cfg.beam_size == 4
    h = Hypothesis(tokens=[2, 100, 200, 3], log_prob=-1.5, complete=True)
    assert h.score(1.0) == -1.5 / 4

def test_rate_limiter():
    from security.rate_limiter import RateLimiter, RateLimit
    rl   = RateLimiter()
    tier = RateLimit(limit=3, window_sec=60)
    rl.set_tier("test_key", tier)
    for _ in range(3):
        stats = rl.check("test_key")
    from modules.utils.error_handler import RateLimitError
    with pytest.raises(RateLimitError):
        rl.check("test_key")
