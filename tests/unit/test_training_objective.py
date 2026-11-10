import logging
import math

import pytest
import torch
import torch.nn as nn

from transformer.core.transformer_model import DGBTransformer

logger = logging.getLogger(__name__)


@pytest.mark.xfail(strict=True, reason="label leakage — fixed in T0.2")
def test_training_objective_does_not_leak_labels():
    torch.manual_seed(101)
    original_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(original_num_threads, 2))
    try:
        pad_idx = 0
        model = DGBTransformer(
            vocab_size=50,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            d_ff=64,
            dropout=0.0,
            max_seq_len=32,
            pad_idx=pad_idx,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        printed_losses = {}

        model.train()
        for step in range(301):
            ids = torch.randint(6, 50, (8, 32))
            src, tgt = ids[:, :-1], ids[:, 1:]
            src_mask = model.make_padding_mask(src)
            tgt_mask = model.make_padding_mask(tgt)

            logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            if step in (0, 100, 200, 300):
                printed_losses[step] = loss.item()
                message = f"step {step}: loss={loss.item():.6f}"
                print(message, flush=True)
                logger.info(message)

            if step < 300:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        assert tuple(printed_losses) == (0, 100, 200, 300)
        assert printed_losses[300] > 0.8 * math.log(44)
    finally:
        torch.set_num_threads(original_num_threads)
