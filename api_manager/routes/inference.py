"""
api_manager/routes/inference.py
=================================
Inference routes — REST + SSE streaming.

FIX B3 (v3.0.0):
    cfg.tokenizer_dir() / cfg.models_dir() replaced with
    PathResolver.tokenizer_dir() / PathResolver.models_dir().

Endpoints
---------
POST /inference/generate          — sync completion (JSON response)
POST /inference/stream            — SSE token-by-token streaming
POST /v1/chat/completions         — OpenAI-compatible chat
POST /v1/chat/grounded            — Web-search grounded completion with citations
POST /v1/tokenizer/encode         — Tokenize a string
GET  /v1/tokenizer/vocab_size     — Vocabulary size
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api_manager.auth.api_security import get_current_user, TokenData
from modules.utils.error_handler import (
    InferenceError, TokenizerNotTrainedError, ModelInitError, CheckpointError
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["inference"])

# ── Global singletons loaded lazily on first request ─────────────────────────
_tokenizer = None
_model     = None
_device    = None
_engine    = None


def _get_resolver():
    """FIX B3: construct PathResolver — never call cfg.tokenizer_dir() directly."""
    from configs.loader import get_config
    from modules.utils.path_resolver import init_path_resolver
    cfg = get_config()
    return init_path_resolver(cfg.project.model_id, cfg)


def _load_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    from tokenizer.dgb_tokenizer import DGBTokenizer
    res     = _get_resolver()
    tok_dir = res.tokenizer_dir(create=False)  # FIX B3: correct call

    candidates = sorted(tok_dir.glob("*vocabulary.json"))
    if not candidates:
        raise TokenizerNotTrainedError()

    _tokenizer = DGBTokenizer.from_pretrained(tok_dir)
    logger.info("Tokenizer loaded: vocab_size=%d", _tokenizer.vocab_size)
    return _tokenizer


def _load_model():
    global _model, _device
    if _model is not None:
        return _model, _device

    from configs.loader import get_config
    from transformer.core.transformer_model import DGBTransformer
    from transformer.utils.model_helpers import resolve_device, load_checkpoint, latest_checkpoint

    cfg    = get_config()
    tf     = cfg.transformer
    res    = _get_resolver()
    _device = resolve_device(cfg.training.device)

    m = DGBTransformer(
        vocab_size=tf.vocab_size,
        d_model=tf.d_model,
        n_heads=tf.n_heads,
        n_encoder_layers=tf.n_encoder_layers,
        n_decoder_layers=tf.n_decoder_layers,
        d_ff=tf.d_ff,
        dropout=0.0,
        max_seq_len=tf.max_seq_len,
        pad_idx=tf.pad_idx,
        tie_embeddings=tf.tie_embeddings,
    ).to(_device)

    models_dir = res.models_dir(create=False)  # FIX B3: correct call
    ckpt = latest_checkpoint(models_dir)
    if ckpt:
        load_checkpoint(ckpt, m, device=_device)
    else:
        logger.warning("No model checkpoint found — using random weights")

    m.eval()
    _model = m
    return _model, _device


def _get_engine():
    global _engine
    if _engine is None:
        from inference.engine.generator import init_engine
        tok = _load_tokenizer()
        mdl, dev = _load_model()
        _engine = init_engine(mdl, tok, dev)
    return _engine


# ── Request / response models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt:         str
    max_new_tokens: int   = Field(default=256, ge=1, le=2048)
    temperature:    float = Field(default=1.0,  ge=0.01, le=4.0)
    top_p:          float = Field(default=0.9,  ge=0.0,  le=1.0)
    top_k:          int   = Field(default=50,   ge=0)
    do_sample:      bool  = True
    use_beam:       bool  = False
    beam_size:      int   = Field(default=4,    ge=1, le=16)


class GenerateResponse(BaseModel):
    text:           str
    input_tokens:   int
    output_tokens:  int
    model:          str = "dgb"


class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    model:          str              = "dgb-m"
    messages:       List[ChatMessage]
    max_tokens:     int   = Field(default=512,  ge=1, le=4096)
    temperature:    float = Field(default=1.0,  ge=0.01, le=4.0)
    top_p:          float = Field(default=0.9,  ge=0.0, le=1.0)
    stream:         bool  = False


class GroundedRequest(BaseModel):
    query:                str
    conversation_history: List[Dict[str, str]] = []
    search_provider:      str  = "brave"
    research_mode:        bool = True
    max_tokens:           int  = 1024


class TokenizeRequest(BaseModel):
    text:               str
    add_special_tokens: bool = True


class TokenizeResponse(BaseModel):
    tokens:     List[int]
    n_tokens:   int
    vocab_size: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/inference/generate", response_model=GenerateResponse)
async def generate(
    req:  GenerateRequest,
    user: TokenData = Depends(get_current_user),
):
    from inference.sampling.top_p_sampler import GenerationConfig
    from inference.sampling.beam_search import BeamSearchConfig
    try:
        engine = _get_engine()
        cfg    = GenerationConfig(
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            do_sample=req.do_sample,
        )
        bc   = BeamSearchConfig(beam_size=req.beam_size) if req.use_beam else None
        text = await engine.complete(req.prompt, cfg, use_beam=req.use_beam, beam_cfg=bc)
        tok  = _load_tokenizer()
        return GenerateResponse(
            text=text,
            input_tokens=len(tok.encode(req.prompt)),
            output_tokens=len(tok.encode(text)),
        )
    except (TokenizerNotTrainedError, CheckpointError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/inference/stream")
async def stream_generate(
    req:  GenerateRequest,
    user: TokenData = Depends(get_current_user),
):
    from inference.sampling.top_p_sampler import GenerationConfig
    engine = _get_engine()
    cfg    = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        do_sample=req.do_sample,
    )
    async def _iter() -> AsyncGenerator[str, None]:
        async for tok in engine.stream(req.prompt, cfg):
            yield f"data: {json.dumps({'token': tok})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_iter(), media_type="text/event-stream")


@router.post("/v1/chat/completions")
async def openai_chat(req: ChatRequest):
    """OpenAI-compatible /v1/chat/completions endpoint."""
    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    from inference.sampling.top_p_sampler import GenerationConfig
    engine = _get_engine()
    cfg    = GenerationConfig(
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    if req.stream:
        async def _stream_iter():
            async for tok in engine.stream(prompt, cfg):
                chunk = {
                    "choices": [{"delta": {"content": tok}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream_iter(), media_type="text/event-stream")

    text = await engine.complete(prompt, cfg)
    tok  = _load_tokenizer()
    return {
        "id": "dgb-completion",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens":     len(tok.encode(prompt)),
            "completion_tokens": len(tok.encode(text)),
            "total_tokens":      len(tok.encode(prompt)) + len(tok.encode(text)),
        },
    }


@router.post("/v1/chat/grounded")
async def grounded_chat(req: GroundedRequest):
    """
    Web-search grounded chat. Retrieves live sources, injects them as
    context, generates a response, and emits SSE:
      {type: sources, sources: [...]}
      {type: token, content: "..."}
      {type: citations, citations: [...]}
    """
    from inference.sampling.top_p_sampler import GenerationConfig
    engine = _get_engine()
    cfg    = GenerationConfig(max_new_tokens=req.max_tokens, temperature=0.7)

    async def _grounded_stream():
        sources = []
        if req.research_mode:
            try:
                from integrations.web_search.grounded_pipeline import (
                    search_and_build_context, extract_citations
                )
                sources, system_prompt = await search_and_build_context(req.query)
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                prompt = f"{system_prompt}\n\nUser: {req.query}\nAssistant:"
            except Exception as exc:
                logger.warning("Search failed: %s — falling back to direct inference", exc)
                prompt = req.query
                yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
        else:
            prompt = req.query
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"

        full_text = ""
        async for tok in engine.stream(prompt, cfg):
            full_text += tok
            yield f"data: {json.dumps({'type': 'token', 'content': tok})}\n\n"

        if sources:
            try:
                from integrations.web_search.context_builder import extract_citations
                from integrations.web_search.adapters.base import SearchResult
                sr_list = [SearchResult(**s) for s in sources if isinstance(s, dict)]
                citations = extract_citations(full_text, sr_list)
                yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
            except Exception:
                yield f"data: {json.dumps({'type': 'citations', 'citations': []})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(_grounded_stream(), media_type="text/event-stream")


@router.post("/v1/tokenizer/encode", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest):
    try:
        tok    = _load_tokenizer()
        tokens = tok.encode(req.text, add_special_tokens=req.add_special_tokens)
        return TokenizeResponse(tokens=tokens, n_tokens=len(tokens), vocab_size=tok.vocab_size)
    except TokenizerNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/v1/tokenizer/vocab_size")
async def vocab_size():
    try:
        tok = _load_tokenizer()
        return {"vocab_size": tok.vocab_size, "trained": tok.is_trained}
    except Exception:
        return {"vocab_size": 0, "trained": False}
