"""Weight loading for Qwen2.5 + tied-LM-head handling.

Qwen2.5-0.5B-Instruct ties the LM head to the token embeddings: checkpoints
may or may not ship a separate ``lm_head.weight``. :func:`lm_head_weight`
encodes that rule in one place.
"""
import glob
import os

__all__ = [
    "MODEL_ID",
    "FALLBACK_DIMS",
    "snapshot_path",
    "load_weights",
    "load_config",
    "lm_head_weight",
]

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Offline fallback for Qwen2.5-0.5B (matches HF config).
FALLBACK_DIMS = {
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "hidden_size": 896,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
}


def snapshot_path(repo_id=MODEL_ID, local_dir=None):
    """Local HF snapshot dir (lazy huggingface_hub), downloading if needed."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, allow_patterns=["*.safetensors"],
                             local_dir=local_dir)


def load_weights(weights_path=None, device="cuda:0", repo_id=MODEL_ID):
    """Load all ``*.safetensors`` from a dir (or HF snapshot) -> dict."""
    from safetensors.torch import load_file
    if weights_path is None:
        weights_path = snapshot_path(repo_id)
    if os.path.isdir(weights_path):
        files = sorted(glob.glob(os.path.join(weights_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(
                "no .safetensors under %s" % weights_path)
        w = {}
        for f in files:
            w.update(load_file(f, device=device))
        return dict(w)
    return dict(load_file(weights_path, device=device))


def load_config(repo_id=MODEL_ID):
    """Model dims via lazy transformers, else the 0.5B fallback table."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(repo_id)
        return {
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "hidden_size": cfg.hidden_size,
            "rms_norm_eps": float(cfg.rms_norm_eps),
            "max_position_embeddings": cfg.max_position_embeddings,
            "rope_theta": float(getattr(cfg, "rope_theta", None)
                                or 1000000.0),
        }
    except Exception:
        return dict(FALLBACK_DIMS)


def lm_head_weight(w):
    """Return the output-projection matrix, honouring the tied LM head."""
    if "lm_head.weight" in w:
        return w["lm_head.weight"]
    return w["model.embed_tokens.weight"]
