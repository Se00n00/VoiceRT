"""Weight loading + key-mapping helpers for Whisper-base.

Canonical layout is the HF ``openai/whisper-base`` checkpoint
(``model.safetensors``): ``model.encoder.*`` / ``model.decoder.*`` with the
LM head tied to ``model.decoder.embed_tokens`` (no separate tensor/bias --
the source engine relies on this).
"""
import glob
import os

__all__ = [
    "MODEL_ID",
    "REQUIRED_PREFIXES",
    "snapshot_path",
    "load_weights",
    "map_key",
    "check_coverage",
]

MODEL_ID = "openai/whisper-base"
REQUIRED_PREFIXES = (
    "model.encoder.conv1.weight",
    "model.encoder.conv2.weight",
    "model.encoder.embed_positions.weight",
    "model.encoder.layers.0.self_attn.q_proj.weight",
    "model.decoder.embed_tokens.weight",
    "model.decoder.embed_positions.weight",
    "model.decoder.layers.0.self_attn.q_proj.weight",
    "model.decoder.layer_norm.weight",
)


def snapshot_path(repo_id=MODEL_ID, local_dir=None):
    """Local HF snapshot dir (lazy huggingface_hub), downloading if needed."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, allow_patterns=["*.safetensors"],
                             local_dir=local_dir)


def load_weights(weights_path, device="cuda:0"):
    """Load ``.safetensors`` file or directory -> {name: float tensor}."""
    from safetensors.torch import load_file
    if os.path.isdir(weights_path):
        files = sorted(glob.glob(os.path.join(weights_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(
                "no .safetensors under %s" % weights_path)
        sd = {}
        for f in files:
            sd.update(load_file(f, device=device))
    else:
        sd = load_file(weights_path, device=device)
    out = {}
    for k, v in sd.items():
        out[map_key(k)] = v.float() if hasattr(v, "float") else v
    return out


def map_key(key):
    """Canonicalise one state-dict key.

    HF whisper checkpoints already use the ``model.*`` layout the engine
    indexes, so this is the identity for known keys; it also translates the
    common ``encoder.`` / ``decoder.`` (no ``model.``) prefix variants.
    """
    if key.startswith("model."):
        return key
    if key.startswith("encoder.") or key.startswith("decoder."):
        return "model." + key
    return key


def check_coverage(weights):
    """Verify required prefixes exist. Returns (ok, missing_list)."""
    keys = set(weights.keys())
    missing = [p for p in REQUIRED_PREFIXES
               if not any(k == p or k.startswith(p) for k in keys)]
    tied_ok = ("model.decoder.embed_tokens.weight" in keys)  # LM head tied
    return (not missing and tied_ok, missing)
