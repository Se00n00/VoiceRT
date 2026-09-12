"""Leg registry + lazy loaders reading configs/*.yaml.

Canonical legs: vad / stt / llm / tts. Config filenames differ from leg
names for historical reasons (whisper.yaml, qwen.yaml), so FILENAME maps
leg -> file. load_leg() lazily imports models.<leg>.model and instantiates
the first importable candidate class; when the model package is absent it
raises an ImportError that names the missing module (omit-and-report: the
caller keeps the leg as None and reports it instead of stubbing).
"""
import importlib
import os

import yaml

LEG_ORDER = ("vad", "stt", "llm", "tts")

FILENAME = {
    "vad": "vad.yaml",
    "stt": "whisper.yaml",
    "llm": "qwen.yaml",
    "tts": "tts.yaml",
}

# Leg -> candidate (module, class) pairs tried in order.
# Single-file models (models/<name>.py) are canonical; legacy
# models/<name>.model paths are kept as fallback.
CANDIDATES = {
    "vad": [
        ("models.silero_vad.model", "SileroVAD"),
        ("models.silero_vad.model", "SileroVad"),
        ("models.silero_vad.model", "VadEngine"),
    ],
    "stt": [
        ("models.whisper", "WhisperEngine"),
        ("models.whisper", "WhisperTriton"),
        ("models.whisper", "WhisperModel"),
        ("models.whisper.model", "WhisperEngine"),
        ("models.whisper.model", "WhisperTriton"),
        ("models.whisper.model", "WhisperModel"),
    ],
    "llm": [
        ("models.qwen", "QwenEngine"),
        ("models.qwen", "TinyLLM"),
        ("models.qwen", "QwenModel"),
        ("models.qwen.model", "QwenEngine"),
        ("models.qwen.model", "TinyLLM"),
        ("models.qwen.model", "QwenModel"),
    ],
    "tts": [
        ("models.tts", "KokoroEngine"),
        ("models.tts", "TtsEngine"),
        ("models.tts", "KokoroTTS"),
        ("models.tts.model", "KokoroEngine"),
        ("models.tts.model", "TtsEngine"),
        ("models.tts.model", "KokoroTTS"),
    ],
}


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def config_path_for(leg, config_dir=None):
    """Filesystem path of the YAML config for a leg."""
    if leg not in FILENAME:
        raise KeyError(f"unknown leg {leg!r}; expected one of {sorted(FILENAME)}")
    base = config_dir or os.path.join(_repo_root(), "configs")
    return os.path.join(base, FILENAME[leg])


def list_legs():
    """Canonical leg names in pipeline order."""
    return list(LEG_ORDER)


def load_config(leg, config_dir=None):
    """Parse and return the YAML config dict for a leg."""
    path = config_path_for(leg, config_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config for leg {leg!r} not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a YAML mapping")
    cfg.setdefault("_path", path)
    cfg.setdefault("_leg", leg)
    return cfg


def load_all_configs(config_dir=None):
    """{leg: config_dict} for every registered leg."""
    return {leg: load_config(leg, config_dir) for leg in LEG_ORDER}


def _first_importable(candidates):
    errors = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        cls = getattr(module, class_name, None)
        if cls is None:
            errors.append(f"{module_name} has no {class_name}")
            continue
        return cls
    return None, errors


def resolve_leg_class(leg):
    """Return the model class for a leg, or raise ImportError listing attempts."""
    if leg not in CANDIDATES:
        raise KeyError(f"unknown leg {leg!r}; expected one of {sorted(CANDIDATES)}")
    found = _first_importable(CANDIDATES[leg])
    if isinstance(found, tuple):
        _, errors = found
        tried = "; ".join(errors) if errors else "no candidates"
        raise ImportError(
            f"no model class importable for leg {leg!r} "
            f"(tried {CANDIDATES[leg]}; {tried}). "
            f"Expected e.g. models/qwen.py to define one."
        )
    return found


def load_leg(leg, config_dir=None, **overrides):
    """Instantiate a leg's model class with its YAML config as kwargs.

    Extra overrides win over config values. Keys starting with '_' and the
    'kernels' documentation list are not passed to the constructor.
    """
    cfg = load_config(leg, config_dir)
    cls = resolve_leg_class(leg)
    kwargs = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "kernels"}
    kwargs.update(overrides)
    try:
        return cls(**kwargs)
    except TypeError:
        # Fall back to a bare constructor when the class takes no config kwargs.
        return cls()


def leg_status(config_dir=None):
    """Per-leg availability: {leg: {'config': bool, 'class': name-or-None}}."""
    status = {}
    for leg in LEG_ORDER:
        try:
            load_config(leg, config_dir)
            has_cfg = True
        except (FileNotFoundError, ValueError):
            has_cfg = False
        try:
            cls = resolve_leg_class(leg)
            cls_name = f"{cls.__module__}.{cls.__name__}"
        except ImportError:
            cls_name = None
        status[leg] = {"config": has_cfg, "class": cls_name}
    return status
