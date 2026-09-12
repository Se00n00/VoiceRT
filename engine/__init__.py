"""engine: VoiceEngine, serial VoicePipe, and pipeline helpers."""
from engine.audio import duration_s, load_wav, normalize, resample, save_wav, to_mono
from engine.engine import STT_SR, SYSTEM_PROMPT, TTS_SR, VoiceEngine
from engine.executor import LegExecutor
from engine.generation import (
    run_token_loop,
    sample_next_token,
    should_stop,
    strip_stop_tail,
)
from engine.model import (
    CANDIDATES,
    FILENAME,
    LEG_ORDER,
    config_path_for,
    leg_status,
    list_legs,
    load_all_configs,
    load_config,
    load_leg,
    resolve_leg_class,
)
from engine.pipeline import VoicePipe
from engine.streaming import (
    SPLIT,
    SentenceSplitter,
    done_frame,
    format_sse,
    is_complete_sentence,
    split_sentences,
)

__all__ = [
    "duration_s",
    "load_wav",
    "normalize",
    "resample",
    "save_wav",
    "to_mono",
    "STT_SR",
    "SYSTEM_PROMPT",
    "TTS_SR",
    "VoiceEngine",
    "LegExecutor",
    "run_token_loop",
    "sample_next_token",
    "should_stop",
    "strip_stop_tail",
    "CANDIDATES",
    "FILENAME",
    "LEG_ORDER",
    "config_path_for",
    "leg_status",
    "list_legs",
    "load_all_configs",
    "load_config",
    "load_leg",
    "resolve_leg_class",
    "VoicePipe",
    "SPLIT",
    "SentenceSplitter",
    "done_frame",
    "format_sse",
    "is_complete_sentence",
    "split_sentences",
]
