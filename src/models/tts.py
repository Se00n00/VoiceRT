"""TTS leg: Kokoro synthesis behind a clean async class."""
import asyncio
import difflib
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Kokoro ships one .pt style vector per voice under
# `voices/<name>.pt` and fetches it on first use (kokoro/pipeline.py:
# load_single_voice), so the catalogue below is the only way to know what
# can be picked without touching the hub. Names are
# <lang><gender>_<name>; lang/gender are derived from the prefix, never
# listed twice. Mirrors hexgrad/Kokoro-82M main @ f3ff357 (54 voices).
_VOICES_BY_PREFIX = {
    "af": ("af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
           "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"),
    "am": ("am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
           "am_michael", "am_onyx", "am_puck", "am_santa"),
    "bf": ("bf_alice", "bf_emma", "bf_isabella", "bf_lily"),
    "bm": ("bm_daniel", "bm_fable", "bm_george", "bm_lewis"),
    "ef": ("ef_dora",),
    "em": ("em_alex", "em_santa"),
    "ff": ("ff_siwis",),
    "hf": ("hf_alpha", "hf_beta"),
    "hm": ("hm_omega", "hm_psi"),
    "if": ("if_sara",),
    "im": ("im_nicola",),
    "jf": ("jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro"),
    "jm": ("jm_kumo",),
    "pf": ("pf_dora",),
    "pm": ("pm_alex", "pm_santa"),
    "zf": ("zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"),
    "zm": ("zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"),
}

_GENDER = {"f": "female", "m": "male"}


class TtsVoice(BaseModel):
    """One catalogue entry. `lang`/`language` come from kokoro's LANG_CODES."""

    model_config = ConfigDict(frozen=True)

    name: str
    lang: str
    language: str
    gender: str

    def __str__(self) -> str:
        return f"{self.name:<16} {self.gender:<6} {self.language}"


def _lang_codes() -> dict:
    # Imported lazily: kokoro pulls misaki/espeak on import and callers that
    # only want the catalogue should not pay for the phonemizer.
    try:
        from kokoro.pipeline import ALIASES, LANG_CODES

        return dict(LANG_CODES), dict(ALIASES)
    except Exception:
        return {}, {}


_VOICE_LANG_CODES, _LANG_ALIASES = _lang_codes()


def _build_catalogue() -> dict:
    codes, _ = _lang_codes()
    out = {}
    for prefix, names in _VOICES_BY_PREFIX.items():
        lang = prefix[0]
        language = codes.get(lang, lang)
        gender = _GENDER.get(prefix[1], "unknown")
        for name in names:
            out[name] = TtsVoice(name=name, lang=lang, language=language,
                               gender=gender)
    return out


VOICES = _build_catalogue()


def voice_info(name: str) -> TtsVoice:
    """Catalogue entry for `name`, or raise with the closest real options.

    Typo-tolerant: 'af_hear' resolves to af_heart, the same way the TUI
    command palette suggests, so a bad name is never silently a silent synth.
    """
    key = str(name).strip().lower()
    if key in VOICES:
        return VOICES[key]
    near = difflib.get_close_matches(key, VOICES, n=3, cutoff=0.6)
    hint = f" did you mean {' or '.join(repr(n) for n in near)}?" if near else ""
    raise ValueError(
        f"unknown voice {name!r}:{hint} pick from list_voices()")


def _resolve_lang(lang: str) -> str:
    """Accept 'a' or an alias like 'en-gb' -> 'b'."""
    text = str(lang).strip().lower()
    if text in _VOICE_LANG_CODES:
        return text
    return _LANG_ALIASES.get(text, text)


def list_voices(lang: str | None = None, gender: str | None = None):
    """Catalogue voices, optionally filtered. Returns (list, language map).

    Returns the matching :class:`TtsVoice` entries plus the language codes
    actually represented, so a caller can print "e: ['ef_dora', 'em_alex']"
    without a second lookup. Alias langs ('en-gb') are accepted.
    """
    want = _resolve_lang(lang) if lang else None
    sex = None
    if gender:
        sex = gender.strip().lower()
        sex = _GENDER.get(sex, sex)
    picked = [
        v for v in VOICES.values()
        if (want is None or v.lang == want)
        and (sex is None or v.gender == sex)
    ]
    picked.sort(key=lambda v: v.name)
    langs = {}
    for v in picked:
        langs.setdefault(v.lang, []).append(v.name)
    return picked, langs


class TtsConfig(BaseModel):
    """No YAML: construct (or override fields) in code."""

    model_config = ConfigDict(frozen=True)

    voice: str = "af_heart"
    lang: str = "a"
    sample_rate: int = 24000
    speed: float = 1.0
    device: str = "cuda"


class TtsAudio(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    wav: Any = Field(default_factory=lambda: np.zeros(0, dtype=np.float32))  # float32 mono
    sample_rate: int = 24000
    sentence: str = ""
    synth_s: float = 0.0

    @field_validator("wav", mode="before")
    @classmethod
    def _coerce_wav(cls, v):
        return np.asarray(v if v is not None else [],
                          dtype=np.float32).ravel()


class TtsModel:
    """Async facade over the proven Kokoro leg."""

    def __init__(self, config: TtsConfig | None = None):
        self.config = config or TtsConfig()
        self._leg = None

    # --- voices ------------------------------------------------------------

    def _apply_voice(self, name: str) -> None:
        """Force a voice id onto config and a live pipeline, unvalidated."""
        self.config = self.config.model_copy(update={"voice": name})
        if self._leg is not None:
            self._leg.voice = name

    @property
    def voice(self) -> str:
        """Currently selected voice id."""
        return self.config.voice

    def voices(self, lang: str | None = None, gender: str | None = None):
        """Catalogue entries this model can speak with (see list_voices)."""
        picked, _ = list_voices(lang, gender)
        return picked

    def describe_voices(self, lang: str | None = None,
                        gender: str | None = None) -> str:
        """Human-readable catalogue, one line per voice, grouped by language.

        Cheap enough for a REPL or an /voice help listing; loads no weights.
        """
        picked, langs = list_voices(lang, gender)
        if not picked:
            return "no voices match that language/gender filter"
        codes = _VOICE_LANG_CODES
        out = [f"{len(picked)} voice{'s' if len(picked) != 1 else ''}"]
        for lc in sorted(langs):
            head = f"\n{lc} ({codes.get(lc, lc)}):"
            out.append(head)
            for v in picked:
                if v.lang == lc:
                    mark = "*" if v.name == self.config.voice else " "
                    out.append(f" {mark} {v}")
        return "\n".join(out)

    def set_voice(self, voice: str, *, strict: bool = True) -> "TtsModel":
        """Switch voice at runtime. Returns self so calls can chain.

        Kokoro resolves `voice` to a style-vector id, so switching costs
        nothing once the pipeline exists: no reload, no re-download unless
        that specific voice has not been fetched yet (kokoro caches each
        `voices/<name>.pt` after the first use). Both a warmed and a cold
        pipeline are updated, so this is safe before or after warm().

        With strict=True (default) a voice whose language disagrees with
        self.config.lang is refused: kokoro accepts e.g. bf_emma under lang
        'a' without complaint and the result is subtly wrong-accented
        speech, not an error. Pass strict=False to override. Note that
        gender never trips this guard — am_michael is American English and
        is a valid swap for af_heart.
        """
        info = voice_info(voice)
        if strict and info.lang != self.config.lang:
            raise ValueError(
                f"voice {info.name!r} is {info.gender} {info.language} "
                f"(lang {info.lang!r}) but this TtsModel is lang "
                f"{self.config.lang!r}; set_voice(lang={info.lang!r}) first "
                f"or pass strict=False")
        self._apply_voice(info.name)
        return self

    def set_lang(self, lang: str, *, strict: bool = True) -> "TtsModel":
        """Switch language, picking a voice that speaks it when needed.

        Keeps the current voice if it already matches the new language,
        otherwise falls back to that language's first catalogue entry so the
        two never drift apart.
        """
        code = _resolve_lang(lang)
        self.config = self.config.model_copy(update={"lang": code})
        if self._leg is not None:
            self._leg.lang_code = code
        cur = VOICES.get(self.config.voice)
        if not strict or (cur and cur.lang == code):
            return self
        picked, _ = list_voices(lang=code)
        return self.set_voice(picked[0].name, strict=False) if picked else self

    def next_voice(self) -> str:
        """Step to the next voice for the current language; returns its name."""
        picked, _ = list_voices(lang=self.config.lang)
        if not picked:
            return self.config.voice
        names = [v.name for v in picked]
        i = names.index(self.config.voice) if self.config.voice in names else -1
        return self.set_voice(names[(i + 1) % len(names)]).voice

    def _backend(self):
        if self._leg is None:
            import torch

            # fused single-file model: src/models/kokoro.py (batched, VRAM-aware, fused tts kernels)
            from src.models.kokoro import KokoroFused

            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self._leg = KokoroFused(
                lang_code=self.config.lang,
                voice=self.config.voice,
                device=device,
                sample_rate=self.config.sample_rate,
                enhance=False,
                batch_size=4,
                speed=self.config.speed,
            )
        return self._leg

    async def warm(self, voice: str | None = None) -> "TtsModel":
        from src.models.runtime.guard import TTS_RAM_MB, TTS_VRAM_MB, preflight

        preflight("tts warm", vram_mb=TTS_VRAM_MB, ram_mb=TTS_RAM_MB)
        if voice:
            self.set_voice(voice)

        def _run():
            leg = self._backend()
            try:
                leg.speak("warmup.")
            except Exception:
                pass
            return leg

        await asyncio.to_thread(_run)
        return self

    async def speak(self, text: str, voice: str | None = None) -> TtsAudio:
        """Full text -> one audio chunk. Empty text -> empty audio.

        `voice` overrides the selection for this call only and is restored
        afterwards, even if synthesis raises. Omitting it uses whatever
        set_voice last selected.
        """
        import time as _time

        text = (text or "").strip()
        if not text:
            return TtsAudio(wav=np.zeros(0, dtype=np.float32),
                            sample_rate=self.config.sample_rate)
        previous = self.config.voice
        if voice:
            self.set_voice(voice)
        leg = self._backend()
        t0 = _time.perf_counter()

        def _run():
            return leg.speak(text)

        try:
            wav, sr = await asyncio.to_thread(_run)
        finally:
            if voice:
                self._apply_voice(previous)
        return TtsAudio(wav=np.asarray(wav, dtype=np.float32),
                        sample_rate=int(sr), sentence=text,
                        synth_s=_time.perf_counter() - t0)

    async def speak_stream(self, sentences, voice: str | None = None):
        """Iterable of sentences -> :class:`TtsAudio` per sentence."""
        for sent in sentences:
            if (sent or "").strip():
                yield await self.speak(sent, voice=voice)
