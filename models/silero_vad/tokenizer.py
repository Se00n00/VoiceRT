"""Segment post-processing for VAD output.

VAD models emit no text, so this module owns segment formatting: merging
nearby speech spans and rendering SubRip (SRT) subtitles.
"""

__all__ = [
    "merge_segments",
    "filter_short",
    "to_srt",
    "format_timestamp",
]


def merge_segments(segments, max_gap_s=0.25, pad_s=0.0):
    """Merge spans separated by less than ``max_gap_s``. Real code."""
    segs = sorted((float(s), float(e)) for s, e in segments)
    if not segs:
        return []
    out = [[segs[0][0], segs[0][1]]]
    for s, e in segs[1:]:
        if s - out[-1][1] <= max_gap_s:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    if pad_s:
        out = [[max(0.0, s - pad_s), e + pad_s] for s, e in out]
    return [(s, e) for s, e in out]


def filter_short(segments, min_dur_s=0.15):
    """Drop spans shorter than ``min_dur_s``."""
    return [(s, e) for s, e in segments if (e - s) >= min_dur_s]


def format_timestamp(seconds):
    """Seconds -> ``HH:MM:SS,mmm`` SRT timestamp."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def to_srt(segments, texts=None):
    """[(start, end)] (+ optional per-segment texts) -> SRT string."""
    lines = []
    for i, (s, e) in enumerate(segments):
        lines.append(str(i + 1))
        lines.append("%s --> %s" % (format_timestamp(s), format_timestamp(e)))
        if texts is not None and i < len(texts):
            lines.append(str(texts[i]))
        else:
            lines.append("[speech]")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + ("\n" if lines else "")
