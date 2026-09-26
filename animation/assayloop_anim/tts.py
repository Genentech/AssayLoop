"""Speech synthesis with explicit control over pacing.

Kokoro (StyleTTS2-derived) handles prosody inside a sentence well, but the
thing that makes a read sound rushed is the *gaps between* clauses, which no
TTS gets right from punctuation alone. So we synthesize clause by clause and
place the silences ourselves:

    "."  ";"       end of thought      -> long gap
    ","  ":"       clause break        -> short gap
    "|"  "||"      author's pause      -> explicit, whatever the text says

That way a line can be re-paced without touching the words, and the same
script reads the same way every render.
"""

from __future__ import annotations

import os
import re
import warnings

import numpy as np

SR = 24000
DEFAULT_VOICE = "am_liam"   # brisker and lighter than am_michael

# Gap in seconds after a chunk, keyed by what terminated it.
GAP = {".": 0.34, "?": 0.38, "!": 0.36, ";": 0.28, ",": 0.17, ":": 0.24,
       "|": 0.32, "||": 0.62, "": 0.20}

_PIPELINE = None

# misaki's inline pronunciation override: [written form](/phonemes/). Used
# sparingly, where the phonemes are right but the model still garbles them.
_OVERRIDE = re.compile(r"\[([^\]]+)\]\(/[^/]*/\)")


def written(text: str) -> str:
    """The human-readable form of a narration line.

    Strips the pause markers and any pronunciation overrides, so subtitles
    show prose rather than the markup the synthesizer needs.
    """
    t = _OVERRIDE.sub(r"\1", text)
    t = re.sub(r"\s*\|\|?\s*", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _pipeline(device: str | None = None):
    global _PIPELINE
    if _PIPELINE is None:
        warnings.filterwarnings("ignore")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import torch
        from kokoro import KPipeline
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _PIPELINE = KPipeline(lang_code="a", device=dev)
    return _PIPELINE


def split_clauses(text: str) -> list[tuple[str, str]]:
    """Break a line into (chunk, terminator) pairs.

    Splits only at sentence ends and at explicit `|` / `||` markers. Commas are
    deliberately left *inside* a chunk: the model needs the whole sentence to
    get its intonation right, and handing it a two-word fragment is exactly how
    a read starts sounding clipped.
    """
    out: list[tuple[str, str]] = []
    for part in re.split(r"(\|\|?)", text):
        if part in ("|", "||"):
            if out:                      # the pause belongs to the chunk before it
                out[-1] = (out[-1][0], part)
            continue
        for m in re.finditer(r"[^.?!;]+([.?!;]?)", part):
            chunk = m.group(0).strip()
            if chunk.strip(".?!;, "):
                out.append((chunk, m.group(1)))
    return out


def synthesize_line(text: str, voice: str = DEFAULT_VOICE,
                    speed: float = 1.0, device: str | None = None) -> np.ndarray:
    """Render one narration line to mono float32 at SR, with placed pauses.

    Note on articulation: a `|` splits the line into a separate synthesis
    call, and short chunks occasionally come out slurred even when the
    phonemes are right — "is this gene a hit?" once rendered closer to "is
    this join a hit?", though its phonemes were a clean /dʒiːn/. Most short
    chunks are fine, so there is no useful rule to enforce here; if a word
    comes out wrong, the first thing to try is removing the `|` before it so
    the model sees the whole sentence.
    """
    import torch
    pipe = _pipeline(device)
    pieces: list[np.ndarray] = []
    clauses = split_clauses(text)
    for i, (clause, term) in enumerate(clauses):
        segs = list(pipe(clause, voice=voice, speed=speed))
        if not segs:
            continue
        audio = torch.cat([s.audio for s in segs]).cpu().numpy().astype(np.float32)
        audio = _trim_silence(audio)
        pieces.append(audio)
        if i < len(clauses) - 1:         # no trailing gap; the beat pad owns that
            pieces.append(np.zeros(int(GAP.get(term, GAP[""]) * SR), np.float32))
    if not pieces:
        return np.zeros(1, np.float32)
    return np.concatenate(pieces)


def _trim_silence(x: np.ndarray, thresh: float = 2e-3, keep: int = 480) -> np.ndarray:
    """Strip leading/trailing near-silence so our gaps are the only gaps."""
    loud = np.where(np.abs(x) > thresh)[0]
    if loud.size == 0:
        return x
    a = max(0, loud[0] - keep)
    b = min(x.size, loud[-1] + keep)
    return x[a:b]


def write_wav(path: str, audio: np.ndarray, sr: int = SR) -> float:
    import wave
    audio = np.clip(audio, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    return audio.size / sr


def sample_voices(text: str, voices: list[str], outdir: str) -> None:
    """Render the same line in several voices, for picking one by ear."""
    os.makedirs(outdir, exist_ok=True)
    for v in voices:
        p = os.path.join(outdir, f"voice_{v}.wav")
        dur = write_wav(p, synthesize_line(text, voice=v))
        print(f"  {v:12s} {dur:5.2f}s  -> {p}")
