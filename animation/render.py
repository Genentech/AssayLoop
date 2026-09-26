"""Render the AssayLoop explainer.

    uv run python animation/render.py --preview        # contact sheet of key frames
    uv run python animation/render.py --stills 12.0 47.5
    uv run python animation/render.py                  # full MP4 + audio + srt

Audio is synthesized first so the visuals can be timed from the real beat
durations; pass --keep-vo to reuse an existing synthesis.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animation.assayloop_anim import narration as N   # noqa: E402
from animation.assayloop_anim import tts as TTS       # noqa: E402
from animation.assayloop_anim import style as S       # noqa: E402
from animation.assayloop_anim.canvas import Canvas     # noqa: E402
from animation.assayloop_anim.scenes import Timing, draw_frame  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
VOICE = os.path.join(os.path.dirname(HERE), ".voices", "en_US-ryan-high.onnx")


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# --- audio -----------------------------------------------------------------
def build_voiceover(schedule: list[dict], path: str) -> float:
    """Lay each beat's wav onto one silent track at its scheduled start."""
    with wave.open(schedule[0]["wav"], "rb") as w:
        rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()

    total = schedule[-1]["end"] + schedule[-1]["pad"] + 0.5
    track = np.zeros(int(total * rate), dtype=np.float32)
    for beat in schedule:
        with wave.open(beat["wav"], "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1)
        i = int(beat["start"] * rate)
        track[i:i + pcm.size] += pcm.astype(np.float32) / 32768.0

    peak = float(np.abs(track).max())
    if peak > 0:
        track *= 0.89 / peak                       # leave a little headroom
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes((track * 32767).astype(np.int16).tobytes())
    return total


def write_srt(schedule: list[dict], path: str) -> None:
    def ts(x: float) -> str:
        h, r = divmod(x, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")

    with open(path, "w") as fh:
        for i, b in enumerate(schedule, 1):
            fh.write(f"{i}\n{ts(b['start'])} --> {ts(b['end'])}\n"
                     f"{TTS.written(b['say'])}\n\n")


# --- video -----------------------------------------------------------------
def render_video(T: Timing, path: str, fps: int = S.FPS,
                 t0: float = 0.0, t1: float | None = None) -> str:
    t1 = T.total if t1 is None else t1
    n = int(round((t1 - t0) * fps))
    c = Canvas()
    proc = subprocess.Popen(
        [ffmpeg_exe(), "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{S.W}x{S.H}",
         "-r", str(fps), "-i", "-",
         "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE)
    try:
        for i in range(n):
            draw_frame(c, t0 + i / fps, T)
            proc.stdin.write(c.rgb())
            if i % (fps * 5) == 0:
                pct = 100.0 * i / max(n, 1)
                print(f"\r  {i:5d}/{n}  {pct:5.1f}%  t={t0 + i/fps:6.2f}s",
                      end="", flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
    print(f"\r  {n}/{n}  100.0%  ->  {path}")
    return path


def mux(video: str, audio: str, out: str) -> str:
    subprocess.run(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", video, "-i", audio,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
         "-movflags", "+faststart", out], check=True)
    return out


# --- previews ---------------------------------------------------------------
def contact_sheet(T: Timing, path: str, cols: int = 4, rows: int = 5) -> str:
    """One grid image sampling the whole timeline — the fast way to eyeball it."""
    import matplotlib.pyplot as plt
    n = cols * rows
    times = np.linspace(0.4, T.total - 0.4, n)
    c = Canvas()
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.45),
                             facecolor=S.BG)
    for ax, t in zip(axes.ravel(), times):
        draw_frame(c, float(t), T)
        img = np.frombuffer(c.rgb(), dtype=np.uint8).reshape(S.H, S.W, 4)
        ax.imshow(img)
        ax.set_title(f"{t:.1f}s  ·  {T.scene_of(float(t))}", color=S.GREY,
                     fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=88, facecolor=S.BG)
    print(f"  -> {path}")
    return path


def stills(T: Timing, times: list[float], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    c = Canvas()
    for t in times:
        draw_frame(c, t, T)
        p = os.path.join(outdir, f"t{t:07.2f}.png")
        c.fig.savefig(p, dpi=S.DPI, facecolor=S.BG)
        print(f"  -> {p}")


# --- main -------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="contact sheet only")
    ap.add_argument("--stills", nargs="*", type=float, help="render these times as PNGs")
    ap.add_argument("--clip", nargs=2, type=float, metavar=("T0", "T1"))
    ap.add_argument("--fps", type=int, default=S.FPS)
    ap.add_argument("--keep-vo", action="store_true", help="reuse existing narration")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    sched_path = os.path.join(OUT, "schedule.json")

    if args.keep_vo and os.path.exists(sched_path):
        schedule = N.load(sched_path)
    else:
        print("Synthesizing narration...")
        schedule = N.synthesize(os.path.join(OUT, "vo"), VOICE, quiet=False)
        N.save(schedule, sched_path)

    T = Timing(schedule)
    print(f"Timeline: {T.total:.2f}s")

    if args.preview:
        contact_sheet(T, os.path.join(OUT, "contact_sheet.png"))
        return
    if args.stills:
        stills(T, args.stills, os.path.join(OUT, "stills"))
        return

    print("Building voiceover track...")
    audio = os.path.join(OUT, "narration.wav")
    build_voiceover(schedule, audio)
    write_srt(schedule, os.path.join(OUT, "assayloop.srt"))

    print(f"Rendering {args.fps} fps...")
    t0, t1 = args.clip if args.clip else (0.0, T.total)
    silent = os.path.join(OUT, "_silent.mp4")
    render_video(T, silent, fps=args.fps, t0=t0, t1=t1)

    final = os.path.join(OUT, "assayloop_explainer.mp4")
    if args.clip:
        final = os.path.join(OUT, f"clip_{t0:.0f}_{t1:.0f}.mp4")
        os.replace(silent, final)
    else:
        mux(silent, audio, final)
        os.remove(silent)
    print(f"\nDone: {final}  ({os.path.getsize(final)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
