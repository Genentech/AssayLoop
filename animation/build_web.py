"""Build the interactive web version from the same narration schedule as the MP4.

    uv run python animation/build_web.py

Injects the measured beat timings into web/index.template.html and transcodes
the narration to mp3, so the browser version and the video can never drift
apart: both are generated from animation/out/schedule.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animation.assayloop_anim import narration as N  # noqa: E402
from animation.assayloop_anim import tts as TTS      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
OUT = os.path.join(HERE, "out")


def build(out_html: str, audio_href: str, audio_out: str,
          video_href: str, srt_href: str) -> None:
    schedule = N.load(os.path.join(OUT, "schedule.json"))
    total = round(schedule[-1]["end"] + schedule[-1]["pad"], 2)

    beats = [
        {"scene": b["scene"], "start": round(b["start"], 3),
         "end": round(b["end"], 3), "say": TTS.written(b["say"])}
        for b in schedule
    ]

    tpl = open(os.path.join(WEB, "index.template.html")).read()
    html = tpl.replace("__BEATS__", json.dumps(beats, ensure_ascii=False))
    html = html.replace("__TOTAL__", str(total))
    html = html.replace("__AUDIO__", audio_href)
    html = html.replace("__VIDEO__", video_href)
    html = html.replace("__SRT__", srt_href)

    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w") as fh:
        fh.write(html)
    print(f"  -> {out_html}  ({len(html)/1024:.0f} KB, {len(beats)} beats, {total}s)")

    import imageio_ffmpeg
    os.makedirs(os.path.dirname(os.path.abspath(audio_out)), exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-i", os.path.join(OUT, "narration.wav"),
                    "-c:a", "libmp3lame", "-b:a", "128k", audio_out], check=True)
    print(f"  -> {audio_out}  ({os.path.getsize(audio_out)/1e6:.1f} MB)")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WEB, "index.html"),
                    help="where to write the built page")
    ap.add_argument("--audio-href", default="narration.mp3",
                    help="src the page uses for the narration, relative to itself")
    ap.add_argument("--audio-out", default=None,
                    help="where to write the mp3 (defaults next to --out)")
    ap.add_argument("--video-href", default="../out/assayloop_explainer.mp4")
    ap.add_argument("--srt-href", default="../out/assayloop.srt")
    args = ap.parse_args()
    audio_out = args.audio_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), args.audio_href)
    build(args.out, args.audio_href, audio_out, args.video_href, args.srt_href)


if __name__ == "__main__":
    main()
