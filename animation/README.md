# AssayLoop explainer animation

A ~3 minute 3Blue1Brown-style explainer for *Biology-in-the-loop: Amortized
Adaptive Hit Discovery in CRISPR Screens* ([arXiv:2609.11877](https://arxiv.org/abs/2609.11877)),
in two forms:

| Output | Path |
|---|---|
| Video, 1920×1080 @ 60 fps, with voiceover | `out/assayloop_explainer.mp4` |
| Subtitles | `out/assayloop.srt` |
| Interactive web version | `web/index.html` + `web/narration.mp3` |
| Poster / teaser still | `out/poster.png`, `out/poster_960.png` |

Published on the project site as `docs/explainer.html`, with the MP4, SRT, mp3 and
poster under `docs/assets/explainer/`.

## Running it

```bash
uv run python animation/render.py              # synthesize VO, render MP4 + srt
uv run python animation/build_web.py           # build the local web version
uv run python animation/make_poster.py         # teaser still

# rebuild the published page (paths are relative to docs/)
uv run python animation/build_web.py --out docs/explainer.html \
    --audio-href assets/explainer/narration.mp3 \
    --video-href assets/explainer/assayloop-explainer.mp4 \
    --srt-href   assets/explainer/assayloop-explainer.srt
```

Useful while iterating — both skip the ~1 min of speech synthesis with
`--keep-vo`, reusing `out/schedule.json`:

```bash
uv run python animation/render.py --keep-vo --preview          # contact sheet
uv run python animation/render.py --keep-vo --stills 15.4 95.9 # single frames
uv run python animation/render.py --keep-vo --clip 80 100      # a range, no audio
```

## Why not Manim

Manim needs `pangocairo`, whose development headers are not installable on this
cluster without root (only the runtime `.so` is present, and `manimpango`
publishes no Linux wheels). `assayloop_anim/canvas.py` is a small stand-in
covering the primitives the storyboard uses — draw-on paths, fade-and-rise
entrances, typewriter text — over a 16×9 world-coordinate frame that mirrors
manim's convention. Frames go to `ffmpeg` via the binary bundled with
`imageio-ffmpeg`; speech is [piper](https://github.com/OHF-Voice/piper1-gpl)
running locally on CPU.

## How it fits together

```
narration.py   voiceover script -> per-beat wavs -> measured durations
     |
schedule.json  the single source of timing
     |
     +-- scenes.py  ---> render.py      ---> out/assayloop_explainer.mp4
     +-------------- ---> build_web.py  ---> web/index.html
```

Visuals are timed **from** the audio, never the other way round: each beat is
synthesized, its real duration measured, and the scene code expresses every
animation as "between these two seconds, take this quantity from 0 to 1"
against those measured boundaries. Rewording a line reflows the animation
instead of desynchronizing it.

`web/index.html` is generated — edit `web/index.template.html` and rebuild. Its
JS mirrors the Python scene-for-scene, sharing the coordinate system, easing
functions and palette, and uses the audio element as its clock while playing so
picture cannot drift. Both sides bind their beats *by scene name*, so inserting
a line into the script cannot silently re-point a later scene at the wrong audio.

## Voice

Speech is [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) (StyleTTS2-derived),
run locally on GPU. Pacing is the part TTS usually gets wrong, so `tts.py` does not
hand it a whole paragraph: it splits each line at sentence ends and at explicit
markers in the script, then places the silences itself.

    .  ;     end of thought      long gap
    |        author's pause      ~0.32 s
    ||       longer pause        ~0.62 s

Commas are deliberately *not* split on — the model needs a whole sentence to get
its intonation right, and feeding it a two-word fragment is exactly how a read
starts sounding clipped. Change the voice with `--voice` (any Kokoro voice id,
e.g. `af_heart`, `am_fenrir`, `am_michael`) and the overall rate with `speed=`.
The default is `am_liam` — `am_michael` reads deeper and slower.

## The numbers

All figures come from the paper; `assayloop_anim/data.py` holds Table 1 and the
Figure 4 panels verbatim.

The recall curves in `assayloop_anim/curves.json` are **measured**, exported
from the real test-set runs by
`src/bridgeloop/scripts/export_recovery_curves.py` and re-keyed from that
file's older method names to the paper's final ones. Round-10 values reproduce
Table 1's Frac.-hits column exactly, which is what pins the mapping:

| Curve | CSV `method` | round 10 | Table 1 FH |
|---|---|---|---|
| AssayLoop | `Gemini-3.1-Pro - AssayLoop Handoff` | 0.2770 | 27.7% |
| AssayFormer | `Transformer + GRPO (= AssayLoop)` | 0.2316 | 23.2% |
| Gemini 3.1 Pro | `Gemini-3.1-pro` | 0.2465 | 24.6% |
| BPMF | `BPMF` | 0.1966 | 19.7% |
| ICBR-EF | `ICBR-EF` | 0.1463 | 14.5% |

Note the CSV's `Transformer + GRPO (= AssayLoop)` is today's **AssayFormer**,
not AssayLoop. The mapping is confirmed verbatim by the paper's own figure
script, `release/assayloop/paper_files/make_assayloop_figures_v3.py`, whose
`METHODS` dict uses the same five keys.

**Random is not a measured series, and is not fitted either.** No Random curve
is exported, because none is needed: a random policy recovers hits in
proportion to the fraction of the library it has queried, so its recall is
linear in budget spent and exact by construction. The paper draws this as the
identity line `y = x` labelled *Random expectation*; `data.random_expectation()`
is the same line parameterized by round. `data.is_measured()` distinguishes it
from the five measured curves.

Two deliberate departures from the paper's Figure 4A, both for legibility in a
short video:

- **x axis is a nominal fraction of the library**, computed as `round × 100 /
  20,000`, not the paper's per-method `(n_in_library + n_out_of_universe) /
  library_size`. The real x-grid differs per method (0.043–0.052 at round 10)
  and would need interpolation or ragged line ends; the nominal axis keeps the
  rounds 1–3 overlap landing exactly on top of itself, which is the point of the
  handoff scene. Round numbering fades in underneath where `k` is discussed.
- **Colours are semantic, not the paper's.** Yellow means *LLM prior* and blue
  means *learned policy* consistently across all six scenes; the paper uses gold
  for AssayFormer and rust for Gemini. Green for AssayLoop matches both.

The handoff visible in scene 5 is real, not a stylization: AssayLoop and Gemini
3.1 Pro are *identical* for rounds 1–3 in the measured data and diverge from
round 4, which is exactly the k=3 warm start.

## Editing

- **Script and pacing** — `assayloop_anim/narration.py`. Re-run `render.py`
  without `--keep-vo` to re-synthesize and re-time.
- **Visuals** — `assayloop_anim/scenes.py`, one function per scene, each a pure
  function of absolute time so any frame renders standalone.
- **Palette and type** — `assayloop_anim/style.py`.
- **Voice** — any [Kokoro voice](https://huggingface.co/hexgrad/Kokoro-82M) id.
  Currently `am_liam`; see the Voice section above.
