"""Voiceover script, and the timeline it implies.

The visuals are timed *from the audio*, not the other way round: we synthesize
each beat, measure how long it actually came out, and hand the resulting
schedule to the scene code. That way the animation cannot drift out of sync
when a line is reworded.

Numbers are spelled out in `say` (piper reads words more reliably than
numerals) while `caption` keeps the typographic form for the screen.
"""

from __future__ import annotations

import json
import os
import wave
from dataclasses import dataclass, asdict

from . import tts


@dataclass
class Beat:
    scene: str
    say: str
    caption: str = ""      # on-screen text; "" means the scene draws its own
    pad: float = 0.35      # silence appended after this beat, seconds


SCRIPT: list[Beat] = [
    # --- 1. the library ----------------------------------------------------
    Beat("intro",
         # "genome-wide" phonemises as one crammed 4-syllable token that the
         # model slurs; the override splits it into two clean stressed words.
         # Dropping "in the genome" also removes a redundancy with it.
         "A [genome-wide](/ʤˈinOm wˈId/) CRISPR screen puts a question to every gene: "
         "is this gene a hit? || There are twenty thousand candidate genes.",
         "~20,000 candidate genes", pad=1.0),

    # --- 2. what a screen actually is --------------------------------------
    Beat("screen",
         "But what is a CRISPR screen? || "
         "A screen tests which genes control a specific cellular behaviour or trait.",
         "what a screen measures"),
    Beat("screen",
         "You set up an experiment. | Say, neuroblastoma cells, | and whether they "
         "resist iron-mediated cell death. || "
         "Then you switch one gene off, or on, | and see whether the outcome changed.",
         "perturb one gene, re-run the assay"),
    Beat("screen",
         "Do that across the whole genome, | and you learn which genes control "
         "the behaviour you care about.",
         "repeat across the genome", pad=0.5),

    # --- 3. hits ------------------------------------------------------------
    Beat("hits",
         "A few of those genes are hits. | A hit is a gene where switching it off, or on, "
         "actually produces the effect you are looking for. || "
         "In a typical screen, only one in a hundred are hits.",
         "a hit: perturbing it produces the phenotype"),
    Beat("hits",
         "You do not know which ones they are. || That is the whole point of running the screen.",
         "~1% are hits  ·  you cannot see which", pad=0.5),

    # --- 4. the budget ------------------------------------------------------
    Beat("budget",
         "But testing all of them is often out of reach. | "
         "The readout is too slow, or too expensive. || "
         "So you get a budget: | one thousand genes, | five percent of the library.",
         "budget: 1,000 genes  ·  5%"),
    Beat("budget",
         "Which thousand should you pick?",
         "which 1,000?", pad=0.65),

    # --- 2. the loop -------------------------------------------------------
    Beat("loop",
         "You don't have to choose all at once. || Spend the budget in rounds. | "
         "One hundred genes, ten times.",
         "10 rounds  ×  100 genes"),
    Beat("loop",
         "After each round you learn which of those hundred were hits, | "
         "and that answer can change what you pick next. || That is the loop.",
         "propose  →  observe  →  adapt"),
    Beat("loop",
         "This is AssayBench-Loop: | one thousand three hundred and eighty-nine "
         "real CRISPR screens, split by time. || "
         "Train on the past, | test on twenty screens from the future.",
         "1,349 train   ·   20 test", pad=0.5),

    # --- 3. how we measure a strategy --------------------------------------
    Beat("curve",
         "We can picture any strategy with a recovery curve: || "
         "the fraction of all hits you have found, | "
         "against the fraction of the library you have sampled.",
         "recovery curve"),
    Beat("curve",
         "Pick genes at random, and you find hits at exactly the rate you sample them. || "
         "A straight line. | That is the thing to beat.",
         "random expectation", pad=0.5),

    # --- 4. the LLM prior --------------------------------------------------
    Beat("llm",
         "One way to fill that first round is to ask a language model. || "
         "It has read the literature, | so it already has an opinion about which "
         "genes are plausible, | before you measure anything.",
         "knowledge from papers, not from this assay", pad=0.95),
    Beat("llm",
         "And it works. || Gemini three point one Pro finds hits at nearly five times "
         "the random rate.",
         "Gemini 3.1 Pro   ·   EF 4.71"),
    Beat("llm",
         "But watch what happens as results come back. || The curve flattens. || "
         "The prior is strong. | The learning from feedback is weak.",
         "strong prior, weak adaptation", pad=0.5),

    # --- 5. AssayFormer ----------------------------------------------------
    Beat("former",
         "AssayFormer is the opposite animal. || A transformer trained across all "
         "thirteen hundred historical screens. || It takes the history as input: | "
         "every gene tested so far, and whether it was a hit.",
         "history-conditioned policy"),
    Beat("former",
         "It starts colder. || Before the first batch, it knows little about "
         "this particular screen.",
         "AssayFormer   ·   EF 4.83"),
    Beat("former",
         "But every round of feedback sharpens it, | and it keeps climbing.",
         "weak prior, strong adaptation", pad=0.5),

    # --- 6. the handoff ----------------------------------------------------
    Beat("handoff",
         "LLMs have a strong prior but weak adaptation. || "
         "AssayFormer has a weak prior but strong adaptation. || "
         "So use each one where it is best.",
         ""),
    Beat("handoff",
         "Let the language model drive the first three rounds. || "
         "Then hand the accumulated history to AssayFormer, | and let it run.",
         "handoff at k = 3"),
    Beat("handoff",
         "That is AssayLoop.",
         "AssayLoop", pad=0.65),

    # --- 7. the result -----------------------------------------------------
    Beat("result",
         "Five point seven times better than random. || "
         "Twenty-seven point seven percent of hits in the screen are found "
         "by sampling only five percent of the library.",
         ""),
    Beat("result",
         "And because the policy is learned from past experiments, | "
         "it improves as those accumulate, || with no sign yet of saturating.",
         "more screens  →  better policy"),
    Beat("result",
         "Biology in the loop. || Try it on our website today!",
         "", pad=1.5),
]


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def synthesize(outdir: str, voice: str = tts.DEFAULT_VOICE,
               speed: float = 1.0, quiet: bool = True) -> list[dict]:
    """Render every beat to its own wav; return the schedule with real durations.

    Pauses come from the `|` / `||` markers and punctuation in each line (see
    `tts.split_clauses`), not from whatever the model feels like doing.
    """
    os.makedirs(outdir, exist_ok=True)
    schedule, t = [], 0.0
    for i, beat in enumerate(SCRIPT):
        path = os.path.join(outdir, f"beat_{i:02d}.wav")
        dur = tts.write_wav(path, tts.synthesize_line(beat.say, voice=voice, speed=speed))
        schedule.append({**asdict(beat), "index": i, "wav": path,
                         "start": t, "dur": dur, "end": t + dur})
        t += dur + beat.pad
        if not quiet:
            print(f"  {i:02d} {beat.scene:9s} {dur:5.2f}s  {beat.say[:54]}")
    return schedule


def scene_spans(schedule: list[dict]) -> dict[str, tuple[float, float]]:
    """First-start / last-end for each scene key, in order of appearance."""
    spans: dict[str, tuple[float, float]] = {}
    for b in schedule:
        s, e = b["start"], b["end"] + b["pad"]
        if b["scene"] in spans:
            spans[b["scene"]] = (spans[b["scene"]][0], e)
        else:
            spans[b["scene"]] = (s, e)
    return spans


def save(schedule: list[dict], path: str) -> None:
    with open(path, "w") as fh:
        json.dump(schedule, fh, indent=2)


def load(path: str) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)
