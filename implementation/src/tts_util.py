"""Query-audio synthesis via espeak-ng.

HONESTY NOTE (read before using this module for anything beyond smoke-testing):
There is no real target-dialect farmer speech corpus available in this
environment (the paper itself says this data does not yet exist -- see
agents/components/speech-native-retrieval.md). To exercise the ASR cascade and
the speech-native retrieval adapter end-to-end at all, we synthesize query audio
with espeak-ng reading the query TEXT (including the romanized Hindi/Bengali
colloquial queries in data/eval_queries.jsonl) using its English voice.

This does NOT produce authentic dialectal pronunciation -- espeak-ng's English
voice applies English letter-to-sound rules to romanized Hindi/Bengali text, so
the resulting audio sounds like an English speaker reading transliterated words
aloud, not a native speaker of the target dialect. It exists purely so every
downstream component (Whisper transcription, wav2vec2 encoding, the trained
adapter) has a real waveform to consume, so the MECHANISM can be run and
measured end-to-end. Do not read any latency/accuracy number produced from this
audio as evidence about real dialectal speech performance -- that would require
the real corpus this project does not have, exactly as flagged in the paper's
own Section V/VI.

espeak-ng itself: the `espeak-ng` binary was not preinstalled system-wide in
this environment (only its runtime data/library packages were). Since
passwordless sudo was not available to `apt install` it, the .deb was fetched
with `apt-get download` (no root required for download) and the single binary
extracted with `dpkg -x` into implementation/tools/espeak-ng -- no source was
compiled, no system files were modified. It links against libespeak-ng.so.1 and
the espeak-ng-data files, both already present system-wide in this environment
via the `libespeak-ng1` / `espeak-ng-data` apt packages.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
_ESPEAK_BIN = _TOOLS_DIR / "espeak-ng"


def synthesize(text: str, out_wav_path: str | Path, voice: str = "en", speed_wpm: int = 160) -> Path:
    """Synthesize `text` to a 22.05kHz mono WAV file at out_wav_path using espeak-ng.

    Raises FileNotFoundError if the bundled espeak-ng binary is missing, and
    subprocess.CalledProcessError if synthesis fails for some other reason
    (e.g. missing espeak-ng-data on a machine other than the one this was built
    on -- see the module docstring for how that data was obtained here).
    """
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    if not _ESPEAK_BIN.exists():
        raise FileNotFoundError(
            f"espeak-ng binary not found at {_ESPEAK_BIN}. See src/tts_util.py docstring "
            "for how it was obtained in the reference environment (apt-get download + dpkg -x, "
            "no root required); a plain `apt-get install espeak-ng` also works if you have sudo."
        )
    cmd = [str(_ESPEAK_BIN), "-v", voice, "-s", str(speed_wpm), "-w", str(out_wav_path), text]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav_path


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "The paddy crop has yellow stem borer infestation"
    out = synthesize(text, "/tmp/tts_util_test.wav")
    print(f"wrote {out}")
