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

espeak-ng itself was not preinstalled system-wide in either environment this
project has been built in, and passwordless sudo was not available to install
it system-wide in either. Two different no-root extraction methods were used
depending on the host distro, both producing a self-contained binary+library+
data tree under implementation/tools/ (no source compiled, no system files
touched):
  - Debian/Ubuntu host: `apt-get download` (no root required to download) the
    espeak-ng .deb, extracted with `dpkg -x`.
  - Arch Linux host: `pacman -Sp` (prints the mirror URL without installing)
    for espeak-ng and its two link-time deps (pcaudiolib, libsonic), fetched
    directly with `curl` and unpacked with `tar --zstd -x` into
    implementation/tools/espeak-ng-dist/{bin,lib,espeak-ng-data}/. The binary
    is run with LD_LIBRARY_PATH and ESPEAK_DATA_PATH pointed at that tree
    (see below) so it needs no system-wide library or data files at all.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
_DIST_DIR = _TOOLS_DIR / "espeak-ng-dist"
_ESPEAK_BIN = _DIST_DIR / "bin" / "espeak-ng"
_ESPEAK_LIB = _DIST_DIR / "lib"
_ESPEAK_DATA = _DIST_DIR / "espeak-ng-data"
# Legacy single-binary layout (Debian/Ubuntu dpkg -x extraction), still
# supported so this module works unmodified on either host this project has
# run on -- see module docstring.
_ESPEAK_BIN_LEGACY = _TOOLS_DIR / "espeak-ng"


def synthesize(text: str, out_wav_path: str | Path, voice: str = "en", speed_wpm: int = 160) -> Path:
    """Synthesize `text` to a 22.05kHz mono WAV file at out_wav_path using espeak-ng.

    Raises FileNotFoundError if no bundled espeak-ng binary is found, and
    subprocess.CalledProcessError if synthesis fails for some other reason.
    """
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if _ESPEAK_BIN.exists():
        bin_path = _ESPEAK_BIN
        env["LD_LIBRARY_PATH"] = f"{_ESPEAK_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
        env["ESPEAK_DATA_PATH"] = str(_ESPEAK_DATA)
    elif _ESPEAK_BIN_LEGACY.exists():
        bin_path = _ESPEAK_BIN_LEGACY
    else:
        raise FileNotFoundError(
            f"No espeak-ng binary found at {_ESPEAK_BIN} or {_ESPEAK_BIN_LEGACY}. "
            "See src/tts_util.py docstring for how it was obtained in the reference "
            "environments (no root required); a plain `apt-get install espeak-ng` / "
            "`pacman -S espeak-ng` also works if you have sudo."
        )
    cmd = [str(bin_path), "-v", voice, "-s", str(speed_wpm), "-w", str(out_wav_path), text]
    subprocess.run(cmd, check=True, capture_output=True, env=env)
    return out_wav_path


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "The paddy crop has yellow stem borer infestation"
    out = synthesize(text, "/tmp/tts_util_test.wav")
    print(f"wrote {out}")
