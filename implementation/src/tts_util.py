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
import shutil
import subprocess
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
_DIST_DIR = _TOOLS_DIR / "espeak-ng-dist"
_ESPEAK_BIN = _DIST_DIR / "bin" / "espeak-ng"
_ESPEAK_LIB = _DIST_DIR / "lib"
_ESPEAK_DATA = _DIST_DIR / "espeak-ng-data"
# Legacy single-binary layout (Debian/Ubuntu dpkg -x extraction), still
# supported so this module works unmodified on either host this project has
# run on -- see module docstring. NOTE: this binary is committed to git as
# built on the original Ubuntu dev box, dynamically linked against
# libespeak-ng.so.1 with no rpath -- it only runs on a host that already has
# libespeak-ng1 + espeak-ng-data installed system-wide, which is *not*
# guaranteed (the module docstring above notes neither reference environment
# had it preinstalled). It is tried last, after the self-contained dist
# bundle and after a real system install, precisely because it's the most
# likely candidate to be present-but-broken on a fresh clone.
_ESPEAK_BIN_LEGACY = _TOOLS_DIR / "espeak-ng"


def _candidates() -> list[tuple[Path, dict[str, str]]]:
    """Ordered (binary, extra_env) candidates, most to least likely to work."""
    base_env = os.environ.copy()
    candidates: list[tuple[Path, dict[str, str]]] = []
    if _ESPEAK_BIN.exists():
        dist_env = dict(base_env)
        dist_env["LD_LIBRARY_PATH"] = f"{_ESPEAK_LIB}:{base_env.get('LD_LIBRARY_PATH', '')}"
        dist_env["ESPEAK_DATA_PATH"] = str(_ESPEAK_DATA)
        candidates.append((_ESPEAK_BIN, dist_env))
    system_bin = shutil.which("espeak-ng")
    if system_bin:
        candidates.append((Path(system_bin), base_env))
    if _ESPEAK_BIN_LEGACY.exists():
        candidates.append((_ESPEAK_BIN_LEGACY, base_env))
    return candidates


def synthesize(text: str, out_wav_path: str | Path, voice: str = "en", speed_wpm: int = 160) -> Path:
    """Synthesize `text` to a 22.05kHz mono WAV file at out_wav_path using espeak-ng.

    Tries, in order: the self-contained tools/espeak-ng-dist/ bundle, a
    system-installed `espeak-ng` on PATH, then the legacy committed binary
    (see _ESPEAK_BIN_LEGACY's comment for why it's last and least reliable).
    Falls through to the next candidate if one exists but fails to run (e.g.
    missing shared library), rather than surfacing a cryptic subprocess error
    from a candidate that was never going to work on this host.

    Raises FileNotFoundError if no candidate is available or all of them
    fail to run.
    """
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = _candidates()
    if not candidates:
        raise FileNotFoundError(
            f"No espeak-ng binary found at {_ESPEAK_BIN}, on PATH, or at "
            f"{_ESPEAK_BIN_LEGACY}. See src/tts_util.py docstring for how it was "
            "obtained in the reference environments (no root required); a plain "
            "`apt-get install espeak-ng` / `pacman -S espeak-ng` also works if you "
            "have sudo."
        )

    errors = []
    for bin_path, env in candidates:
        cmd = [str(bin_path), "-v", voice, "-s", str(speed_wpm), "-w", str(out_wav_path), text]
        try:
            subprocess.run(cmd, check=True, capture_output=True, env=env)
            return out_wav_path
        except (subprocess.CalledProcessError, OSError) as exc:
            detail = exc.stderr.decode(errors="replace") if getattr(exc, "stderr", None) else str(exc)
            errors.append(f"{bin_path}: {detail.strip()}")

    raise FileNotFoundError(
        "Every espeak-ng candidate failed to run:\n"
        + "\n".join(f"  - {e}" for e in errors)
        + "\nSee src/tts_util.py docstring for setup options."
    )


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "The paddy crop has yellow stem borer infestation"
    out = synthesize(text, "/tmp/tts_util_test.wav")
    print(f"wrote {out}")
