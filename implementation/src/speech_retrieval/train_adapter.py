"""Toy adapter training script -- proof-of-concept for speech-native retrieval.

HONESTY NOTE (read this before trusting any number this script prints):
There is no real paired dialect-speech/passage dataset for this project -- the
paper is explicit that the target dialect speech corpus does not exist yet (see
agents/components/speech-native-retrieval.md, "Because dialect-specific speech
corpora barely exist..."). This script does NOT claim to fix that. Instead it:

  1. Builds training sentences from three sources, none of which read
     eval_queries.jsonl (the eval set stays genuinely held out):
       a. every sentence of every KB passage (symptom/management vocabulary);
       b. short English templates over EVERY entity alias of each passage;
       c. short romanized-Hindi templates over every dialect_lexicon.json term
          (e.g. "Mere Cotton me gulabi sundi lag gayi hai"). The eval set reuses
          the same lexicon terms by design -- exactly as the text-mode mapper
          uses the lexicon at test time -- but never the same sentences.
  2. Synthesizes each sentence with several espeak-ng English voices and speeds
     (src/tts_util.py) as cheap acoustic augmentation -- still synthetic TTS,
     not real recorded farmer speech in any dialect.
  3. Extracts frozen wav2vec2-base features and trains the small adapter
     (adapter.py) to reproduce the frozen text retriever's embedding of the
     utterance's OWN text (speech->text-embedding distillation, as in
     SpeechRAG), with an in-batch InfoNCE + cosine loss.

Embedding-space dialect mapping ("architecture #2-lite"): for a dialect
sentence (source c), the distillation target is the text embedding of the
DialectMapper-NORMALIZED sentence, not the raw one. The adapter thereby learns
to emit, straight from audio, the embedding the text path would have produced
after lexicon normalization -- so the speech-native path gets dialect mapping
with no transcript at inference time. The lexicon is used only at training
time.

This demonstrates the MECHANISM -- that a small adapter can learn to project
speech-encoder output into a frozen text-retriever's embedding space well enough
to retrieve the right passage without transcription -- end to end, on this
machine, with real numbers. It is NOT evidence that this generalizes to real
dialect speech, accented pronunciation, or vocabulary the templates didn't cover.
Treat every number train_adapter.py or the resulting speech-native retriever
reports as demo-scale validation of the architecture, not a research result.
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.dialect_mapping.mapper import DialectMapper
from src.retrieval.dense_retriever import DenseRetriever
from src.speech_retrieval.adapter import FrozenSpeechEncoder, SpeechToRetrievalAdapter
from src.tts_util import synthesize

REPO_ROOT = Path(__file__).resolve().parents[2]
KB_PATH = REPO_ROOT / "data" / "kb.json"
LEXICON_PATH = REPO_ROOT / "data" / "dialect_lexicon.json"
AUDIO_DIR = REPO_ROOT / "data" / "synthetic_audio" / "train"
ADAPTER_OUT = REPO_ROOT / "models" / "speech_adapter.pt"

TEMPLATES = [
    "What can I do about {entity} in {crop}?",
    "My {crop} crop has a problem with {entity}.",
    "Please tell me how to manage {entity} affecting {crop}.",
    "I think my field has {entity}, what should I do?",
]
DIALECT_TEMPLATES = [
    "Mere {crop} me {term} lag gaya hai.",
    "{term} ka kya ilaj hai?",
    "Khet me {term} dikh raha hai, kya karu?",
]
# (voice, words-per-minute) pairs -- cheap acoustic augmentation.
VOICES = [("en", 160), ("en-us", 140), ("en-gb-x-rp", 180)]


def load_waveform_16k(path: Path) -> np.ndarray:
    import librosa

    wav, _sr = librosa.load(str(path), sr=16000, mono=True)
    return wav.astype(np.float32)


def build_training_sentences(passages: list[dict], lexicon: list[dict], normalize_dialect: bool = True) -> list[dict]:
    """Returns [{"passage_id", "text" (spoken), "target_text" (embedded)}]."""
    mapper = DialectMapper(lexicon)
    entity_to_pids: dict[str, list[str]] = {}
    out = []
    for p in passages:
        crop = p.get("crop", "the field")
        for sent in re.split(r"(?<=[.!?])\s+", p["text"]):
            if len(sent.split()) >= 4:
                out.append({"passage_id": p["id"], "text": sent, "target_text": sent})
        for i, ent in enumerate(p.get("entities") or [p["title"]]):
            entity_to_pids.setdefault(ent.lower(), []).append(p["id"])
            t = TEMPLATES[i % len(TEMPLATES)].format(entity=ent, crop=crop)
            out.append({"passage_id": p["id"], "text": t, "target_text": t})
    for e in lexicon:
        for ent in e["maps_to"]:
            for pid in entity_to_pids.get(ent.lower(), [])[:1]:
                crop = next(p.get("crop", "") for p in passages if p["id"] == pid)
                for template in DIALECT_TEMPLATES:
                    t = template.format(term=e["term"], crop=crop)
                    out.append({"passage_id": pid, "text": t, "target_text": mapper.map_query(t).normalized_query if normalize_dialect else t})
    return out


def main(epochs: int = 150, batch_size: int = 64, lr: float = 1e-3, temperature: float = 0.07,
         val_fraction: float = 0.15, seed: int = 42, normalize_dialect: bool = True,
         out_path: Path = ADAPTER_OUT):
    t_start = time.time()
    random.seed(seed)
    torch.manual_seed(seed)

    with open(KB_PATH, encoding="utf-8") as f:
        passages = json.load(f)
    passage_index = {p["id"]: i for i, p in enumerate(passages)}

    with open(LEXICON_PATH, encoding="utf-8") as f:
        lexicon = json.load(f)["entries"]

    print(f"[1/5] Building (non-eval-set) training sentences for {len(passages)} passages...")
    sentences = build_training_sentences(passages, lexicon, normalize_dialect)
    rng = random.Random(seed)
    rng.shuffle(sentences)
    # Split by SENTENCE, before voice augmentation, so no val sentence is heard in training.
    n_val = max(1, int(len(sentences) * val_fraction))
    val_examples = [dict(s, voice=v, wpm=w) for s in sentences[:n_val] for v, w in VOICES[:1]]
    train_examples = [dict(s, voice=v, wpm=w) for s in sentences[n_val:] for v, w in VOICES]
    examples = train_examples + val_examples
    print(f"    {len(train_examples)} train examples, {len(val_examples)} val examples "
          f"(disjoint from data/eval_queries.jsonl, which this script never reads)")

    print(f"[2/5] Synthesizing TTS audio for {len(examples)} examples with espeak-ng ({len(VOICES)} voice/speed variants)...")
    import hashlib

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for ex in examples:
        key = hashlib.sha1(f"{ex['voice']}|{ex['wpm']}|{ex['text']}".encode()).hexdigest()[:16]
        wav_path = AUDIO_DIR / f"{key}.wav"
        if not wav_path.exists():
            synthesize(ex["text"], wav_path, voice=ex["voice"], speed_wpm=ex["wpm"])
        ex["wav_path"] = wav_path

    print("[3/5] Extracting frozen wav2vec2-base features + frozen text-retriever targets...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speech_encoder = FrozenSpeechEncoder(device=device)
    dense = DenseRetriever.from_kb_file(KB_PATH, device=device)
    # Frozen target embedding per passage, computed once via the same text encoder
    # used at retrieval time -- this IS "the frozen text retriever's embedding space".
    target_embeddings = torch.tensor(dense.doc_embeddings, dtype=torch.float32)  # (n_passages, 384)

    def featurize(exs: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        feats = [speech_encoder.encode(load_waveform_16k(ex["wav_path"])).squeeze(0) for ex in exs]
        # Distillation target: the text encoder's embedding of what was said
        # (dialect-normalized for dialect sentences -- see module docstring).
        targets = torch.tensor(dense.model.encode([ex["target_text"] for ex in exs], convert_to_numpy=True,
                                                  normalize_embeddings=True, show_progress_bar=False))
        return torch.stack(feats), targets

    train_feats, train_targets = featurize(train_examples)
    val_feats, val_targets = featurize(val_examples)
    print(f"    train_feats {tuple(train_feats.shape)}  val_feats {tuple(val_feats.shape)}")

    print(f"[4/5] Training adapter for {epochs} epochs with in-batch InfoNCE contrastive loss...")
    adapter = SpeechToRetrievalAdapter().to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.05)
    train_feats_d = train_feats.to(device)
    train_targets_d = F.normalize(train_targets, dim=-1).to(device)
    val_feats_d = val_feats.to(device)
    val_targets_d = F.normalize(val_targets, dim=-1).to(device)

    n = train_feats_d.shape[0]
    val_labels = torch.tensor([passage_index[ex["passage_id"]] for ex in val_examples], device=device)
    all_target_norm = F.normalize(target_embeddings.to(device), dim=-1)

    def val_top1_acc() -> float:
        adapter.eval()
        with torch.no_grad():
            val_speech_emb = adapter(val_feats_d)
            val_sims = val_speech_emb @ all_target_norm.T
            return (val_sims.argmax(dim=1) == val_labels).float().mean().item()

    history = []
    best_val_acc = -1.0
    best_state = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        adapter.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if idx.numel() < 2:
                continue
            speech_emb = adapter(train_feats_d[idx])           # (b, 384)
            text_emb = train_targets_d[idx]                     # (b, 384)
            logits = speech_emb @ text_emb.T / temperature       # (b, b)
            labels = torch.arange(idx.numel(), device=device)
            # InfoNCE alone treats same-sentence voice variants in a batch as
            # negatives; the cosine term anchors each to its exact target.
            loss = F.cross_entropy(logits, labels) + (1 - (speech_emb * text_emb).sum(-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        history.append(epoch_loss / max(1, n_batches))

        val_acc = val_top1_acc()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  train_loss={history[-1]:.4f}  val_top1_passage_acc={val_acc:.3f}")

    # Keep the best-on-validation checkpoint rather than whatever the last epoch
    # happened to land on -- the val set is small, so per-epoch accuracy is noisy.
    adapter.load_state_dict(best_state)
    adapter.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adapter.save(str(out_path))
    print(f"[5/5] Best checkpoint: epoch {best_epoch}, val_top1_passage_acc={best_val_acc:.3f} "
          f"(chance level = 1/{len(passages)} = {1/len(passages):.3f}). Saved to {out_path}")
    print(f"Total wall time: {time.time() - t_start:.1f}s")
    return {"history": history, "n_train": len(train_examples), "n_val": len(val_examples),
            "best_val_acc": best_val_acc, "best_epoch": best_epoch}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dialect-normalization", action="store_true",
                    help="ablation: distill dialect sentences to their RAW text embedding")
    ap.add_argument("--out", type=Path, default=ADAPTER_OUT)
    a = ap.parse_args()
    main(seed=a.seed, normalize_dialect=not a.no_dialect_normalization, out_path=a.out)
