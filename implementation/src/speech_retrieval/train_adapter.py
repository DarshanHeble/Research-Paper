"""Toy adapter training script -- proof-of-concept for speech-native retrieval.

HONESTY NOTE (read this before trusting any number this script prints):
There is no real paired dialect-speech/passage dataset for this project -- the
paper is explicit that the target dialect speech corpus does not exist yet (see
agents/components/speech-native-retrieval.md, "Because dialect-specific speech
corpora barely exist..."). This script does NOT claim to fix that. Instead it:

  1. Auto-generates a handful of short templated English query sentences per
     KB passage (e.g. "What can I do about pink bollworm in Cotton?") -- NOT
     the eval_queries.jsonl sentences, so the eval set stays genuinely held out.
  2. Synthesizes each with espeak-ng's English voice (src/tts_util.py) -- so the
     "speech" here is synthetic TTS audio of standard English phrasing, not real
     recorded farmer speech in any dialect.
  3. Extracts frozen wav2vec2-base features for that audio and trains the small
     adapter (adapter.py) with an in-batch InfoNCE contrastive loss to align
     each utterance's pooled speech features with the frozen dense retriever's
     text embedding of the SAME passage it was generated from.

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
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.retrieval.dense_retriever import DenseRetriever
from src.speech_retrieval.adapter import FrozenSpeechEncoder, SpeechToRetrievalAdapter
from src.tts_util import synthesize

REPO_ROOT = Path(__file__).resolve().parents[2]
KB_PATH = REPO_ROOT / "data" / "kb.json"
AUDIO_DIR = REPO_ROOT / "data" / "synthetic_audio" / "train"
ADAPTER_OUT = REPO_ROOT / "models" / "speech_adapter.pt"

TEMPLATES = [
    "What can I do about {entity} in {crop}?",
    "My {crop} crop has a problem with {entity_lower}.",
    "Please tell me how to manage {entity} affecting {crop}.",
    "I think my field has {entity}, what should I do?",
]


def load_waveform_16k(path: Path) -> np.ndarray:
    import librosa

    wav, _sr = librosa.load(str(path), sr=16000, mono=True)
    return wav.astype(np.float32)


def build_training_examples(passages: list[dict]) -> list[dict]:
    examples = []
    rng = random.Random(42)
    for p in passages:
        entity = p["entities"][0] if p.get("entities") else p["title"]
        crop = p.get("crop", "the field")
        chosen_templates = rng.sample(TEMPLATES, k=min(3, len(TEMPLATES)))
        for t_idx, template in enumerate(chosen_templates):
            text = template.format(entity=entity, entity_lower=entity.lower(), crop=crop)
            examples.append({"passage_id": p["id"], "text": text, "template_idx": t_idx})
    return examples


def main(epochs: int = 100, batch_size: int = 16, lr: float = 1e-3, temperature: float = 0.07,
         val_fraction: float = 0.15, seed: int = 42):
    t_start = time.time()
    random.seed(seed)
    torch.manual_seed(seed)

    with open(KB_PATH, encoding="utf-8") as f:
        passages = json.load(f)
    passage_index = {p["id"]: i for i, p in enumerate(passages)}

    print(f"[1/5] Building auto-generated (non-eval-set) training query templates for {len(passages)} passages...")
    examples = build_training_examples(passages)
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_fraction))
    val_examples, train_examples = examples[:n_val], examples[n_val:]
    print(f"    {len(train_examples)} train examples, {len(val_examples)} val examples "
          f"(disjoint from data/eval_queries.jsonl, which this script never reads)")

    print("[2/5] Synthesizing TTS audio for each training/val example with espeak-ng (English voice)...")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for i, ex in enumerate(examples):
        wav_path = AUDIO_DIR / f"ex{i:04d}.wav"
        if not wav_path.exists():
            synthesize(ex["text"], wav_path, voice="en")
        ex["wav_path"] = wav_path

    print("[3/5] Extracting frozen wav2vec2-base features + frozen text-retriever targets...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speech_encoder = FrozenSpeechEncoder(device=device)
    dense = DenseRetriever.from_kb_file(KB_PATH, device=device)
    # Frozen target embedding per passage, computed once via the same text encoder
    # used at retrieval time -- this IS "the frozen text retriever's embedding space".
    target_embeddings = torch.tensor(dense.doc_embeddings, dtype=torch.float32)  # (n_passages, 384)

    def featurize(exs: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        feats = []
        targets = []
        for ex in exs:
            wav = load_waveform_16k(ex["wav_path"])
            feat = speech_encoder.encode(wav)  # (1, 768)
            feats.append(feat.squeeze(0))
            targets.append(target_embeddings[passage_index[ex["passage_id"]]])
        return torch.stack(feats), torch.stack(targets)

    train_feats, train_targets = featurize(train_examples)
    val_feats, val_targets = featurize(val_examples)
    print(f"    train_feats {tuple(train_feats.shape)}  val_feats {tuple(val_feats.shape)}")

    print(f"[4/5] Training adapter for {epochs} epochs with in-batch InfoNCE contrastive loss...")
    adapter = SpeechToRetrievalAdapter().to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
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
            loss = F.cross_entropy(logits, labels)
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
    # happened to land on -- with only 18 held-out examples, per-epoch val
    # accuracy is noisy (each example is worth ~5.6 percentage points).
    adapter.load_state_dict(best_state)
    adapter.eval()
    ADAPTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    adapter.save(str(ADAPTER_OUT))
    print(f"[5/5] Best checkpoint: epoch {best_epoch}, val_top1_passage_acc={best_val_acc:.3f} "
          f"(chance level = 1/{len(passages)} = {1/len(passages):.3f}). Saved to {ADAPTER_OUT}")
    print(f"Total wall time: {time.time() - t_start:.1f}s")
    return {"history": history, "n_train": len(train_examples), "n_val": len(val_examples),
            "best_val_acc": best_val_acc, "best_epoch": best_epoch}


if __name__ == "__main__":
    main()
