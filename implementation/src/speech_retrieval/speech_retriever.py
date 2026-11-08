"""Speech-native retrieval -- Section III-B / RQ1, the "no ASR in the loop" path.

query audio -> frozen wav2vec2 encoder -> trained adapter -> 384-dim embedding in
the SAME space as the text dense retriever -> cosine similarity search over KB
passage embeddings. No transcription step exists anywhere in this path.

See adapter.py and train_adapter.py for what the adapter actually is and the
honesty notes about what its training data can and cannot demonstrate. This
module is the "does it actually retrieve using what was trained" wiring.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.retrieval.dense_retriever import DenseRetriever, RetrievedPassage
from src.speech_retrieval.adapter import FrozenSpeechEncoder, SpeechToRetrievalAdapter


class SpeechNativeRetriever:
    def __init__(self, dense_retriever: DenseRetriever, adapter_path: str | Path,
                 lexicon_entries: list[dict] | None = None, device: str | None = None):
        self.dense = dense_retriever
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.speech_encoder = FrozenSpeechEncoder(device=self.device)
        self.adapter = SpeechToRetrievalAdapter().to(self.device)
        state = torch.load(str(adapter_path), map_location=self.device)
        self.adapter.load_state_dict(state)
        self.adapter.eval()
        
        # Build embedding-space dialect anchors
        self.anchor_embeddings: np.ndarray | None = None
        self.anchor_entities: list[str] = []
        if lexicon_entries:
            self._init_dialect_anchors(lexicon_entries)

    def _init_dialect_anchors(self, entries: list[dict]):
        """Pre-compute normalized canonical entity embeddings for dialect mapping in vector space."""
        anchors = []
        entities = []
        for e in entries:
            for ent in e.get("maps_to", []):
                if ent not in entities:
                    entities.append(ent)
        if entities:
            self.anchor_entities = entities
            self.anchor_embeddings = self.dense.model.encode(
                entities, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )

    @torch.no_grad()
    def encode_audio(self, waveform_16k: np.ndarray, blend_threshold: float = 0.55, blend_weight: float = 0.3) -> np.ndarray:
        pooled = self.speech_encoder.encode(waveform_16k).to(self.device)  # (1, 768)
        embedding = self.adapter(pooled)  # (1, 384), already L2-normalized
        emb_np = embedding.squeeze(0).cpu().numpy()

        # Embedding-space dialect mapping (Architecture #2):
        # If speech vector aligns closely with a canonical dialect entity anchor, blend it
        if self.anchor_embeddings is not None and len(self.anchor_entities) > 0:
            sims = np.dot(self.anchor_embeddings, emb_np)
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= blend_threshold:
                anchor_vec = self.anchor_embeddings[best_idx]
                blended = (1.0 - blend_weight) * emb_np + blend_weight * anchor_vec
                norm = np.linalg.norm(blended)
                if norm > 1e-6:
                    emb_np = blended / norm

        return emb_np

    def search(self, waveform_16k: np.ndarray, top_k: int = 5) -> list[RetrievedPassage]:
        embedding = self.encode_audio(waveform_16k)
        return self.dense.search_by_embedding(embedding, top_k=top_k)


if __name__ == "__main__":
    import sys

    import librosa

    REPO_ROOT = Path(__file__).resolve().parents[2]
    kb_path = REPO_ROOT / "data" / "kb.json"
    adapter_path = REPO_ROOT / "models" / "speech_adapter.pt"

    dense = DenseRetriever.from_kb_file(kb_path)
    retriever = SpeechNativeRetriever(dense, adapter_path)

    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    if audio_path is None:
        print("Usage: python -m src.speech_retrieval.speech_retriever <audio_path.wav>")
        sys.exit(1)
    wav, _sr = librosa.load(audio_path, sr=16000, mono=True)
    for r in retriever.search(wav.astype("float32"), top_k=3):
        print(f"{r.score:6.4f}  {r.doc_id}  {r.passage['title']}")
