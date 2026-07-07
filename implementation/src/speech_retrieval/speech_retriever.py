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
    def __init__(self, dense_retriever: DenseRetriever, adapter_path: str | Path, device: str | None = None):
        self.dense = dense_retriever
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.speech_encoder = FrozenSpeechEncoder(device=self.device)
        self.adapter = SpeechToRetrievalAdapter().to(self.device)
        state = torch.load(str(adapter_path), map_location=self.device)
        self.adapter.load_state_dict(state)
        self.adapter.eval()

    @torch.no_grad()
    def encode_audio(self, waveform_16k: np.ndarray) -> np.ndarray:
        pooled = self.speech_encoder.encode(waveform_16k).to(self.device)  # (1, 768)
        embedding = self.adapter(pooled)  # (1, 384), already L2-normalized
        return embedding.squeeze(0).cpu().numpy()

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
