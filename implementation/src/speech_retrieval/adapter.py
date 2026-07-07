"""Speech-to-retrieval adapter -- Section III-B, "speech-native retrieval".

A small nn.Module that projects a frozen wav2vec2 speech-encoder representation
of a query utterance into the SAME 384-dim embedding space used by the frozen
text dense retriever (sentence-transformers/all-MiniLM-L6-v2, see
retrieval/dense_retriever.py). This mirrors the SpeechRAG/S2R adapter pattern
described in agents/components/speech-native-retrieval.md: the speech encoder and
the text retriever both stay frozen; only this small adapter is trained.

Architecture: mean-pool wav2vec2's last_hidden_state over time (768-dim) -> a
2-layer MLP with a GELU nonlinearity and LayerNorm -> L2-normalize to 384-dim.
This is intentionally small (a few hundred thousand parameters) since the
"training set" available here is a synthesized toy set (see train_adapter.py) --
a large adapter would just overfit it harder, not learn anything more real.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

WAV2VEC2_HIDDEN_SIZE = 768
TEXT_EMBEDDING_DIM = 384


class SpeechToRetrievalAdapter(nn.Module):
    def __init__(self, in_dim: int = WAV2VEC2_HIDDEN_SIZE, out_dim: int = TEXT_EMBEDDING_DIM, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pooled_speech_features: torch.Tensor) -> torch.Tensor:
        """pooled_speech_features: (batch, in_dim) mean-pooled wav2vec2 hidden states.

        Returns L2-normalized (batch, out_dim) embeddings directly comparable
        (cosine similarity) with DenseRetriever's passage/query embeddings.
        """
        projected = self.net(pooled_speech_features)
        return F.normalize(projected, dim=-1)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu", **kwargs) -> "SpeechToRetrievalAdapter":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location=map_location))
        model.eval()
        return model


class FrozenSpeechEncoder:
    """Wraps facebook/wav2vec2-base to produce mean-pooled utterance features.

    Kept separate from the adapter module (which stays a plain nn.Module you can
    train) since this encoder is always frozen/eval-mode and does audio I/O.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: str | None = None):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, waveform_16k, sampling_rate: int = 16000) -> torch.Tensor:
        """waveform_16k: 1-D numpy array or list of them (batch). Returns (batch, 768)."""
        is_batch = isinstance(waveform_16k, list)
        inputs = self.feature_extractor(
            waveform_16k if is_batch else [waveform_16k],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        outputs = self.model(input_values=input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, T, 768)

        if attention_mask is not None:
            # wav2vec2's internal downsampling means the feature-time-axis mask needs
            # the model's own conv output-length helper, not the raw sample-level mask.
            feat_mask = self.model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
            feat_mask = feat_mask.unsqueeze(-1).float()
            pooled = (hidden * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = hidden.mean(dim=1)
        return pooled.cpu()
