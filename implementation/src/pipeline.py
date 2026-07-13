"""End-to-end orchestration -- ties every component together into one pipeline.

query (text OR audio) -> [dialect mapping] -> [hybrid retrieval OR speech-native
retrieval] -> [confidence gate] -> templated/LLM-grounded answer, OR an
"ESCALATE TO HUMAN EXPERT" response.

Three query modes, corresponding to the three retrieval paths this prototype
implements (see src/retrieval, src/asr, src/speech_retrieval):

  * "text"          -- text query -> dialect mapping -> hybrid retrieval.
  * "asr_cascade"    -- audio -> Whisper transcription -> dialect mapping ->
                        hybrid retrieval (Section IV-C baseline #1; the thing
                        speech-native retrieval is compared against for RQ1).
  * "speech_native"  -- audio -> frozen wav2vec2 + trained adapter -> direct
                        embedding-space retrieval, no transcription step.
                        NOTE: dialect mapping is NOT applied in this mode -- it
                        is a text-lexicon lookup (src/dialect_mapping/mapper.py)
                        and there is no text in this path. A production system
                        wanting both speech-native retrieval AND dialect mapping
                        would need "architecture #2" from
                        agents/components/dialect-entity-mapping.md (embedding-
                        space dialect mapping), which this prototype does not
                        implement. This is a documented limitation, not a bug.

Generation is real local-LLM generation (Qwen2.5-0.5B-Instruct via
src/generation/llm_generator.py) when it can be loaded, and a deterministic
templated fallback otherwise -- the result always reports which one was used.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_PATH = REPO_ROOT / "data" / "kb.json"
DEFAULT_LEXICON_PATH = REPO_ROOT / "data" / "dialect_lexicon.json"
DEFAULT_ADAPTER_PATH = REPO_ROOT / "models" / "speech_adapter.pt"


@dataclass
class PipelineResult:
    mode: str
    original_query: str            # text query, or ASR transcript, or "<audio, speech-native>"
    normalized_query: str | None   # after dialect mapping, if applicable
    matched_dialect_terms: list[str] = field(default_factory=list)
    top1_id: str | None = None
    top1_title: str | None = None
    top1_score: float | None = None
    top2_score: float | None = None
    confidence: float | None = None
    escalate: bool = True
    answer: str = ""
    answer_source: str = ""  # "llm" | "template" | "escalation"
    timings_sec: dict = field(default_factory=dict)


class AdvisoryPipeline:
    def __init__(
        self,
        kb_path: str | Path = DEFAULT_KB_PATH,
        lexicon_path: str | Path = DEFAULT_LEXICON_PATH,
        adapter_path: str | Path | None = DEFAULT_ADAPTER_PATH,
        confidence_threshold: float = 0.5,
        use_llm: bool = True,
        whisper_model_size: str = "tiny",
        device: str | None = None,
        verbose: bool = True,
    ):
        from src.confidence.gate import ConfidenceGate
        from src.dialect_mapping.mapper import DialectMapper
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever

        self.verbose = verbose
        self._log("Loading KB + retrievers...")
        self.bm25 = BM25Retriever.from_kb_file(kb_path)
        self.dense = DenseRetriever.from_kb_file(kb_path, device=device)
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        self.mapper = DialectMapper.from_lexicon_file(lexicon_path)
        self.gate = ConfidenceGate(threshold=confidence_threshold)

        self._asr_cascade = None  # lazy, see _get_asr_cascade()
        self._speech_retriever = None  # lazy, see _get_speech_retriever()
        self._whisper_model_size = whisper_model_size
        self._adapter_path = Path(adapter_path) if adapter_path else None
        self._device = device

        self.llm = None
        self.answer_source_if_no_llm = "template"
        if use_llm:
            try:
                from src.generation.llm_generator import LocalLLMGenerator

                self._log("Loading local LLM generator (Qwen2.5-0.5B-Instruct)...")
                self.llm = LocalLLMGenerator(device=device)
            except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure -> documented fallback
                self._log(f"LLM generator unavailable ({type(e).__name__}: {e}); "
                          f"falling back to templated answers.")
                self.llm = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[pipeline] {msg}")

    def _get_asr_cascade(self):
        if self._asr_cascade is None:
            from src.asr.whisper_cascade import WhisperASRCascade

            self._asr_cascade = WhisperASRCascade(
                self.hybrid, model_size=self._whisper_model_size, device=self._device
            )
        return self._asr_cascade

    def _get_speech_retriever(self):
        if self._speech_retriever is None:
            if self._adapter_path is None or not self._adapter_path.exists():
                raise FileNotFoundError(
                    f"No trained adapter found at {self._adapter_path}. Run "
                    "`python -m src.speech_retrieval.train_adapter` first."
                )
            from src.speech_retrieval.speech_retriever import SpeechNativeRetriever

            self._speech_retriever = SpeechNativeRetriever(self.dense, self._adapter_path, device=self._device)
        return self._speech_retriever

    # ------------------------------------------------------------------ text --
    def answer_text_query(self, query: str, top_k: int = 5) -> PipelineResult:
        timings = {}
        t0 = time.time()
        mapping = self.mapper.map_query(query)
        timings["dialect_mapping"] = time.time() - t0

        t0 = time.time()
        retrieved = self.hybrid.search(mapping.normalized_query, top_k=top_k)
        timings["retrieval"] = time.time() - t0

        return self._finish(
            mode="text",
            original_query=query,
            normalized_query=mapping.normalized_query,
            matched_dialect_terms=mapping.matched_terms,
            retrieved=retrieved,
            query_for_generation=query,
            timings=timings,
        )

    # ---------------------------------------------------------- asr_cascade --
    def answer_audio_query_asr_cascade(self, audio_path: str | Path, top_k: int = 5) -> PipelineResult:
        timings = {}
        cascade = self._get_asr_cascade()

        t0 = time.time()
        transcript, t_asr = cascade.transcribe(audio_path)
        timings["asr_transcription"] = t_asr
        timings["asr_transcription_wall"] = time.time() - t0

        t0 = time.time()
        mapping = self.mapper.map_query(transcript)
        timings["dialect_mapping"] = time.time() - t0

        t0 = time.time()
        retrieved = self.hybrid.search(mapping.normalized_query, top_k=top_k)
        timings["retrieval"] = time.time() - t0

        return self._finish(
            mode="asr_cascade",
            original_query=transcript,
            normalized_query=mapping.normalized_query,
            matched_dialect_terms=mapping.matched_terms,
            retrieved=retrieved,
            query_for_generation=transcript,
            timings=timings,
        )

    # -------------------------------------------------------- speech_native --
    def answer_audio_query_speech_native(self, audio_path: str | Path, top_k: int = 5) -> PipelineResult:
        import librosa

        timings = {}
        retriever = self._get_speech_retriever()

        t0 = time.time()
        wav, _sr = librosa.load(str(audio_path), sr=16000, mono=True)
        timings["audio_load"] = time.time() - t0

        t0 = time.time()
        retrieved = retriever.search(wav.astype("float32"), top_k=top_k)
        timings["speech_encode_and_retrieve"] = time.time() - t0

        return self._finish(
            mode="speech_native",
            original_query="<audio query, speech-native retrieval -- no transcript produced>",
            normalized_query=None,  # dialect mapping not applied in this mode -- see module docstring
            matched_dialect_terms=[],
            retrieved=retrieved,
            query_for_generation="Please advise on the issue described in the farmer's spoken query.",
            timings=timings,
        )

    # ------------------------------------------------------------- shared --
    def _finish(self, mode, original_query, normalized_query, matched_dialect_terms, retrieved,
                query_for_generation, timings) -> PipelineResult:
        if not retrieved:
            return PipelineResult(
                mode=mode, original_query=original_query, normalized_query=normalized_query,
                matched_dialect_terms=matched_dialect_terms, answer="ESCALATE TO HUMAN EXPERT (no passages retrieved)",
                answer_source="escalation", escalate=True, timings_sec=timings,
            )

        top1 = retrieved[0]
        top2_score = retrieved[1].score if len(retrieved) > 1 else None

        t0 = time.time()
        if self.llm is not None:
            try:
                draft_answer = self.llm.generate(query_for_generation, top1.passage["text"])
                answer_source = "llm"
            except Exception as e:  # noqa: BLE001
                self._log(f"LLM generation failed at inference time ({type(e).__name__}: {e}); using template.")
                from src.generation.llm_generator import templated_answer

                draft_answer = templated_answer(query_for_generation, top1.passage)
                answer_source = "template"
        else:
            from src.generation.llm_generator import templated_answer

            draft_answer = templated_answer(query_for_generation, top1.passage)
            answer_source = "template"
        timings["generation"] = time.time() - t0

        t0 = time.time()
        conf = self.gate.score(
            top1_score=top1.score, top2_score=top2_score,
            answer_text=draft_answer, passage_text=top1.passage["text"],
        )
        timings["confidence_gate"] = time.time() - t0

        if conf.escalate:
            final_answer = (
                "ESCALATE TO HUMAN EXPERT: the system is not confident enough in an automated "
                f"answer (confidence={conf.confidence:.2f} < threshold={conf.threshold:.2f}). "
                f"Closest match found was '{top1.passage['title']}' -- a human expert should verify."
            )
            answer_source = "escalation"
        else:
            final_answer = draft_answer

        return PipelineResult(
            mode=mode,
            original_query=original_query,
            normalized_query=normalized_query,
            matched_dialect_terms=matched_dialect_terms,
            top1_id=top1.doc_id,
            top1_title=top1.passage["title"],
            top1_score=top1.score,
            top2_score=top2_score,
            confidence=conf.confidence,
            escalate=conf.escalate,
            answer=final_answer,
            answer_source=answer_source,
            timings_sec=timings,
        )


if __name__ == "__main__":
    import sys

    pipeline = AdvisoryPipeline(use_llm=True)
    # len(sys.argv) > 1, not truthiness of the joined string, so an explicit empty-string
    # argument is actually run (and escalates, same as any other unmatched query) instead of
    # silently being replaced by the demo default -- found by black-box testing that fed "" on
    # the command line and got the demo query's result back instead.
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Kapas ke phool aur tinde me gulabi sundi lag gayi hai"
    result = pipeline.answer_text_query(query)
    print(f"\nquery       : {result.original_query}")
    print(f"normalized  : {result.normalized_query}")
    print(f"top1        : {result.top1_id} ({result.top1_title})  score={result.top1_score:.4f}")
    print(f"confidence  : {result.confidence:.3f}  escalate={result.escalate}")
    print(f"answer [{result.answer_source}]: {result.answer}")
    print(f"timings     : {result.timings_sec}")
