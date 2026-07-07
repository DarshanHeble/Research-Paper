"""Optional local LLM generation step -- Section III (pipeline stage 4b).

The spec for this prototype treats a full local-LLM generation step as
"optional/nice-to-have": a templated "here is the most relevant passage" answer
is an acceptable, clearly-documented stand-in if generation can't be wired up.
This module wires up a REAL small local model (Qwen2.5-0.5B-Instruct, ~1GB fp16,
comfortably fits the 6GB RTX 3050 target alongside the other loaded models) so
the pipeline can do real grounded generation, not just templating -- but the
pipeline (pipeline.py) always falls back to the deterministic template if this
model can't be loaded (no network, no VRAM headroom, etc.), and always logs
which path was actually used. Never silently pretend templated output came from
the LLM or vice versa.

This is still a small, general-purpose instruct model with no agricultural
domain fine-tuning and no dialect-language training -- it is prompted to answer
ONLY from the retrieved passage (grounded generation), which is what the gate's
open-book grounding-overlap signal in confidence/gate.py then checks.
"""
from __future__ import annotations

from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# In the reference environment, huggingface_hub's chunked/resumable downloader
# (used transparently by from_pretrained) repeatedly stalled/reset partway through
# this ~1GB safetensors file for reasons never fully diagnosed (a plain `curl`
# fetched the same URL over the same network without issue -- see the build
# history/README for the working-around-it steps: a full weights file was fetched
# with curl and assembled into a local directory alongside the small tokenizer/
# config files copied out of the (correctly, fully) cached snapshot). If that
# local copy exists, prefer it; otherwise fall back to the normal hub id so this
# still works out-of-the-box on a machine that doesn't hit the same download issue.
_LOCAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "qwen2.5-0.5b-instruct"

PROMPT_TEMPLATE = """You are an agricultural advisory assistant helping a smallholder farmer. \
Answer using ONLY the information in the passage below -- do not add outside knowledge. \
Be concise (2-4 sentences) and practical.

Passage: {passage_text}

Farmer question: {query}

Answer:"""


class LocalLLMGenerator:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None, max_new_tokens: int = 120):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = str(_LOCAL_MODEL_DIR) if _LOCAL_MODEL_DIR.exists() else model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(resolved, dtype=dtype).to(self.device)
        self.model.eval()

    def generate(self, query: str, passage_text: str) -> str:
        import torch

        prompt = PROMPT_TEMPLATE.format(passage_text=passage_text, query=query)
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def templated_answer(query: str, passage: dict) -> str:
    """Deterministic non-LLM fallback: a templated grounded answer from the top passage."""
    return (
        f"Based on your query, this most likely matches: {passage['title']}. "
        f"{passage['text']}"
    )
