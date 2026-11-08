"""Dialect-to-scientific-entity mapping layer -- Section III-C / RQ2.

Looks up colloquial/regional terms in data/dialect_lexicon.json and normalizes a
query by APPENDING the matched scientific entity name(s) to it before retrieval.
Appending rather than replacing is a deliberate design choice: it preserves any
other context words in the farmer's original phrasing (crop name, symptom
description) that may themselves help retrieval, while adding the vocabulary
bridge the KB is indexed under.

This module implements "architecture #1" from
agents/components/dialect-entity-mapping.md (post-hoc text normalization) applied
after ASR (whisper_cascade.py) or after any other text-producing step -- it is a
lexicon lookup over text, not an embedding-space alignment. Architecture #2
(embedding-space dialect mapping with no ASR at all) is out of scope for this
demo; see speech_retrieval/ for the no-ASR path, which does not currently include
dialect-term handling in embedding space -- a documented limitation, not a hidden
gap. See the module docstring in speech_retrieval/speech_retriever.py.

Bootstrapping note: per the paper's own scoping constraint (contribution 2), a
production version of this lexicon must be built from public resources (EPPO
ontologies, extension-service glossaries, KGs) with domain-expert review, not
invented ad hoc. data/dialect_lexicon.json is explicitly a small demo seed lexicon
-- see its own "_readme" field.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MappingResult:
    original_query: str
    normalized_query: str
    matched_terms: list[str] = field(default_factory=list)
    candidate_entities: list[str] = field(default_factory=list)  # union of all maps_to, flattened
    ambiguous: bool = False  # True if any matched term had >1 maps_to candidate


class DialectMapper:
    def __init__(self, lexicon_entries: list[dict], lexicon_path: Path | None = None):
        # Sort longest-term-first so multi-word terms are matched before their
        # single-word substrings (e.g. "hara tela" before "tela").
        self.entries = sorted(lexicon_entries, key=lambda e: len(e["term"]), reverse=True)
        self.lexicon_path = lexicon_path

    @classmethod
    def from_lexicon_file(cls, path: str | Path) -> "DialectMapper":
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["entries"], lexicon_path=p)

    def learn_term(self, term: str, maps_to: list[str], region_note: str = "Learned from expert feedback", confidence: str = "high", persist: bool = True) -> bool:
        """Dynamically add or update a colloquial-to-scientific mapping (Self-Learning)."""
        term_clean = term.strip().lower()
        if not term_clean or not maps_to:
            return False

        # Check if already present
        existing = next((e for e in self.entries if e["term"].lower() == term_clean), None)
        if existing:
            # Merge candidate entities
            for ent in maps_to:
                if ent not in existing["maps_to"]:
                    existing["maps_to"].append(ent)
            existing["confidence"] = confidence
        else:
            new_entry = {
                "term": term_clean,
                "region_note": region_note,
                "maps_to": maps_to,
                "confidence": confidence,
            }
            self.entries.append(new_entry)
            self.entries.sort(key=lambda e: len(e["term"]), reverse=True)

        if persist and self.lexicon_path and self.lexicon_path.exists():
            with open(self.lexicon_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Update data['entries']
            entries_list = data.get("entries", [])
            found = False
            for e in entries_list:
                if e["term"].lower() == term_clean:
                    for ent in maps_to:
                        if ent not in e["maps_to"]:
                            e["maps_to"].append(ent)
                    e["confidence"] = confidence
                    found = True
                    break
            if not found:
                entries_list.append({
                    "term": term_clean,
                    "region_note": region_note,
                    "maps_to": maps_to,
                    "confidence": confidence,
                })
            data["entries"] = entries_list
            with open(self.lexicon_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return True

    def map_query(self, query: str) -> MappingResult:
        query_lower = query.lower()
        matched_terms: list[str] = []
        candidate_entities: list[str] = []
        ambiguous = False
        consumed_spans: list[tuple[int, int]] = []

        for entry in self.entries:
            term = entry["term"].lower()
            # Boundary-aware token matching (\b) so substrings inside unrelated words don't match,
            # while still permitting multi-word expressions like "hara tela".
            pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
            for m in re.finditer(pattern, query_lower):
                span = m.span()
                if any(not (span[1] <= s or span[0] >= e) for s, e in consumed_spans):
                    continue  # already covered by a longer term match
                consumed_spans.append(span)
                matched_terms.append(entry["term"])
                maps_to = entry["maps_to"]
                if len(maps_to) > 1:
                    ambiguous = True
                for ent in maps_to:
                    if ent not in candidate_entities:
                        candidate_entities.append(ent)

        if candidate_entities:
            normalized_query = query + " " + " ".join(candidate_entities)
        else:
            normalized_query = query

        return MappingResult(
            original_query=query,
            normalized_query=normalized_query,
            matched_terms=matched_terms,
            candidate_entities=candidate_entities,
            ambiguous=ambiguous,
        )


if __name__ == "__main__":
    import sys

    lex_path = Path(__file__).resolve().parents[2] / "data" / "dialect_lexicon.json"
    mapper = DialectMapper.from_lexicon_file(lex_path)
    q = " ".join(sys.argv[1:]) or "Kapas ke phool aur tinde me gulabi sundi lag gayi hai"
    result = mapper.map_query(q)
    print("original  :", result.original_query)
    print("normalized:", result.normalized_query)
    print("matched   :", result.matched_terms)
    print("candidates:", result.candidate_entities, "(ambiguous)" if result.ambiguous else "")
