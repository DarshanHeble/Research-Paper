# Component 2 — Dialect-to-Scientific-Entity Mapping

Answers **RQ2**: Does a dialect-to-scientific-entity mapping layer improve
retrieval accuracy for queries phrased in regional or colloquial terms?

## What it is

A normalization layer that maps the colloquial/regional/dialectal term a
farmer actually uses (spoken) onto the standard scientific entity name
(pesticide, crop disease, pest) that the knowledge base is indexed under.
Without this, even a perfect speech-native retriever fails when the farmer's
word for a pest has no lexical or embedding-space relationship to the
scientific term in the documents.

## Prior art this builds on

- **Crop GraphRAG** (`wu2026cropgraphrag`) — multi-relational KG over 28 crop
  categories, retrieval jointly over graph subgraphs + text. Built from
  standardized written text only.
- **AgCNER** (`yao2024agcner`) — large-scale NER dataset for agricultural
  pests/diseases (Chinese). Useful as a methodology template for entity
  tagging, not directly for the target dialect/language.
- **EPPO ontology and related KG standards** (`kgeditorial2023`) — cross-system
  interoperable codes for plant/pest entities; a plausible target vocabulary
  for the "scientific entity" side of the mapping.
- **Liu et al., soft-prompt-tuning for colloquial pest/disease classification**
  (`liu2025colloquialpest`) — extracts entities from colloquial, unstructured
  *text* symptom descriptions via a pretrained agricultural LM, queries a KG,
  classifies. Confirms conventional classifiers degrade badly on farmers'
  natural language — but this is text-domain and classification, not
  speech-domain retrieval.
- **Bengali cross-lingual RAG case study** (`hossain2026bengalirag`) — simpler
  keyword-injection strategy bridging colloquial Bengali terms with
  scientific English nomenclature, in text. This paper's own future-work
  section explicitly names ASR integration and dialect-aware normalization of
  *spoken* queries as unaddressed — direct confirmation the speech-domain
  version of this problem is open.

## The gap this paper targets

All existing colloquial-to-formal mapping work operates on **written text**.
None of it operates on **spoken** queries, and none of it is designed as a
component that feeds a *retrieval* pipeline rather than a classifier.

## Design implications for the architecture

- **Bootstrapping constraint (contribution 2):** must be built from low-cost,
  publicly available resources — existing lexicons, KGs, written corpora,
  EPPO-style ontologies — not by collecting new dialect speech data. This is
  a hard scoping constraint, not a nice-to-have; any design that assumes a
  large dialect speech corpus violates the paper's own stated contribution.
- Two plausible architectures, both worth naming explicitly in §III:
  1. **Post-ASR text normalization** — cheaper, but re-introduces dependence
     on a transcription step for this one layer even if retrieval itself is
     speech-native elsewhere; only viable if paired with a dialect-adapted
     ASR component (à la A.A.H.A.R. / Krishi Sathi) purely for entity
     extraction, not for full-query transcription.
  2. **Embedding-space mapping** — align dialect-term speech embeddings (or a
     small labeled set of dialect term → scientific entity pairs) directly
     into the same space the speech-native retriever already uses, so no
     text transcription is needed at all. More consistent with the paper's
     "no ASR in the loop" framing but harder to bootstrap cheaply.
- Evaluation must specifically include a colloquial/regional-term test subset
  (not just standard-terminology queries) to actually test RQ2 — a system
  that only helps on already-standard phrasing hasn't answered the question.
