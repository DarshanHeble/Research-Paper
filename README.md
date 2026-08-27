# Speech-Native Retrieval-Augmented Generation for Agricultural Advisory in Low-Resource Dialects

Welcome to the repository for the IEEE conference paper: **"Speech-Native Retrieval-Augmented Generation for Agricultural Advisory in Low-Resource Dialects"**, authored by Darshan Pundlik Heble (PG Scholar, Dept. of Computer Science, Christ (Deemed to be University), Bangalore).

This repository contains both the source code for the paper itself and a fully runnable software prototype of the proposed architecture.

## Overview

The paper proposes a framework/protocol for agricultural voice advisory in dialect-rich, low-resource settings. It addresses the challenges of existing systems by combining four key innovations:

1. **Speech-native retrieval** — retrieving directly from a speech representation, bypassing the ASR transcription step which often fails on domain-critical vocabulary.
2. **Dialect-to-scientific-entity mapping** — normalizing colloquial/regional farmer terms to standard agricultural nomenclature using low-cost public resources.
3. **Confidence-gated escalation** — gracefully abstaining and routing to a human expert when the system is not confident.
4. **Fully offline operation** — designed to run on consumer-grade, GPU-constrained hardware.

For a deeper dive into the research questions and hypotheses, refer to the documentation in `agents/paper.md` and the LaTeX source in `paper/main.tex`.

## Repository Layout

- **`paper/`**: Contains everything needed to compile the IEEE conference paper.
  - `main.tex`: The LaTeX source code.
  - `reference.bib`: Bibliography references.
- **`implementation/`**: The runnable Python prototype of the proposed system. Contains the CLI pipeline, hybrid retrieval, dialect mapping, and the local LLM generation.
- **`frontend/`**: A lightweight, browser-based static UI to interact with the system via HTTP.
- **`agents/`**: Extensive documentation outlining the architecture, components, and guidelines for working on this repository.
  - Read `AGENTS.md` first to orient yourself.
- **`ARCHITECTURE.md`**: Details on the demo setup (Frontend + Backend) and how the components interact.
- **`start.sh`**: A one-click script to spin up the entire demo environment.

## Running the Demo

The repository includes a live, demoable system comprising a FastAPI backend and a static HTML frontend. 

To start the system, simply run:

```bash
./start.sh
```

This command will:
1. Start the backend server on port `8000`. This process loads the Knowledge Base, retrievers, and the local LLM into memory.
2. Serve the frontend on port `5173`.
3. Wait for the backend health check to pass and output the URLs to access the demo.

Open `http://localhost:5173` in your browser to interact with the system. You can type text queries or record voice queries to see the pipeline in action.

## Important Notes

- **Experimental Data**: No experiments have been run yet on the target dialect. The system is a prototype that demonstrates the feasibility of the architecture. Section VI of the paper reports numbers on small demo data as a mechanism check.
- **Adding Contributions**: Please read `AGENTS.md` before making any modifications. It outlines the ground rules for maintaining IEEE formatting and the integrity of the paper's claims.

## Author

**Darshan Pundlik Heble**  
PG Scholar, Dept. of Computer Science, Christ (Deemed to be University), Bangalore.
