# `paper/reference.bib` and `paper/pdfs/` — Bibliography Component

## Purpose

`paper/reference.bib` is the single source of truth for citations used in
`paper/main.tex`. `paper/pdfs/` holds a local copy of every entry — a PDF where one is
freely obtainable, otherwise a `.txt` citation stub — so the work doesn't
depend on live internet access to re-verify what was cited.

## Verification-status tags (from the header of `reference.bib`)

The file's own header defines a status key — preserve this convention for any
reference you add or edit:

- `[VERIFIED]` — confirmed on IEEE Xplore / ACL Anthology / NeurIPS / ScienceDirect / Nature / Frontiers. Scopus + WoS indexed.
- `[ACCEPTED]` — accepted at an IEEE conference; Xplore record not yet live. Cite via arXiv now, swap in the Xplore DOI once it appears.
- `[PREPRINT]` — arXiv only, no peer-reviewed venue confirmed.
- `[WEB]` — blog post/announcement, not a citable paper in the academic sense (used sparingly, for systems like SukhaRakshak AI and Google S2R that are only documented via blog).
- `[UNVERIFIED]` — found via a secondary source; bibliographic details not independently confirmed. Must be verified before final submission (currently only `kgeditorial2023`).

When adding a new reference: pick the correct tag honestly, don't default to
`[VERIFIED]`. If unsure, use `[UNVERIFIED]` and leave a `note` field
explaining what's missing.

## Entries requiring attention before final submission

As of 13 July 2026, all previously-open verification items have been resolved
via CrossRef API / Frontiers full-text lookups (not just secondary sources),
and `reference.bib`'s status tags and notes updated accordingly:

- `min2025speechrag` — **Resolved.** DOI `10.1109/ICASSP49660.2025.10888900`
  independently confirmed via the CrossRef API (title, all six authors, and
  publication date 2025-04-06 match exactly), not just inferred from the
  Xplore document ID as before. Preprint cross-check at arXiv:2412.16500
  (`paper/pdfs/min2025speechrag.pdf`) still stands.
- `singh2023respin` — **Resolved.** Full 18-author list and DOI
  (`10.1007/978-3-031-48312-7_14`) confirmed via CrossRef API, replacing the
  `Singh, Aditya and others` placeholder. No open-access PDF found (Springer
  paywalled) — `paper/pdfs/singh2023respin.txt` still has the citation stub;
  that's expected, not a gap.
- `kgeditorial2023` — **Resolved and reclassified from `[UNVERIFIED]` to
  `[VERIFIED]`.** Full author list (Roussey, Guéret, Laporte), journal
  (Frontiers in Artificial Intelligence, vol. 6, article 1319844), and DOI
  (`10.3389/frai.2023.1319844`) confirmed directly from the Frontiers
  full-text page, not just the PMC secondary reference. Since it's open
  access, the real PDF has been fetched to `paper/pdfs/kgeditorial2023.pdf`
  (verified by extracting its text and matching the DOI/author header),
  replacing the old `.txt` stub.
- `google2025s2r` — previously had no author list or URL in the `.bib`.
  Both have now been confirmed (Ehsan Variani and Michael Riley, Google
  Research Blog, published 7 Oct. 2025,
  research.google/blog/speech-to-retrieval-s2r-a-new-approach-to-voice-search)
  and the `.bib` entry updated; full article text is saved in
  `paper/pdfs/google2025s2r.txt` since blog posts have no PDF.
- Any entry currently backed only by a `paper/pdfs/<key>.txt` fallback (see
  below) instead of a real PDF — the citation itself may still be fine for
  submission (paywalled venues are normal to cite), but don't assume a PDF
  exists locally.

## `paper/pdfs/` folder convention

One file per bibkey, named `<bibkey>.pdf` or `<bibkey>.txt` — the key matches
the `@entrytype{key, ...}` identifier in `reference.bib` exactly, so the two
files are trivially cross-referenced.

`.txt` fallback format (used when no open-access PDF exists — e.g. Elsevier,
IEEE Xplore, Springer paywalled content, or blog posts with no stable PDF):

```
Title: <title>
Authors: <authors>
Venue/Year: <venue, year>
DOI/URL: <doi or url, or "unknown">
Status: <why no PDF, e.g. "IEEE Xplore paywalled", "no verified source found">
```

Only open-access sources were fetched as real PDFs: arXiv, PMLR, ACL
Anthology, NeurIPS Proceedings, Nature Scientific Data, Frontiers journals.
Paywalled IEEE/Elsevier/Springer content and blog posts without a stable PDF
got `.txt` stubs instead — this was a deliberate choice, not an oversight;
don't try to defeat paywalls to "complete" the set.

## Adding a new reference

1. Add the BibTeX entry to `reference.bib` in the right section (grouped by
   verification status, matching the existing file's organization).
2. Fetch/save the paper into `paper/pdfs/<key>.pdf` if open access, else write
   `paper/pdfs/<key>.txt` in the format above.
3. Cite it in `main.tex` with `\cite{key}` and make sure the surrounding
   prose actually supports the claim being attributed to it — don't cite
   speculatively.
