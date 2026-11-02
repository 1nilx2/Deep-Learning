# Local Study/Lab reading dossiers

The EmbeddingGemma, Laya, and TurboQuant dossiers summarize the existing tracked experiment reports. They do not run models or rewrite the underlying experiment protocols. Their downloadable reports are copied from the repository at build time.

EmbeddingGemma panels import the original Exp 1 `summary.json`. Laya and TurboQuant panels use the recorded tables in their existing RESULTS.md files. If those reports change, review the curated chapter copy and measured constants in `experiment-labs.jsx` as well.

`spd-chapters.json` retains the supplied SPD chapter titles/tags for visual generation. The original interactive SPD page and its numerical logic remain canonical in `paper/spd/index.html`; the build replaces its carousel images and adds the new paper-colored panel treatment.

`website/scripts/research-plates.mjs` renders a distinct topic-specific plate for every chapter: 8 SPD, 8 GenRec, 8 Artwork, and 5 for each experimental Lab. Architecture and numerical teaching examples are identified as such. Reported measurements retain their comparison bases and interpretation limits.
