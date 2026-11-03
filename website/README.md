# Deep Learning — Fieldnotes

[Live website](https://1nilx2.github.io/Deep-Learning/) · [SPD interactive study](https://1nilx2.github.io/Deep-Learning/studies/spd/)

A repository homepage using the supplied **TiltedGridHero**: a continuous ribbon of metallic cards curving across the hero, with motion controls and a featured-study selector. The shared paper studies use the original React + OGL **FlexCarousel** for chapter navigation, lens rendering, and click-to-focus zoom.

## Build and preview

From the repository root:

```sh
npm ci --prefix website
npm run build --prefix website
npm run check --prefix website
python -m http.server 4173 --directory website/dist
```

Open http://localhost:4173. The collection includes seven coordinated Study/Lab pages:

- `studies/gr2/` — Learn to reason. Learn to rerank. An English study following the supplied handwritten reading map.
- `studies/spd/` — A single pass. A complete ranking.
- `studies/genrec/` — A history of choices. A new way to rank.
- `studies/artwork/` — One title. A different first impression.
- `studies/embedding/` — Audio–text retrieval and modality signatures.
- `studies/laya/` — Accuracy, calibration, and measured CPU latency.
- `studies/turbo/` — KV allocation, total Metal memory, and storage projections.

The output is a static website with bundled dependencies and local artwork. It needs no CDN or server framework. The homepage ribbon uses CSS transforms, supports reduced motion, and includes a pause control. Its images are decorative; the study selector and archive provide navigation. The paper studies use WebGL 2 for the original lens effects and zoom. GenRec and Artwork also provide a card-list fallback when WebGL 2 is unavailable. Archive search and category filters work locally.

## Keep GitHub and the website in sync

- **SPD notes:** edit `paper/spd/README.md`. The build regenerates both the downloadable `.md` file and the study's embedded notes download from this file.
- **Chapter plates:** `website/scripts/research-plates.mjs` generates 46 chapter-specific research plates into `dist/assets/plates/`. They share the muted paper-and-ink palette of the homepage thumbnails and use schematic mechanisms or existing reported results, with interpretation boundaries retained.
- **Experimental Lab pages:** curated chapter content is in `website/src/study/data/`; explorers are in `experiment-labs.jsx`. Full report downloads are copied from their canonical source notes. These dossiers do not rewrite the underlying protocols.
- **SPD learning panels:** edit `paper/spd/index.html`, the canonical standalone study. The build copies it to `dist/studies/spd/index.html` and adds homepage/GitHub navigation. It replaces the embedded carousel artwork with generated chapter plates while retaining the original matrix and loss interactions.
- **Homepage content:** edit `website/content.json` to add or change featured works and archive entries. A `page` field opens a local interactive study; `path` links to its repository notes.
- **Homepage thumbnails:** `website/src/thumbnails/` contains topic-specific research plates. `heroImage` and `heroAlt` in `content.json` select them. See the thumbnail README for the provenance of schematic examples and measured values. `thumbnail-review.html` presents all seven plates.
- **Repository figures:** edit the original paths listed in `content.json` under `assets`. Each build copies those sources, so figures are not maintained twice.
- **Design and behavior:** `website/src/index.html`, `styles.css`, `app.tsx`, `home.css`, and `archive.js`.
- **GenRec and Artwork chapters:** edit `paper/genrec/study.json` or `paper/netflix-artwork-personalization/study.json`. Each has eight chapters and a named explorer. The shared study UI is in `website/src/study/`; the original metallic artwork is retained as source material; deployed chapter cards use generated research plates. Builds regenerate the deployed pages and notes downloads from these sources.
- **Homepage component:** `website/src/components/ui/tilted-grid-hero.tsx`, supplied with the approved homepage design. Only its utility import is adapted. Its Tailwind utility classes are implemented in scoped CSS in `home.css`; esbuild compiles its TSX without a framework migration.
- **Study component:** `website/src/components/FlexCarousel.jsx` and `FlexCarousel.css`.
- **Generated files:** `website/dist/`; rebuild instead of editing them directly.

A repository note is not automatically transformed into an interactive learning panel. The source Markdown, interactive HTML, and curated homepage summary serve different purposes. The build keeps the deployed study and downloads aligned with their canonical repository files.

The homepage features 7 studies and indexes 15 selected entries. Section links lead to the complete Papers, Notes, and Labs collections. The EmbeddingGemma spotlight describes Exp 1 explicitly.

## Automatic GitHub Pages deployment

`.github/workflows/website-pages.yml` runs on pushes to `main` affecting `website/`, `paper/`, `notes/`, `labs/`, `assets/`, or the workflow itself. It installs locked dependencies, rebuilds from repository sources, verifies study synchronization, links, and the numerical explorers, and deploys `website/dist`. Manual dispatch is also supported.

GitHub Pages is configured to use **GitHub Actions**. Pushing source updates publishes the updated website through the workflow. The existing GitHub repository interface remains a README; GitHub Pages serves the separate interactive site.

Relative URLs support the `/Deep-Learning/` project path.

Official workflow reference: https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages

## Attribution

See `THIRD_PARTY_NOTICES.md` for React Bits, React, and OGL attribution. SPD source provenance is recorded in `paper/spd/source/provenance.md`.

The earlier private Sites registration is retained locally in `.openai/hosting.json` and excluded from GitHub. No files are uploaded to that separate Sites destination by this workflow.

## GR2 reading study

Canonical sources are `paper/gr2/README.md` and `paper/gr2/study.json`. The attached v6 PDF and decoded handwritten reference are in `paper/gr2/source/`. The editable schematic is `paper/gr2/assets/overview.svg`; the homepage thumbnail uses the same artwork in `src/thumbnails/gr2.svg`. `src/study/gr2-labs.jsx` provides seven chapter explorers; reasoning-data selection and student SFT share chapter 3; `gr2-math.mjs` implements equations (14)–(17) for valid target ranks. Checks cover copied rankings, initial-rank-one exceptions, format gating, and equal-reward groups. Teaching examples do not run a model or reproduce training.

All GR2 website copy, plates, explorers, and downloadable notes are in English. The original handwritten image is preserved as a repository source reference; it is not copied into the deployed site.
