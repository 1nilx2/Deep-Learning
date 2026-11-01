# Deep Learning — Fieldnotes

[Live website](https://1nilx2.github.io/Deep-Learning/) · [SPD interactive study](https://1nilx2.github.io/Deep-Learning/studies/spd/)

A repository homepage using the **original React + OGL FlexCarousel** supplied with the SPD study: liquid lens rendering, metallic illustrated cards, drag gestures, keyboard navigation, and click-to-focus zoom. No CSS imitation is used for the main gallery.

## Build and preview

From the repository root:

```sh
npm ci --prefix website
npm run build --prefix website
npm run check --prefix website
python -m http.server 4173 --directory website/dist
```

Open http://localhost:4173. Open http://localhost:4173/studies/spd/ for **A single pass. A complete ranking.**

The output is a static website with bundled dependencies and local artwork. It needs no CDN or server framework. Browsers with WebGL 2 show the original lens effects and zoom; a scrollable card list remains available when WebGL 2 is unavailable. Vertical page scrolling is preserved; horizontal trackpad gestures move the carousel. Archive search and category filters work locally.

## Keep GitHub and the website in sync

- **SPD notes:** edit `paper/spd/README.md`. The build regenerates both the downloadable `.md` file and the study's embedded notes download from this file.
- **SPD learning panels:** edit `paper/spd/index.html`, the canonical standalone study. The build copies it to `dist/studies/spd/index.html` and adds homepage/GitHub navigation.
- **Homepage content:** edit `website/content.json` to add or change featured works and archive entries. A `page` field opens a local interactive study; `path` links to its repository notes.
- **Repository figures:** edit the original paths listed in `content.json` under `assets`. Each build copies those sources, so figures are not maintained twice.
- **Design and behavior:** `website/src/index.html`, `styles.css`, `app.jsx`, and `archive.js`.
- **Original component:** `website/src/components/FlexCarousel.jsx` and `FlexCarousel.css`.
- **Generated files:** `website/dist/`; rebuild instead of editing them directly.

A repository note is not automatically transformed into an interactive learning panel. The source Markdown, interactive HTML, and curated homepage summary serve different purposes. The build keeps the deployed study and downloads aligned with their canonical repository files.

The homepage features 6 studies and indexes 14 selected entries. Section links lead to the complete Papers, Notes, and Labs collections. The EmbeddingGemma spotlight describes Exp 1 explicitly.

## Automatic GitHub Pages deployment

`.github/workflows/website-pages.yml` runs on pushes to `main` affecting `website/`, `paper/`, `notes/`, `labs/`, `assets/`, or the workflow itself. It installs locked dependencies, rebuilds from repository sources, verifies study synchronization and links, and deploys `website/dist`. Manual dispatch is also supported.

GitHub Pages is configured to use **GitHub Actions**. Pushing source updates publishes the updated website through the workflow. The existing GitHub repository interface remains a README; GitHub Pages serves the separate interactive site.

Relative URLs support the `/Deep-Learning/` project path.

Official workflow reference: https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages

## Attribution

See `THIRD_PARTY_NOTICES.md` for React Bits, React, and OGL attribution. SPD source provenance is recorded in `paper/spd/source/provenance.md`.

The earlier private Sites registration is retained locally in `.openai/hosting.json` and excluded from GitHub. No files are uploaded to that separate Sites destination by this workflow.
