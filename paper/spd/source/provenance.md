# SPD study provenance

Imported on November 1, 2026 from the repository owner's local SPD outputs.

- `paper/spd/index.html`: the supplied standalone `outputs/SPD_carousel.html`, retained as the canonical interactive study.
- `paper/spd/README.md`: the supplied English `outputs/SPD_notes.md`, with repository navigation added.
- `paper/spd/source/original-summary.ko.md`: the supplied `outputs/SPD_논문_정리.md`.
- `paper/spd/assets/preview.png`: the supplied carousel screenshot.
- `website/src/components/FlexCarousel.jsx` and `.css`: the original component supplied with the study in `work/spd-react/`, including its existing `navigationRef` adaptation.
- `website/src/cards/`: metallic SVG illustrations extracted from the supplied study's card data, with homepage labels changed.

The website build copies this study to `studies/spd/`, adds navigation back to the homepage and GitHub notes, and regenerates the embedded Markdown download from `paper/spd/README.md`. It does not rewrite the study's learning panels or numerical examples.

The supplied paper record is https://arxiv.org/abs/2609.01807v2. The imported study and its interpretation have not been independently checked against a new copy of the paper in this integration task. The interactive matrices and loss slider are illustrative, not training runs.
