# Research plates for the homepage ribbon

These are original vector thumbnails designed around each study, with a consistent editorial layout and muted ink-on-paper palette. They replace generic decorative sculptures in the homepage ribbon only. The interactive study artwork remains separate.

- **SPD:** the existing three-item assignment example, including M and the maximum-score B → A → C order. Illustrative, not model output.
- **GenRec:** history/context → h → a catalog-aware head, with learned item vectors as a separate input. The score bars are schematic and do not assume a particular undisclosed scoring function.
- **EmbeddingGemma 2:** illustrative audio signals and text, paired with actual Exp 1 Top-1 retrieval of 72.4% / 65.6%. No synthetic UMAP plot is presented as measured data.
- **Laya vs CrossEncoder:** fine-tuned accuracy 90.6% / 93.4% and measured latency 98.8 / 208.0 ms. Latency bars share a zero baseline and a 250 ms scale. Source: the existing lab RESULTS.md.
- **TurboQuant:** measured KV allocation of 896 / 357 MiB at 30,290 input tokens, from the existing lab RESULTS.md. Bars share a zero baseline and 1,000 MiB scale; the 60.2% reduction refers only to KV storage.
- **Artwork:** the three fictional After the Tide posters already used in the interactive study. The marked selection is illustrative, not an inferred Netflix user preference.

`heroImage` and `heroAlt` in `website/content.json` select these assets. The shared asset map copies them to `dist/assets/thumbnails/` during the build. Update numeric plates if their referenced reports change.
