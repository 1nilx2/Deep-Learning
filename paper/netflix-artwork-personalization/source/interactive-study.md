# Interactive study provenance

The eight-chapter `study.json` adapts the existing reviewed English README, preserving the distinctions between SFT, reasoning-SFT, DPO, sampled alternatives, caption generation/matching, and offline IPS results.

The shared React interface is maintained in `website/src/study/` and uses the existing original React Bits FlexCarousel. Metallic chapter art adapts the repository owner's supplied SPD vector artwork. The fictional film *After the Tide*, its schematic poster illustrations, viewer profiles, captions, prompt, token likelihoods, and response log ratios are teaching examples. They are not Netflix titles, Netflix artwork, live model predictions, or measured user preferences.

The simplified word-bigram matcher explains caption-to-candidate mapping; it does not claim to reproduce the paper's exact matching implementation. Its ties and zero-overlap cases explicitly return no unique match. The notes download is copied from the current README at build time.
