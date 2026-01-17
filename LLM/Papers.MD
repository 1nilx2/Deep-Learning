# TTT-E2E
Instead of relying on direct reference to all past tokens via the KV-cache in the original Transformer, TTT-E2E updates the model itself so that contextual information is compressed into the model parameters in a goal-conditioned manner, driven by the next-token prediction loss.

This approach does not preserve exact token-level recall and is therefore weaker at tasks like needle-in-a-haystack retrieval. However, it exhibits scaling behavior comparable to full attention on tasks where reasoning and logical consistency are emphasized, by progressively consolidating relevant information into the model state.

If exact recall is required, this limitation can be mitigated by augmenting the model with an additional hard lookup or external memory mechanism.