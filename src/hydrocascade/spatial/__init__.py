"""Layer 2 — the spatial framework (Doc 01 §6.1, Doc 05).

This layer answers the reviewer question *"what does location actually mean in
your model?"* It builds two supports and the operator between them:

* **H**, the hillslope domain: slope units, ~0.25 km2, where landslides live.
* **N**, the channel network: 500 m directed reaches, where floods live.
* **pi: H -> N**, the projection operator: follow the D8 flow path from a
  hillslope point to the first channel cell it reaches.

Doc 05 §7 calls this a scientific decision rather than plumbing, and builds the
travel-time and reachability tables here because they are expensive and are
needed unchanged throughout Phase 4.
"""
