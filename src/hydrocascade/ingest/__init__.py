"""Layer 1 — data ingestion (Doc 01 §6.1).

Two tracks, per the plan agreed for Phases 1-2:

* **Open sources** are fetched automatically and write ``PROVENANCE.md`` at
  download time: Copernicus DEM GLO-30, HydroSHEDS/HydroBASINS, ESA WorldCover,
  SoilGrids, CHIRPS rainfall, NASA COOLR.
* **Login-gated sources** (GloFAS, ERA5-Land, IMERG, IMD — Doc 10 §7) sit behind
  adapters in :mod:`hydrocascade.ingest.adapters`. Each has a ``real`` path that
  runs once credentials exist and a ``synthetic`` path that keeps the pipeline
  runnable meanwhile. Which one runs is a config flag (``data_mode.*``), never a
  code edit.
"""
