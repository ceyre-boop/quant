"""Layer-1 directional bias model — data pipeline (HYP-064, Phase 2).

Pre-registration: data/research/preregister/HYP-064_layer1_directional_bias.json
Feature spec:     docs/layer1/feature_windows.json + docs/layer1/features.md

Phase 2 builds the historical feature matrix + forward-direction labels ONLY. No model is
trained here (Phase 3) and the 2024+ holdout is never read.
"""
