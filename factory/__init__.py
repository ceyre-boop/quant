"""factory/ — the model factory. Built complete; IGNITION GATED.

Training executes ONLY against a hypothesis-ledger entry whose status is CONFIRMED
(RISK_CONSTITUTION Article 6, enforced in factory.train with a tested refusal).
Everything else here is infrastructure: point-in-time feature store (hashed
snapshots), purged walk-forward validation, a small calibrated model zoo with an
abstain-below-confidence wrapper (Article 4 encoded), a registry where nothing is
anonymous, and a paper adapter that is built, stub-tested, and NOT enabled.

Nothing in this package is importable by the live/backtest execution path.
"""
