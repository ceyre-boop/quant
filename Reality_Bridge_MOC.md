# 🏗️ Reality Bridge MOC

This document is the **Ground Truth** for the CLAWD Trading Desk transition. 
**Current Status:** Phase 0 — Pre-Deployment Audit (Rescuing Q1 Diagnostic)

---

## 🔒 Locked Parameters (DO NOT EDIT)
These parameters have been derived from theory and verified by the Q1 Diagnostic. They are now locked for the true Out-of-Sample (OOS) test.

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| **ATR% Gate** | **4.0%** | Rejects structural noise where ICT patterns lose deterministic value. |
| **Lookahead Delta** | **Next Bar** | Entries at T, Evaluation starts at T+1. Non-negotiable. |
| **Capital Scaler** | **A+: 150%, A: 100%, B: 50%, C: 25%** | Mathematically expresses institutional conviction. |

---

## 🚫 Permanent Blacklist
The following symbols have been removed from the research universe due to structural product design incompatibility (leverage decay).

- **Leveraged ETFs**: `SPXL`, `SPXU`, `SQQQ`, `TQQQ`
- **Volatility Decay**: `UVXY`, `VIXY`, `VXX`
- **Reasoning**: Their mathematical decay resets daily, which conflicts with the multi-day hold period of the ICT Structural Edge.

---

## ⚖️ The Quant Coding Doctrine
The system is built on the 10 Pillars of Institutional Development:

1. **Deterministic Foundations**: Fixed `random_seed: 42`.
2. **Scientific Integrity Layer**: Next-bar execution (Pillar 2) enforced.
3. **Modular Architecture**: Seven-layer decoupled stack.
4. **Defensive Coding**: Pillar 4 Assertions and Type Enforcement.
5. **Logging & Telemetry**: R-multiple attribution logged per trade.
6. **Parameter Governance**: Config locked in `config/parameters.yml`.
7. **Research Workflow**: Falsifiable hypothesis-driven cycles.
8. **Testing Protocol**: Unit and Integration tests (Pillar 8).
9. **Documentation**: Concise, Versioned, Executable.
10. **Governance & Auditability**: Hash-locked manifests per run.

---

## 🔬 Diagnostic Audit Ledger

### Step 1: Rescue Cycle (In-Sample Diagnostic)
- **Range**: Dec 2025 – Apr 2026
- **Purpose**: Verify the mechanical effectiveness of V2.1 Wide-Band Risk.
- **Label**: **IN-SAMPLE** (Mechanism verification only).
- **Result**: +$126 PNL / 44% WR (Integrity verified).
- [x] **Lookahead Bias**: FIXED. Backtest now enforces next-bar P&L logic.
- [x] **Split Calibration**: BLACKLIST ACTIVE. Leveraged ETFs removed to prevent split-related data drift.
- [ ] **Execution Slippage**: TBD. (Phase 1 Paper Trading).

---

## 🚀 The True Frontier (TBD)
**True Out-of-Sample (OOS) Goal**: April 10, 2026 Forward.
- **Condition**: 200 signals over 3 months.
- **Goal**: Positive Equity Curve + 72% WR on unseen data.
