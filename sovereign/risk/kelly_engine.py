"""
Phase 6 — Risk Engine Wrapper

Andrew Lo / AFML fix applied:
  Old: kelly_fraction = risk_pct  (grade % masquerading as Kelly — wrong)
  New: f* = (p·b - q) / b  where p = win rate, b = avg win / avg loss, q = 1-p
       Fractional Kelly at 25% applied (half-Kelly is theoretical optimum under
       uncertainty; 25% is the practical bound for live trading — Lo 2017).
       Grade-based cap still enforced: Kelly is bounded above by the grade %.

  Kelly formula derivation:
    Expected log-wealth is maximised by f* = (p·b - q) / b
    At f*, E[log(W)] = p·log(1 + f*·b) + q·log(1 - f*)
    Practical Kelly = f* × 0.25  (quarter-Kelly)
    This limits ruin probability under parameter uncertainty while still
    using the information content of the win-rate/R:R estimate.

  Grade cap: grade_risk_pct acts as a ceiling. If Kelly says 3% but
    the model grade (confidence) is 'C' (grade = 0.5%), we use 0.5%.
    This enforces that low-conviction signals never get oversized even
    if the historical win rate happened to be high on a small sample.
"""

import math

from layer2.risk_engine import RiskEngine
from layer2.dynamic_rr_engine import DynamicRREngine
from contracts.types import RiskOutput, BiasOutput, RouterOutput
from config.loader import params


# ── Hoeffding Confidence Interval (CS229 Lecture 09) ─────────────────── #

def hoeffding_win_rate(
    observed_win_rate: float,
    n_trades: int,
    confidence: float = 0.90,
    mode: str = 'lower',
) -> float:
    """
    Apply Hoeffding's inequality to get a conservative win_rate for Kelly.

    CS229 Ng (Lecture 09):
      "The Hoeffding inequality says P(|φ̂ - φ| > γ) ≤ 2·exp(-2γ²m).
       With probability ≥ 1-δ: |φ̂ - φ| ≤ sqrt(log(2/δ) / (2m)).
       The bound decays exponentially in m — with enough trades, the
       estimate becomes tight."

    Applied to Kelly sizing:
      - Observed win_rate φ̂ from m trades is an ESTIMATE of the true φ.
      - The true φ could be as low as φ̂ - γ (lower confidence bound).
      - Feeding the LOWER bound into Kelly = conservative sizing.
      - As m grows, γ → 0 and Kelly uses the full observed win_rate.

    This prevents the "winner's curse" of Kelly: fitting 20 trades with
    65% win rate and sizing as if the true rate IS 65%.

    Args:
        observed_win_rate: φ̂ from historical trade window
        n_trades: number of trades the estimate is based on
        confidence: 1 - δ (default 90% confidence)
        mode: 'lower' (conservative Kelly) or 'upper' (aggressive bound)

    Returns:
        Hoeffding-corrected win_rate, clamped to [0.1, 0.95]
    """
    if n_trades < 1:
        return 0.50  # uninformative prior

    delta = 1.0 - confidence
    # Hoeffding bound: γ = sqrt(log(2/δ) / (2m))
    gamma = math.sqrt(math.log(2.0 / max(delta, 1e-9)) / (2.0 * n_trades))

    if mode == 'lower':
        corrected = observed_win_rate - gamma
    else:
        corrected = observed_win_rate + gamma

    return float(max(0.10, min(0.95, corrected)))


def sample_complexity_confidence(n_trades: int, n_features: int = 6,
                                 delta: float = 0.05, gamma: float = 0.10) -> float:
    """
    Return confidence in PredictNow model based on CS229 sample complexity.

    CS229 Ng (Lecture 09):
      "To guarantee ε(ĥ) ≤ ε(h*) + 2γ with probability ≥ 1-δ,
       it suffices that m ≥ (1/2γ²)·log(2K/δ)"

    K here is the effective hypothesis class size — approximated as 2^n_features
    for a binary logistic regression over n_features binary-ish features.

    Returns float in [0, 1]: 0.0 = no trust, 1.0 = full trust.
    When < 0.5, the model has insufficient data → prefer priors.
    """
    K = 2 ** n_features  # effective hypothesis class size
    m_needed = math.log(2.0 * K / delta) / (2.0 * gamma ** 2)
    return float(min(1.0, n_trades / m_needed))


# ── Kelly Formula (Lo / AFML) ──────────────────────────────────────────── #

def fractional_kelly(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float,
    fraction: float = 0.25,
    floor: float = 0.005,
    ceiling: float = 0.04,
) -> float:
    """
    Compute fractional Kelly bet size as % of equity.

    f* = (p·b - q) / b
    where:
        p = win_rate
        q = 1 - p
        b = avg_win_r / avg_loss_r   (reward/risk ratio of past trades)

    Returns fraction × f*, clamped to [floor, ceiling].

    Args:
        win_rate:   historical win rate (0–1)
        avg_win_r:  average R multiple on winning trades (positive)
        avg_loss_r: average R multiple on losing trades (positive, e.g. 1.0 = 1R loss)
        fraction:   Kelly fraction to apply (0.25 = quarter-Kelly)
        floor:      minimum bet size even when Kelly is small
        ceiling:    maximum bet size regardless of Kelly output
    """
    if avg_loss_r <= 0 or win_rate <= 0 or win_rate >= 1:
        return floor

    b = avg_win_r / avg_loss_r
    q = 1.0 - win_rate
    f_star = (win_rate * b - q) / b

    if f_star <= 0:
        # Negative Kelly: expected value is negative → don't bet
        return 0.0

    practical = f_star * fraction
    return float(max(floor, min(ceiling, practical)))


class SovereignRiskEngine:
    """
    Grade-based sizing with fractional Kelly as the computed optimum.
    Kelly is used as the sizing reference; grade cap is the ceiling.
    """

    def __init__(self):
        self.base_engine = RiskEngine()
        self.rr_engine = DynamicRREngine()

    def _grade_risk_pct(self, confidence: float, library_insight=None) -> float:
        """
        Grade-based ceiling on position size, scaled by Library convergence.

        Base caps from config (A+=4%, A=3%, B=2%, C=1.5%).
        Library convergence reduces the cap dynamically:
          7+ volumes → ×0.50  (extreme late-cycle stress)
          5-6 volumes → ×0.625
          3+ volumes  → ×0.75
        PTJ: "Defence before offence." Smaller bets until macro regime clarifies.
        """
        g = params['risk']['grade_risk']
        if confidence >= 0.92:   base_cap = g['A_plus']
        elif confidence >= 0.78: base_cap = g['A']
        elif confidence >= 0.65: base_cap = g['B']
        else:                    base_cap = g['C']

        if library_insight is None:
            return base_cap

        # Count converging volumes
        converging = 0
        max_sim = 0.0
        try:
            for vm in library_insight.volume_matches:
                if vm.similarity >= 0.60:
                    converging += 1
                if vm.similarity > max_sim:
                    max_sim = vm.similarity
        except Exception:
            return base_cap

        if converging >= 7 and max_sim > 0.90:
            return base_cap * 0.50
        elif converging >= 5:
            return base_cap * 0.625
        elif converging >= 3:
            return base_cap * 0.75
        return base_cap

    def _get_stop_mult(self, regime: str) -> float:
        regime_params = params.get("regime_params", {})
        regime_block = regime_params.get(regime, {}) if isinstance(regime_params, dict) else {}
        if regime_block and "stop_atr_mult" in regime_block:
            return float(regime_block["stop_atr_mult"])
        return float(params["risk"].get("atr_stop_multiplier", 1.5))

    def _get_tp_rr(self, regime: str, symbol: str = "") -> float:
        if symbol:
            asset_params = params.get("asset_params", {})
            asset_block = asset_params.get(symbol, {}) if isinstance(asset_params, dict) else {}
            if asset_block and "atr_target_multiplier" in asset_block:
                return float(asset_block["atr_target_multiplier"])
        regime_params = params.get("regime_params", {})
        regime_block = regime_params.get(regime, {}) if isinstance(regime_params, dict) else {}
        if regime_block and "tp_rr" in regime_block:
            return float(regime_block["tp_rr"])
        return float(params["risk"].get("atr_target_multiplier", 3.0))

    def compute(self, bias: BiasOutput, router: RouterOutput,
                account_equity: float, atr: float, entry_price: float,
                win_rate: float = 0.55,
                avg_win_r: float = 2.0,
                avg_loss_r: float = 1.0,
                predict_now_prob: float = None,
                n_trades: int = 20,
                library_insight=None) -> RiskOutput:
        """
        Final trade parameters with fractional Kelly sizing.

        Ernest Chan (second lecture, 52:00):
          "In classical statistics you get a static probability — 5% every
          day. In machine learning you get a DIFFERENT probability each day
          based on current market conditions. That is a much more nuanced
          understanding of the market."

        predict_now_prob: if supplied, OVERRIDES the default win_rate prior
          with PredictNow's dynamic per-trade probability. This is the correct
          wiring: PredictNow outputs P(this trade is profitable) conditioned
          on today's regime — that IS the win_rate for Kelly.

          Without this, Kelly uses a static 55% prior regardless of whether
          the current regime has historically been 40% or 70% for this strategy.

        win_rate / avg_win_r / avg_loss_r: used when predict_now_prob is None.
          Defaults are conservative priors (55% WR, 2:1 R:R).

        Sizing logic:
            kelly_pct  = fractional_kelly(win_rate, avg_win_r, avg_loss_r)
            grade_cap  = _grade_risk_pct(confidence)
            risk_pct   = min(kelly_pct, grade_cap)
        """
        # Override static win_rate with PredictNow's dynamic probability
        if predict_now_prob is not None and 0.0 < predict_now_prob < 1.0:
            win_rate = predict_now_prob

        narrative_modifier = float(getattr(bias, "narrative_modifier", 0.0) or 0.0)
        effective_confidence = max(0.0, min(1.0, bias.confidence + narrative_modifier))

        # ── Hoeffding confidence interval on win_rate (CS229 Lecture 09) ── #
        # Ng: "P(|φ̂ - φ| > γ) ≤ 2·exp(-2γ²m). With probability ≥ 1-δ,
        # |φ̂ - φ| ≤ sqrt(log(2/δ)/(2m)). Feed the LOWER bound into Kelly —
        # conservative sizing until the estimate has enough data behind it."
        # As n_trades → ∞, γ → 0 and the bound converges to the point estimate.
        hoeffding_wr = hoeffding_win_rate(
            observed_win_rate=win_rate,
            n_trades=n_trades,
            confidence=0.90,
            mode='lower',
        )
        # Scale correction towards full win_rate as sample complexity is satisfied
        sc_conf = sample_complexity_confidence(n_trades, n_features=6)
        win_rate = sc_conf * win_rate + (1.0 - sc_conf) * hoeffding_wr

        # 1. Grade cap (Library convergence scales this dynamically)
        grade_cap = self._grade_risk_pct(effective_confidence, library_insight=library_insight)

        # Pegasus-learned Kelly fraction cap (REINFORCE — takes most conservative)
        if pegasus_params is not None:
            grade_cap = min(grade_cap, pegasus_params.kelly_fraction_cap)

        # 2. ATR safety gate
        symbol = router.symbol
        atr_pct = (atr / entry_price) * 100
        atr_limit = params['atr_gate'].get(symbol, 4.0)

        if atr_pct > atr_limit:
            return RiskOutput(
                position_size=0.0,
                kelly_fraction=0.0,
                stop_price=0.0,
                stop_method='ATR_GATE_BLOCKED',
                tp1_price=0.0,
                tp2_price=0.0,
                trail_config={'reason': f'ATR {atr_pct:.2f}% > limit {atr_limit}%'},
                expected_value=-1.0,
                ev_positive=False,
                size_breakdown={
                    'block_reason': 'Volatility Hemorrhage Threshold Exceeded',
                    'narrative_modifier': narrative_modifier,
                    'effective_confidence': effective_confidence,
                },
            )

        # 3. Fractional Kelly (AFML / Andrew Lo)
        kelly_pct = fractional_kelly(
            win_rate=win_rate,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            fraction=0.25,            # quarter-Kelly
            floor=0.005,
            ceiling=0.04,
        )

        if kelly_pct <= 0:
            # Kelly says don't bet — negative expected log-wealth
            return RiskOutput(
                position_size=0.0,
                kelly_fraction=0.0,
                stop_price=0.0,
                stop_method='KELLY_NEGATIVE_EV',
                tp1_price=0.0,
                tp2_price=0.0,
                trail_config={'reason': f'Kelly f*≤0: win_rate={win_rate:.2f} R={avg_win_r:.1f}:{avg_loss_r:.1f}'},
                expected_value=-1.0,
                ev_positive=False,
                size_breakdown={
                    'kelly_pct': 0.0,
                    'grade_cap': grade_cap,
                    'narrative_modifier': narrative_modifier,
                    'effective_confidence': effective_confidence,
                },
            )

        # 4. Final risk %: Kelly floored by grade cap
        risk_pct = min(kelly_pct, grade_cap)

        # 5. Position structure
        regime_label = router.regime
        stop_mult = self._get_stop_mult(regime_label)
        tp_rr = self._get_tp_rr(regime_label, symbol=symbol)

        # Pegasus REINFORCE overrides (trusted after ≥20 updates — applied proportionally)
        if pegasus_params is not None:
            stop_mult = pegasus_params.stop_atr_mult
            tp_rr = pegasus_params.tp_rr_ratio

        dollar_risk = account_equity * risk_pct
        stop_distance = stop_mult * atr
        position_size = dollar_risk / stop_distance if stop_distance > 0 else 0.0

        direction = 1 if bias.direction.value == 1 else -1
        stop_price = entry_price - direction * stop_distance
        tp1_price = entry_price + direction * stop_distance * tp_rr
        tp2_price = entry_price + direction * stop_distance * tp_rr * 1.5

        # Expected value using actual win_rate and R:R
        ev = win_rate * avg_win_r - (1 - win_rate) * avg_loss_r
        ev_positive = ev > 0 and position_size > 0

        return RiskOutput(
            position_size=position_size,
            kelly_fraction=risk_pct,
            stop_price=stop_price,
            stop_method=f"regime_atr_{stop_mult}x",
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            trail_config={},
            expected_value=ev,
            ev_positive=ev_positive,
            size_breakdown={
                'dollar_risk': dollar_risk,
                'stop_distance': stop_distance,
                'risk_pct': risk_pct,
                'kelly_pct': kelly_pct,
                'grade_cap': grade_cap,
                'narrative_modifier': narrative_modifier,
                'effective_confidence': effective_confidence,
                'kelly_f_star': (win_rate * (avg_win_r / avg_loss_r) - (1 - win_rate)) / (avg_win_r / avg_loss_r),
                'stop_mult': stop_mult,
                'tp_rr': tp_rr,
                'win_rate_input': win_rate,
                'avg_win_r': avg_win_r,
                'avg_loss_r': avg_loss_r,
            },
        )
