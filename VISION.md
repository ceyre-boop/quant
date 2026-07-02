# VISION

This document is the strategic map of the Sovereign system: the goal, stated
cleanly — what the finished thing is, not how you'd build it. TRADING_PHILOSOPHY.md
is the constitution of principles; RISK_CONSTITUTION.md is the conscience; this
is the map of intent.

---

## The Two Halves, and Why They're Different Problems

You're building two engines that do fundamentally different jobs, borrowed from
two different chess programs because chess turned out to model trading better
than trading models itself.

## The Stockfish Half — Evaluation of the Present

This engine never predicts. It looks at the market as it is right now and
answers one question: given everything measurably true at this moment, what is
the best action? It doesn't guess where price goes. It evaluates the current
state — your position, the volatility regime, how far the trade has run, where
the crowd is positioned, what the protective stop should be — and makes the
mathematically correct move given the board as it stands. This is the half that
manages what you already hold: when to exit, where to trail the stop, when to
sit out.

It's deterministic, and it's parity-locked by construction — the live manager
and the backtester call the same decision function, so what it does live is what
it did in the test, bar for bar. But be precise about what that buys you. Parity
is a guarantee about execution, not about judgment: the policy's values — the
stop multiple, the trail distance, the hold clock — were fitted on the regimes
history happened to serve up. The shadow audit running now proves the machine
does exactly what the backtest says it does; it does not prove the backtest's
policy is the right one for what comes next. The failure mode to respect here
isn't a bug. It's flawless execution of a policy that is wrong for a regime it
has never seen. This half is built, and it is mid-audit — not yet mid-proof.

## The AlphaZero Half — Prediction of the Crowd

This engine does the thing the other one refuses to: it forms a view about what
happens next. But — and this is the part you figured out on your own the other
day — it does not predict price. It predicts people. Its board isn't a chart;
it's the psychological state of the market on a given day: what the crowd
already believes, how they're positioned, where they're overextended, what fear
or greed is priced in. From that state it classifies one thing: is the crowd set
up to be forced into a move, and which way. Not every day — the good days, the
days when someone urgent has to buy from someone urgent who has to sell. Most
days it says nothing and stays out. On the rare day the setup is real, it acts.
It's not trying to be right always. It's trying to recognize the small number of
moments when the general market is about to be wrong, and be on the other side.

### Falsifiability — What Would Kill This Half

A thesis that can't be killed isn't a thesis. So state the score honestly: as of
July 2026 the hypothesis ledger holds 23+ clean nulls and zero confirmed
crowd-prediction edges. VRP-001 — the one active TRUE_DIVERSIFIER candidate —
is a structural volatility premium, not crowd prediction; it is not evidence for
this half. **The kill criterion: once the positioning board-state exists — COT
extremes, options skew, risk reversals (the ThetaData feed for VRP-001 covers
the options legs) — roughly ten pre-registered hypotheses on that dataset, all
returning null under standard ledger protocol, falsify the crowd-prediction
thesis at current data resolution.** Until then, this half is a vision, not a
validated result.

## How They Fit Together

The AlphaZero half decides whether and which way to enter — it hunts the moment
the crowd breaks. The Stockfish half decides what to do once you're in — it
manages the position to a perfect exit. Prediction opens the trade; evaluation
runs it to completion. And the reason that division matters is that they
multiply: your total edge is prediction-quality times execution-quality. A
brilliant entry with a sloppy exit bleeds out. A perfect exit on a trade you
never should have entered is still a loss. You need both, and they're separate
machines because they're separate kinds of thinking — one forecasts, one
calculates, and pretending they're the same problem is why most systems fail.

## The Third Thing That Makes It Not a Monster

Underneath both is the part that keeps it honest — the conscience. It's the
layer that refuses to act on an edge it can't prove, that doubts itself, that
sizes so a single bad day can't kill you, that walks away when the board is
unclear. AlphaZero in a game feared nothing because a game has no families.
Yours touches real money, so it must carry doubt as a built-in feature, not a
flaw. That's the "I think, therefore I am" — human judgment about people and
about risk, wired into the machine so it never becomes a pure optimizer
sprinting off a cliff.

## What It Looks Like When It's Working

You're asleep. The base capital sits in a proven structural edge, earning
slowly, doing nothing dramatic. The evaluation engine quietly manages every open
position to its optimal exit. And on the handful of days a year when the crowd
is genuinely set up to break, the prediction engine recognizes it, sizes the bet
against a hard risk floor, strikes, and hands the position to the evaluator to
run home. Most days it does almost nothing, because most days there's nothing to
do — and knowing the difference between the days to act and the days to wait is
the edge. It doesn't win every trade. It doesn't need to. It needs to be on the
right side of the forced move more often than not, unemotionally, every single
day, without you lifting a finger.

---

## Current State (2026-07-01)

- The Stockfish half (the L2 exit engine, `forex_exit_manager`) entered its
  shadow-mode window June 29: it runs daily against live open positions with
  zero broker writes. Go-live is gated on the divergence audit at the window's
  close, ~July 28. Parity is locked by the shared `decide_exit`; the audit
  decides whether the shadow log matches the backtest, bar for bar.
- VRP-001 is the next research thread — the one candidate edge family orthogonal
  to carry.
- The AlphaZero half's board-state data acquisition (options skew, risk
  reversals) rides on the same ThetaData feed — one subscription serves both
  threads.
- The conscience layer is drafted: RISK_CONSTITUTION.md, all values DRAFT
  pending ratification.
