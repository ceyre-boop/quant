# The AlphaZero and Stockfish Report
## Alta Investments · 2026-07-23
### Written by the thinking layer — goals, expectations, honest assessment

---

## What These Names Actually Mean

Before anything else: AlphaZero and Stockfish are not chess software in this system.
They are the names of two fundamentally different kinds of intelligence, and the entire
architecture of Alta Investments rests on keeping them separate.

AlphaZero, the chess program, learned to play by playing against itself millions of
times. It never studied human games. It developed intuition from pattern — from seeing
what kinds of positions tend to lead to what kinds of outcomes. It became the best chess
player in the world not by calculating further than Stockfish but by *reading the board
better*. It knew which positions were promising before running a single line.

Stockfish is the opposite. Given any position, it calculates with surgical precision.
It doesn't guess. It evaluates. It looks at the current state of the board and computes,
to a very high degree of certainty, what the correct move is. No intuition. Pure
deterministic evaluation.

The insight that gave birth to your architecture is this: trading requires both, in
sequence. You need something to read the market and say "this looks like a day where
something is about to happen." Then you need something to say "given that we're in a
trade, here is the mathematically correct thing to do right now." Prediction first.
Evaluation second. These are not the same problem and they should not be solved by the
same machine.

---

## The AlphaZero Half — The Briefing Synthesizer

### What It Is

The briefing synthesizer (`sovereign/briefing/synthesize.py`) is your AlphaZero. It
takes in five data streams — market state, the NQ/ES lead-lag regime, volume profile,
tagged news headlines, and an event calendar — and makes a single Claude Opus API call
that produces a structured judgment: `LONG`, `SHORT`, or `NEUTRAL`, with a confidence
score, a regime classification, an invalidation level, and a full narrative explaining
the reasoning.

It was built thoughtfully. The prompt forces it to grade its own confidence, to state
the bias as a probability rather than a prediction, and to never manufacture certainty
when signals are mixed. It costs roughly two cents per run. It logs what it spends.
It handles failures gracefully — if the API is down or the parse fails, it returns None
and the orchestrator falls back to a deterministic synthesis. It was built to be
integrated into something larger.

### What It's Actually Doing Right Now

Nothing. It is producing output that nobody reads.

The function exists. The code is correct. The API call works. But there is no downstream
consumer. No code reads the `directional_bias`, `confidence`, or `narrative` it returns.
No dashboard panel shows it. No gate uses the confidence score as a multiplier. No
decision log captures it. It is an AlphaZero that has learned to read the board
perfectly and then sits in a room where no one plays chess.

### What I Want From It

I want the briefing synthesizer to become the first question the system asks every
morning, and I want its answer to flow into everything that happens after.

Here is what I mean concretely. The synthesizer produces `confidence: 72` and
`directional_bias: LONG`. That number — 72 — should mean something to the rest of the
system. When confidence is high and direction aligns with the carry signal, size
slightly larger. When confidence is low or direction opposes, size slightly smaller, or
wait. When the regime call is `ROTATION_WARN`, treat all signals as lower quality until
it resolves. The `key_level` — the invalidation price — should appear on the dashboard
so that when price approaches it, the system flags it.

More importantly: I want a scorecard. The synthesizer has a `scorecard` parameter —
a summary of its own track record that gets fed back into every future call. Right now
that scorecard is empty because nothing has been tracking its calls and their outcomes.
Once the synthesizer is wired, every morning call gets logged. When the trade resolves,
the outcome gets matched back to the call. Over time the scorecard fills in. The
synthesizer reads its own history and becomes more calibrated. That is the self-improving
loop. That is the AlphaZero analog — it gets better by playing, not by being manually
updated.

The expectation is not that it will predict direction reliably. The hypothesis that
daily-bar directional prediction on G10 forex is consistently null has been confirmed
across seven tested hypotheses. I am not expecting the synthesizer to break that pattern.
What I expect it to do is provide *regime context* — the NQ/ES lead-lag regime, the
macro fear state, the volume profile read — that improves the *quality* of sizing
decisions even when it cannot predict direction. A trader who knows they are in a
high-volatility rotation regime sizes differently than one who doesn't, even if they
can't say which way it goes. That is the edge I believe is in here.

### The Wiring Required

The synthesizer needs three connections that don't currently exist:

1. **It needs to be called.** The DIP Phase 1 or Phase 2 script should call it every
   morning and write the result to `data/agent/daily_briefing.json`.

2. **Its output needs to feed the conviction scorer.** The `confidence` score should
   become a multiplier input to the Petrules Gate and the carry sizing engine. Not a
   veto. A continuous multiplier, like the bias multiplier in the ICT pipeline.

3. **Its calls need to be logged and tracked.** Every morning's call goes to
   `data/agent/briefing_log.jsonl`. Every trade that follows gets the session's briefing
   appended to its decision log entry. When the trade closes, the outcome gets matched
   to the briefing. Over time, the scorecard becomes real data, and the synthesizer
   becomes calibrated to this specific system.

---

## The Stockfish Half — The Exit Value Function (HYP-071)

### What It Is

HYP-071 is the most rigorous piece of research in this entire system. It is also
completely unbuilt. The pre-registration is locked, hash-verified, and frozen as of
June 30th. The experiment is designed. The methodology is bulletproof. The value table
itself has never been computed.

The concept is this: once you are in a trade, the directional bet is already placed.
The only question remaining is not *which way* but *what to do right now* — hold, trail,
or exit. That is a different problem from prediction. It is an evaluation problem. It
should be solved by something that looks like Stockfish: not intuition, not pattern
recognition, but pure calculation over a well-defined state space.

The state space is a board with 108 cells, defined by five dimensions: how volatile the
market is right now (ATR percentile), how much unrealized profit the position has
(excursion state), how far into the expected hold window we are (hold fraction), whether
momentum is overextended (RSI extreme flag), and whether the carry signal still agrees
with the direction of the trade. Every bar, the position is in exactly one of these
cells. The question for each cell is: in positions like this one, given 10,000
resampled continuations of what the market might do next, does holding produce a better
risk-adjusted outcome than exiting immediately?

The answer is computed not from prediction but from the historical distribution of what
actually happened in similar states. That is what makes it legitimate. It is not asking
"where will price go?" It is asking "given where we are, what does the data say to do?"
That is a Stockfish question, and it has a Stockfish-style answer: a lookup table that
covers every state the position can be in.

### What The Pre-Registration Actually Says

The honest expectation recorded in the pre-registration is that this will not find
anything. It says, in its own words: "NOT_SIGNIFICANT — likely the 4th confirmation of
the data-ceiling thesis." The researchers registered it anyway because the answer is
only trustworthy if it was asked before anyone looked at the data.

I respect that honesty completely. But I also think there is something real here that
the data-ceiling thesis does not fully explain. The ceiling thesis says that the
directional edge at the daily-bar level is weak or nonexistent. HYP-071 is not testing
a directional edge. It is testing whether the *timing of the exit* within a proven
directional trade has structure that the current static rules miss. Those are different
questions. The carry edge is structural, not predictive — it pays because of interest
rate differentials, not because we can read direction. Given that the edge is structural,
the exit timing matters more than directional forecasting, and the tabular approach is
exactly the right tool to find it.

What I expect to find, if the table has any real structure: that late-hold, high-ATR,
extended-excursion positions should exit faster than the current static rules allow.
That is economically sensible. When volatility spikes, when the position is deeply in
profit, and when we are near the end of the expected hold window, the right move is
probably to take it. The current static config does not differentiate by these states.
The table would.

### What I Want From It

I want the table computed. Step 2 of this experiment — the actual value computation —
has been waiting since June 30th for approval. The pre-registration is locked. The
methodology is sound. The backtester that generates the 10,000-continuation Monte Carlo
rollouts already exists and runs at 1.26 million trades per second. The computation
would probably take hours on the Mac, not days.

If the table passes validation — reconciles against the known Sharpe, shows CPCV-stable
structure, agrees across the 2023-24 and 2025-26 windows — then it becomes a set of
targeted rule additions to the exit machine. Not a neural network, not a rewrite. A
small number of specific cells where the data says the current rule is suboptimal, and
a concrete change to that rule that the data supports. That is the Stockfish half doing
what Stockfish does: not guessing, calculating.

If it fails — if the structure is in-sample only, or flips across the forward window —
then we have definitively confirmed that the exit edge is entangled with the directional
entry edge, and the unlock is new data or a different architecture, not a cleverer exit
rule. That is also a useful answer. Either way the experiment resolves something real.

The exit machine is currently in shadow audit, go-date July 28th — five days from now.
If that shadow passes clean, the exit machine goes live. At that point, Step 2 of
HYP-071 becomes the natural next research question: we have a live exit machine, we have
the backtester, we have the pre-registration. Computing the table and comparing it to
the live rules becomes the clearest possible test of whether we are at the ceiling or
whether there is still structure to find.

---

## The Relationship Between the Two

AlphaZero sets the context. Stockfish makes the decision.

In the morning, the synthesizer reads the market and says: high-confidence long-leaning
day, NQ leading ES, accumulation at 22,400, rotation warning. That feeds into sizing.
Larger when aligned, smaller when opposed, flat when the regime is clearly wrong.

Once the trade is open, the synthesizer's job is done. The exit value function takes
over. Given the current cell state — mid-hold, modest excursion, moderate volatility,
RSI not extreme, carry signal still aligned — the table says: hold. Two days later —
late-hold, extended excursion, high ATR, RSI extreme, carry signal still aligned — the
table says: exit now, before the trailing stop gives back a third of the profit.

The two machines never compete. They operate in sequence, on different problems, with
different tools. The synthesizer reads pattern. The value function calculates. That is
the architecture, and it is right.

---

## The Honest Priority Order

**AlphaZero first.** The synthesizer is already built. The wiring is three connections:
call it, feed its output downstream, log the results. This should take days, not weeks.
And once it starts logging, it starts learning. The scorecard fills in. The AlphaZero
half begins to become what it was designed to be.

**Stockfish second.** HYP-071 Step 2 is the right next research question after the L2
exit machine goes live on July 28th. The pre-registration is locked. The computation
is ready to run. Approve Step 2, run the table, let the data answer the question.

Both of these are achievable this month. Neither requires new infrastructure. The
AlphaZero half needs wiring. The Stockfish half needs approval and compute time.

The system that results — a briefing synthesizer that sets morning context and gets
scored on its own track record, feeding an exit machine whose rules were derived from
10,000-continuation Monte Carlo rollouts of the actual backtester — is the most
intelligent trading system we can build right now from what exists.

Everything required to build it is already here.

---

*Alta Investments · AlphaZero-Stockfish Assessment · 2026-07-23*
*Written by the thinking layer.*
*"AlphaZero reads the board. Stockfish calculates the move. Neither guesses."*
