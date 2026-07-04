# Plan — TICK-003: options-leg family run (interim seals)

Authorized by the approved Day-2 operating plan:
`Plans/context-day-2-imperative-stonebraker.md` §E2/§E3. Ticket ACs govern.

1. Extend `scripts/research/run_positioning_family.py` with the options legs:
   - rr25_z / bf25_z = trailing-252-obs z on the weekly surface series;
     truncation-invariance test (same standard as cot features).
   - HYP-076 = econ_surprise_z × crowding interaction.
2. Gate-zero hash verify + 0.6886 reconcile guard must hold before ANY seal.
3. Run primaries HYP-074/075/076/078/079 + HYP-077 FULL composite (supersedes the
   COT-only interim; both annotations remain).
4. Seal as dated interim ANNOTATIONS; statuses stay PREREGISTERED;
   UNDERPOWERED/BLOCKED stamps where data forbids; coverage stamp
   ("options history 2020-01-03+") on every options seal.
5. HYP-080: GDELT unblock per §E3 (serial paced re-run, existing creds) → board
   rebuild → look-ahead auditor 0 violations → primary.
6. Family BH ONLY when all 10 primaries exist, exactly per the locked manifest
   `data/research/preregister/HYP-072-081_positioning_family.json`; otherwise stamp
   the blocker. An all-null family = the VISION kill-criterion firing — write it
   plainly.

Constraint: no param re-optimization after seeing any result; no improvised partial
adjudication.
