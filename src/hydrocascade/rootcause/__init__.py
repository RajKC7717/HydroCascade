"""Phase 3 — root cause analysis, susceptibility and external baselines (Doc 06).

Part 1 of the project. It answers *why does a hillslope fail here rather than
there*, and it produces three things Phase 4 depends on:

* ``S_L(u)`` and ``S_F(s)`` — **calibrated** susceptibility, used as a
  multiplier inside the background intensity. Doc 06 §6 makes calibration a
  hard requirement precisely because a multiplier that is systematically 1.4x
  too large corrupts every downstream number.
* ``e~(u)`` — the two-to-four standardised drivers that feed the receptivity
  term ``g_i``. Doc 06 §4 caps this deliberately: every extra driver costs
  identifiability in the cascade model.
* baselines **B1, B2, B3** — the models the cascade engine must beat.

Doc 06's opening line explains why the baselines are built here rather than
after the novel model: *"if you build baselines only after the novel model, you
will unconsciously tune the baseline to lose."*
"""
