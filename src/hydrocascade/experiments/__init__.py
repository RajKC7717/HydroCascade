"""Phase 5 — experiments, ablations and validation (Doc 08).

Doc 08's opening: *"This is the phase that turns a working model into a research
result. It is also the phase most often skipped, which is why so many student
projects end with one number and no argument."*

What lives here:

* :mod:`~hydrocascade.experiments.splits` — the three-way temporal split and the
  **leakage assertion**, which raises rather than warns.
* :mod:`~hydrocascade.experiments.register` — ``experiments/register.csv``. No
  number appears in the paper without an entry.
* :mod:`~hydrocascade.experiments.ablations` — A1-A9, the nine-row table Doc 08
  §3 calls *"the scientific core of the paper"*.
* :mod:`~hydrocascade.experiments.ladder` — the nested M0-M5 ladder and the
  likelihood-ratio tests that go with it.
* :mod:`~hydrocascade.experiments.loco` — leave-one-cascade-out, the answer to
  *"you designed the model around Wayanad and then tested it on Wayanad."*
* :mod:`~hydrocascade.experiments.backtest` — the Wayanad protocol and the money
  figure.
* :mod:`~hydrocascade.experiments.uncertainty` — profile likelihood for anything
  we make a claim about, because Hessian intervals are optimistic here.
* :mod:`~hydrocascade.experiments.transfer` — apply a fitted model to a second
  region without refitting.
"""
