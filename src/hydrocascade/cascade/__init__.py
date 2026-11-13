"""Phase 4 — the cross-hazard cascade model (Doc 07).

Built in five stages, each verified before the next begins:

===== ======================================== ==========================================
Stage What is added                            Verification before proceeding
===== ======================================== ==========================================
S1    1-D Hawkes in time only                  recover known parameters from simulation
S2    space added, single hazard type          recover known parameters in space-time
S3    two hazard types, isotropic, real data    converges; branching ratio < 1
S4    hybrid support: network, projection       beats S3 held out, or the reason is known
S5    travel-time shift + mechanism mixture     beats S4, or components are honestly dropped
===== ======================================== ==========================================

Doc 07's opening paragraph is the rule that governs the whole phase:

    *never add a component until the previous one has been verified on data
    where you already know the answer. If you go straight to the full model and
    it produces nonsense, you will not know whether the bug is in the
    mathematics, the code, the spatial framework, or the catalog. That
    uncertainty is what kills projects in week thirteen.*

So :mod:`hydrocascade.cascade.simulate` exists before
:mod:`hydrocascade.cascade.likelihood` has anything to fit, and the parameter
recovery tests in ``tests/test_cascade.py`` are the gate on everything after.
"""
