"""Phase 6 — alerts, the real-time architecture and delivery (Doc 09).

Doc 09 §1.1, the honest scope statement, repeated here because it governs
everything in this package:

    **What we deliver:** a validated model, a fully specified deployment
    architecture, and a working replay-mode dashboard that reproduces the
    Wayanad event as if live.

    **What we do not deliver:** a running, monitored, 24/7 production service.

And the constraint that shapes the whole design (Doc 09 §1.2): GPM IMERG Early
Run is available at **T + 4 hours**, so rain-to-updated-risk is about 4 h 15 m.
**This is a nowcast layer, not a siren.** Wayanad failed at 02:17 IST; rainfall
from 22:00 the previous night would have reached the system around 02:00. The
value is in the hours-to-days of rising signal beforehand, not in the final
minutes — and that is the first thing an operational agency asks.
"""
