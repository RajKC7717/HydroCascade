# 16 — Demo flow

How to show HydroCascade to someone, in the order that makes it land. Two
versions: a **3-minute** version for a corridor conversation, and a **12-minute**
version for a review panel. Both drive the same screen — `hydrocascade app`.

Before you start, read §0. Demoing this project while misrepresenting what is
real would undo the one thing it is built to protect.

---

## 0. Start the frontend, and know what is synthetic

```bash
cd C:\Users\choud\Desktop\TY\edisem1
.venv\Scripts\activate
hydrocascade app
```

It opens `http://localhost:8501` in a browser. Leave it running; every step
below is a tab in that window.

**The red banner in the sidebar is part of the demo, not an apology.** It names
every dataset currently standing in for real data. At the time of writing it
names two:

* `landslides` — the inventory is a labelled synthetic one, because no real
  inventory has been placed in `data/raw/` yet.
* `glofas` — flood events come from a synthetic discharge series.

So say this once, early, in your own words:

> "The spatial framework, the rainfall, the terrain and the whole pipeline are
> real. The landslide inventory and the river discharge are labelled stand-ins,
> because those two need a download I haven't done yet. Every number derived
> from them is marked, and the quality gates refuse to pass on them. What I'm
> showing you is a working machine, not a finished result."

Saying it up front converts your biggest weakness into the thing that makes the
rest of the demo credible. Saying it when someone catches it does the opposite.
[15_HOW_TO_RUN.md](15_HOW_TO_RUN.md) §3 is exactly what unblocks each one.

---

## 1. The 3-minute version

| # | Tab | What you do | What you say |
|---|---|---|---|
| 1 | **The cascade path** | Point at the flow-path figure and the four metrics | "A landslide at Mundakkai doesn't raise flood risk everywhere at once. It raises it *on the reaches its debris can actually reach*, *after the travel time of the water*, and *diluted as the catchment grows*. That's the whole idea, and this page is the proof the geometry computes it." |
| 2 | **Spatial framework** | Point at the two supports | "Landslides live on 17,138 2-D slope units. Floods live on a 1-D network of 9,038 river reaches. They are different spaces — so the model needs an operator that maps one into the other, and that's π, computed by following the D8 flow path downhill." |
| 3 | **Backtest & alerts** | The money figure | "This is the honest test: refit on data strictly before June 2024, roll forward on observed rainfall only, and ask whether it would have called Wayanad. Lead time and false-alarm rate are always shown together." |
| 4 | **Gates & reports** | Scroll the gate table | "Every claim in this project has an executable check behind it, and they report three states, not two. BLOCKED means *couldn't check* — it is never counted as a pass." |

Then stop. The natural next question is "so did it call Wayanad?" — and §2 step
6 is the honest answer.

---

## 2. The 12-minute version

### Step 1 — Frame the problem (1 min, no screen)

> "On 30 July 2024, at 02:17, a landslide at Mundakkai in Wayanad killed over
> 200 people, and the debris reached the Chaliyar river and pushed a flood
> downstream. Landslide warnings and flood warnings in India are produced by
> different systems that don't talk to each other. So the flood system had no
> way to know a landslide had just loaded the channel upstream. This project
> models the two hazards as **one process** instead of two."

### Step 2 — "The cascade path" tab (2 min)

Show the Wayanad flow-path figure. Then the four metrics: injection reach,
crown-to-channel distance (**0.68 km**), number of downstream reaches, maximum
travel time.

Scroll to the downstream table and point out two columns:

> "Travel time increases monotonically as you go downstream, and contributing
> area grows. That's not a modelling assumption, it's what a correct network
> must do — and it's the spot check that catches a broken flow direction
> immediately."

### Step 3 — "Spatial framework" tab (1.5 min)

Slope units and the channel network, the hybrid support, and π.

> "The mathematical novelty is here. A standard Hawkes process needs one space.
> We have two: hillslopes and a river network. The kernel therefore has a
> runout gate — can the debris physically reach a channel — a downstream gate,
> a dilution term and a travel-time shift. Each of those is a term someone can
> switch off and test, and later I'll show you exactly what happens when we do."

### Step 4 — "Rainfall" tab (1.5 min) — *your most interesting failure*

Show the Wayanad reconciliation numbers: published **577 mm** over 48 h,
CHIRPS **71 mm**.

> "CHIRPS sees 12% of the rain that actually fell. That's not an extraction bug
> — the storm band peaks about 50 km west of the crown, which is the known
> orographic displacement at quarter-degree resolution. This gate **fails**, and
> it's reported as failing. It's also the single highest-value fix in the
> project: a gauge-based IMD grid would see the storm, and the code is already
> wired for it — one config line plus the files."

This is the moment a sceptical reviewer starts trusting the demo.

### Step 5 — "The model" tab (2.5 min)

Start with the verification metrics, *not* the Kerala fit:

> "Before fitting anything to Kerala, the likelihood has to recover parameters
> from simulated catalogs whose truth we set ourselves. S1 and S2 pass, and
> three independent implementations of the likelihood agree to machine
> precision."

Then the fit table (every fit on record, with seed and commit), then open the
**ablations** expander:

> "This table is the scientific core. Each row removes exactly one idea and
> refits everything else, so the change in held-out likelihood is attributable
> to that one component. Some of our ideas don't earn their place — and those
> rows are reported too, which is the point."

Two specific numbers worth naming out loud, because they are the honest ones:

* **A3 (travel time removed) and A2 (network → isotropic) both *improve* the
  held-out likelihood.** On a synthetic catalog, with a rainfall product that
  can't see the trigger, that is the expected result: there is no real cascade
  timing in the data to recover yet. Say so.
* **S4 beats S3 on held-out likelihood by about 800**, which is the cross-hazard
  coupling paying for itself.

### Step 6 — "Backtest & alerts" tab (2.5 min)

> "Refit strictly before 1 June 2024, roll forward day by day on observed
> rainfall, and record when each alert level was crossed. The leakage guard
> raises an exception rather than warning, and there is a test that deliberately
> tries to break it."

Then the honest headline:

> "It does **not** produce a useful lead time, and I can tell you exactly why:
> the model's background rises with rainfall, the rainfall product sees 71 of
> 577 mm, and the landslide catalog is synthetic. This backtest is currently
> measuring the data, not the model. Fix the rainfall and the inventory and this
> is the number that changes."

Then the trade-off figure: lead time against false alarms at cost ratios 1:1,
5:1 and 20:1.

### Step 7 — "Gates & reports" tab (1 min)

Scroll the table and land on the three-state design:

> "PASS, FAIL, BLOCKED. Most projects collapse that to two and lose the
> distinction between 'we checked and it's fine' and 'we couldn't check'. That
> distinction is how a synthetic number ends up on a slide, so here it's
> structural."

### Step 8 — Close (30 s)

> "Six phases, all built and runnable end to end on one machine. What's missing
> is two downloads, not two months of work — and the repo tells you exactly
> which two, where they go, and what each one unblocks."

---

## 3. Questions you will be asked, and the honest answer

| Question | Answer |
|---|---|
| "Is any of this real data?" | The DEM, the basin delineation, the river network, the slope units, the travel times and the CHIRPS rainfall are real. The landslide inventory and the river discharge are labelled stand-ins. The banner says so on every screen. |
| "So does it predict Wayanad?" | Not yet, and the report says why rather than hiding it: the rainfall product can't see the storm and the catalog is synthetic. Both are fixable with downloads that are already wired in. |
| "Why do some ablations show your ideas don't help?" | Because on this data they don't, yet — and reporting that is the only thing that makes the rows where they *do* help believable. |
| "What's actually new here?" | Two hazards on two different supports (2-D hillslopes, 1-D river network) in one point process, with a projection operator between them, and physically interpretable kernel terms — runout gate, dilution, travel-time shift — that can each be switched off and tested. |
| "Could it run operationally?" | The architecture report sets a T+4 h latency budget and the predictor evaluates 2,040 days in 0.6 s. The blocker is data latency, not compute. |
| "How do I know a number in a report came from a real run?" | `experiments/register.csv` has one row per fit: config hash, seed, likelihood, branching ratio. Each fit record also carries the git commit. |
| "What would you do next, with one week?" | Download the IMD rainfall and the Kerala 2018 inventory, re-run, and see whether the ablations change sign. That is the experiment the whole build exists to make cheap. |

---

## 4. If the app will not start

| Symptom | Fix |
|---|---|
| `hydrocascade: command not found` | `.venv\Scripts\activate`, then `pip install -e .` |
| Port 8501 already in use | `hydrocascade app` again after closing the other tab, or `streamlit run src/hydrocascade/dashboard/app.py --server.port 8502` |
| A tab says "run `hydrocascade experiments`" | That phase hasn't been run in this checkout. See [15_HOW_TO_RUN.md](15_HOW_TO_RUN.md) §2. |
| Everything is slow on first click | Each tab loads its data once and caches it. The second visit is instant — click through all nine tabs once before the demo starts. |
