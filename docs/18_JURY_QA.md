# 18 — Questions the jury will ask, and short answers

Every answer is two lines or so, in plain words. Numbers here are the real ones
from the current run, so you can say them with confidence.

**The one rule for the whole viva:** never defend a weakness you can explain.
This project's strongest card is that it knows exactly what is wrong with it.

---

## A. The basics

**Q1. What is your project, in one sentence?**
It predicts landslides and floods as **one connected chain** instead of two separate hazards.
When a hillside collapses into a river, it works out which places downstream will flood, and when.

**Q2. What problem does it solve?**
In India, landslide warnings and flood warnings come from different agencies that don't talk to each other.
So when a hill collapsed at Wayanad in July 2024, the flood system downstream had no idea mud was coming.

**Q3. Why does that matter?**
On 30 July 2024, at 2:17 in the morning, a landslide at Mundakkai killed over 200 people and pushed debris into the Chaliyar river.
The flood that followed was a predictable consequence of the landslide — if anything had been watching both.

**Q4. Where did you study?**
The Chaliyar river basin in Kerala, about 3,000 km², containing Wayanad.
We chose it because the 2024 disaster there is exactly the kind of chain we model.

**Q5. What is actually new here?**
Landslides happen on a **2-D surface** (a hillside) and floods happen along a **1-D line** (a river). Nobody had put both in one model.
We built the bridge between those two kinds of space, and made every physical assumption switchable so it can be tested.

---

## B. The data

**Q6. What data did you use?**
Three real datasets: Copernicus satellite elevation, HydroBASINS river basins, and CHIRPS daily rainfall. All free, all downloaded with receipts.
Plus the event list — and that one is mostly synthetic. (See Q7.)

**Q7. How much of your data is real?**
The **place** is completely real: terrain, rivers, rainfall, 17,138 hillside patches, 9,038 river reaches.
The **event history** is not: 9,764 of 9,765 events are computer-generated stand-ins. Only the Wayanad landslide itself is real.

**Q8. Why is your event data synthetic?**
A real landslide inventory needs a download we haven't done, and flood events need river-flow data that needs an account.
So we wrote labelled stand-ins to test the machinery — and every single one is marked so it can never be mistaken for real.

**Q9. Isn't that a fatal weakness?**
It limits what the numbers mean, and we say so on every report and every screen.
But it is two downloads away from being fixed — the readers are written and tested, so it's an afternoon, not a semester.

**Q10. How do you stop a fake number being reported as real?**
Every synthetic row is tagged, the tag follows it into every report and the app's sidebar, and the quality gates refuse to pass on it.
That's why gate P1.1 says BLOCKED instead of PASS.

**Q11. How big is the dataset?**
About 360 MB in total — 178 MB downloaded, 165 MB computed from it.
It runs on one ordinary laptop; no cloud, no cluster.

---

## C. The method

**Q12. What kind of model is it?**
A **Hawkes process** — the maths used for earthquake aftershocks and disease spread, where every event makes more events likely nearby and soon after.
We extended it so one hazard can trigger a different hazard, in a different kind of space.

**Q13. Why a Hawkes process and not deep learning?**
We have hundreds of events, not millions — deep learning would just memorise.
And every one of our 20 parameters means something physical you can point at, which a neural network can't give you.

**Q14. What does the model actually take in?**
For landslides: steepness, shape, natural wetness, and rain over 1, 3 and 7 days.
For floods: catchment size and slope, plus rain averaged over everything draining into that stretch.

**Q15. What are the physical ideas inside it?**
Four: debris must physically reach a channel (runout gate), effects travel **downstream only**, they weaken as the catchment grows (dilution), and they arrive **after a delay** because water takes time to travel.
Each one can be switched off and tested.

**Q16. What is π (the projection)?**
It's the rule that says where a hillside drains into the river: follow the water downhill until it hits a channel.
It's how a point on the 2-D hill becomes a point on the 1-D river.

**Q17. Why two timescales?**
Because two different things happen. Debris hits the river in about **1 hour**; the weakened ground stays dangerous for about **14 days**.
One timescale would blur two real mechanisms into one wrong average.

---

## D. The models and accuracy

**Q18. Which machine-learning models did you use?**
Random Forest, XGBoost and logistic regression for "which hills are dangerous", plus isotonic calibration to make the percentages honest.
Then the Hawkes cascade model on top, for "what happens next".

**Q19. What is your accuracy?**
Landslides: AUC-PR **0.510**. Floods: AUC-PR **0.617**. But these describe the synthetic stand-ins, not Kerala.
Calibration error dropped from 0.084 to **0.0096**, meaning when it says 10%, about 10% really happen.

**Q20. Why AUC-PR and not accuracy?**
Landslides happen maybe 1 day in 500. Saying "no landslide" every day scores 99.8% accuracy and is useless.
AUC-PR asks the question that matters: when we raise an alarm, how often are we right?

**Q21. What is calibration and why do you keep mentioning it?**
It means the percentages are honest — "10% chance" should come true about 10% of the time.
It matters because that number gets multiplied through everything downstream; if it's 40% too high, every later number is wrong.

**Q22. Did you compare against simpler methods?**
Yes — three baselines, and we built them **first**, before our own model existed, so we couldn't unconsciously tune them to lose.
One of them (the classic rainfall threshold) is what engineers actually use today.

---

## E. The results

**Q23. Does the cross-hazard idea work?**
Yes. Removing the landslide→flood link makes held-out fit worse by **188**, so it earns its place.
Turning on the river network improves it by **802** over ignoring the network.

**Q24. Would it have warned people before Wayanad?**
**No — and we report that.** It never raised a single alert.
The reason isn't the model: our rainfall data recorded only 71 mm of the 577 mm that actually fell.

**Q25. Then what does work?**
The cascade half. Once the landslide is entered, the model names the downstream reach, raises its flood risk, separates out how much is **caused by the landslide**, and predicts arrival in **19.2–38.5 hours**.
The "will it happen" half failed; the "what happens next" half works.

**Q26. Why couldn't your rainfall data see the storm?**
CHIRPS is a satellite product that guesses rain from cloud pictures, and mountains confuse it.
It put the storm about 50 km too far west. Gauge-based IMD data would fix it — the code is already written for it.

**Q27. Did any of your own ideas fail?**
Yes, three. Dilution came out as exactly zero. Flood→landslide triggering came out as basically zero. And the fitted river wave speed is physically implausible.
We report all three, because a model where every idea conveniently works is one nobody should believe.

**Q28. How do you know the model isn't cheating?**
Before fitting anything to Kerala, we made the computer generate fake data whose answers **we** chose, then checked the model could recover them.
Also three independent implementations of the maths agree to 15 decimal places.

**Q29. What about data leakage?**
The model is refitted using only data from before June 2024, and the check **crashes the program** rather than printing a warning.
There's a test in the suite that deliberately tries to sneak future data past it.

---

## F. Engineering and quality

**Q30. How do you know your code is correct?**
**147 automated tests**, and 35 quality gates that check every claim the project makes.
Currently 29 pass, 2 fail, 4 are blocked.

**Q31. Why do you have a "BLOCKED" status?**
Because "we checked and it's fine" and "we couldn't check yet" are completely different things.
With only pass/fail, "not tested" quietly becomes "passed" — that's exactly how a fake number ends up on a slide.

**Q32. Which gates fail and why?**
Two. The rainfall one, because CHIRPS can't see the storm. And the river wave speed, because the fitted value is physically too slow.
Both are real findings, not bugs we forgot to fix.

**Q33. Can someone else reproduce this?**
Yes — every threshold, date and path is in one config file, and every fit is logged with its seed and settings in a register of 67 runs.
A full rebuild from empty takes about 1 hour 15 minutes.

**Q34. Did you find bugs in your own work?**
Twenty, all documented. The worst kind was two pieces of code computing the same quantity differently — those don't crash, they just produce confident wrong answers.
One example: a parameter that was being reported as "fitted" when the maths never actually used it.

---

## G. Harder questions

**Q35. Isn't this just a correlation? How do you know landslides cause floods?**
We don't claim proof of causation. What we test is whether knowing a landslide happened **improves prediction** of downstream floods — and it does, by 188.
The physical gates (debris must reach a channel, effects only go downstream) are what make it more than correlation.

**Q36. Your model says dilution is zero. Doesn't that break your theory?**
It weakens one of four sub-ideas, not the theory. The other three still earn their place.
And the data is synthetic — the right answer is "not detectable yet", which is what we report.

**Q37. Could this actually be deployed?**
The computation is fast — 2,040 days of prediction in 0.6 seconds, well inside a 4-hour warning budget.
The blocker is data arriving quickly enough, not computing power.

**Q38. Who would set the alert thresholds?**
Not us. That's a policy decision about how many false alarms are worth one life saved.
We supply the trade-off curve at three cost ratios and recommend 5:1 as a starting point.

**Q39. Why do you always show lead time and false alarms together?**
Because a system that alarms every single day has a fantastic lead time and is completely worthless.
Either number alone can be made to look great by cheating.

**Q40. What's your biggest limitation?**
Soil and land cover data aren't included — they're free, we just didn't download them.
That's a ceiling on how good this can get, and it's not evidence they don't matter.

**Q41. What would you do with one more week?**
Download the real Kerala landslide inventory and IMD rainfall, re-run everything, and see whether the failing ablations change sign.
The whole build exists to make exactly that experiment cheap.

**Q42. What did you learn?**
That the hard part isn't the maths — it's knowing when a number is meaningless.
Most of our engineering effort went into making sure a wrong number announces itself instead of quietly ending up on a slide.

---

## H. If you only remember five things

1. **One model, two hazards** — landslides on a surface, floods on a line, joined by a projection.
2. **The place is real, the event history mostly isn't** — and every fake number is labelled.
3. **The cascade half works** (predicts *where* and *when* downstream); **the warning half doesn't yet** (the rainfall data can't see the storm).
4. **Three of our own ideas failed**, and we printed them.
5. **Two downloads** stand between this and real results.

---

## I. Questions to be careful with

| If they ask… | Don't say | Say |
|---|---|---|
| "Is this real?" | "Yes, it works" | "The terrain and rainfall are real. The event list is mostly synthetic and labelled everywhere." |
| "What's your accuracy?" | "0.51 AUC-PR" and stop | "0.51 — but that measures the stand-in data. The honest answer is we can't quote accuracy on Kerala yet." |
| "Did it predict Wayanad?" | Anything hedging | "No. And the report says exactly why: our rainfall product saw 71 of 577 mm." |
| "Why did X fail?" | Apologise | State the cause in one sentence, then what would fix it. |
