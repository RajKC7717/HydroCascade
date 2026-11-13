# 17 — What every tab means, in very simple words

Open the app with `hydrocascade app` and go to **http://localhost:8501**.

There are nine tabs across the top. This page explains each one like you are
five years old. No maths. The proper explanations are in the other documents.

---

## First, the big idea

Imagine a hill with a village on it, and a river at the bottom.

When it rains and rains, part of the hill can **slip down**. That is a
**landslide**. All that mud and rock and trees fall into the river.

Now the river has too much stuff in it. So the river gets angry and spills over
its banks further down. That is a **flood**.

So one bad thing (hill slips) makes another bad thing (river floods), a little
while later, **further down the river**.

Here is the sad part. In real life, the people who watch for landslides and the
people who watch for floods are **two different teams who don't talk to each
other**. So when the hill slipped at Wayanad on 30 July 2024 at 2:17 in the
morning, the river people downstream had no idea that mud was coming.

**Our project is one computer program that watches both at the same time**, and
says: "the hill just slipped *here*, so the river will be angry *there*, in
about *this many hours*."

The nine tabs show you the different pieces of that program.

---

## Tab 1 — "The cascade path"

*The path the mud takes.*

**What you see:** a map with a line drawn on it, and four numbers.

**What it means:** Imagine pouring water on the exact spot where the hill
slipped, and watching where it runs. It runs downhill, finds a stream, and
follows the stream down. That drawn line is that path.

The four numbers say:

* which bit of river the mud falls into first,
* how far the mud has to travel to reach the river (**0.68 km** — quite close!),
* how many bits of river are downstream of it (they could all get the dirty
  water),
* how long the water takes to get all the way down.

Underneath there is a long table. As you go down the table, the travel time gets
bigger and bigger. That is exactly right — the further down the river you go,
the longer the water takes to reach you. If that number ever went *down*, our
map of the river would be wrong, and we'd know straight away.

**Why we care:** A landslide does **not** make everywhere dangerous at once. It
makes the places *below it* dangerous, *after a wait*. This tab proves the
program knows the difference.

---

## Tab 2 — "Event catalog"

*Our list of every bad thing that happened.*

**What you see:** bar charts, a map with dots, and a big table.

**What it means:** It's a diary. Every landslide and every flood we know about
gets one line: what day, where, how big.

* One bar chart shows **which years** had lots of them.
* The other shows **which months**. You'll see a big fat bump in June, July,
  August, September. That's the monsoon — the rainy season. Bad things happen
  when it rains. Good, that's what we expected, so our diary is probably right.
* Every dot on the map is one event. You can tick boxes to show only landslides,
  or only floods.

**⚠️ Important honest bit:** there is a yellow warning box at the top saying some
rows are **synthetic**. Synthetic means **pretend**. We made them up with the
computer, because we haven't downloaded the real list yet.

Think of it like practising football with a plastic ball because the real
leather one hasn't arrived. The practice is real and useful. But you must not
tell people you scored a goal in a real match. So we put a big label on the
pretend ones, everywhere, always.

**Why we care:** Everything the program learns, it learns from this list. A
wrong list means a wrong answer.

---

## Tab 3 — "Spatial framework"

*The two different kinds of "where".*

**What you see:** two maps. One is the hill cut into lots of little patches. One
is the river drawn as lots of little segments.

**What it means:** This is the clever bit, so here's the simple version.

**Landslides happen on a surface.** A hill is like a big blanket. A landslide
can happen anywhere on the blanket. To say where it was, you need two numbers,
like on a treasure map: go across a bit, go up a bit.

**Floods happen along a line.** A river is like a piece of string. You can only
be *somewhere along the string*. To say where, you need one number: how far
along.

So they live in two different kinds of "where"! A blanket and a piece of string.

The program needs a rule for "if mud falls *here* on the blanket, which bit of
the string does it land on?" That rule is the arrow you see between the two
maps. The rule is simply: **follow the water downhill until you hit the river.**

**Why we care:** Nobody had joined a blanket and a string together in one model
like this before. It's the main new idea of the whole project.

---

## Tab 4 — "Rainfall"

*How much rain fell, and the mistake we found.*

**What you see:** a wiggly line chart, and three numbers in boxes.

**What it means:** The wiggly line is how much rain fell, day by day. You can
pick the year with the slider, and pick how many days to add up — one day (R1),
or the last fifteen days added together (R15). Fifteen days matters because a
hill gets heavy and soggy over time, not in one afternoon.

Now the three boxes. This is the most interesting and most **embarrassing** part
of the project, and we show it on purpose.

* On the night the hill slipped at Wayanad, the real rain that fell was
  **577 mm**. That's more than half a metre of water. Enormous.
* Our rain data (from a satellite called CHIRPS) says only **71 mm** fell.
* So we only see about **12 out of every 100 drops**.

There is a red box explaining why: the satellite *did* see a huge rainstorm —
but it put it about 50 km too far west, over the wrong side of the mountains.
Satellites guess rain from cloud pictures, and mountains confuse them.

So our program is a bit like a detective trying to solve a case, but someone
gave them a blurry photo. **It's not the detective's fault, and it's not the
detective's mistake to fix — it's the photo.**

The fix is to download rain measurements from actual rain buckets on the ground
(India's weather office keeps them). The program is already built to accept
them. It's a download, not new code.

**Why we care:** This one broken thing explains a lot of the other
disappointing results. It is much more useful to *know* that than to hide it.

---

## Tab 5 — "Drivers & susceptibility"

*Which hills are the scared ones?*

**What you see:** a colourful map, and two charts.

**What it means:** Not every hill is equally likely to slide. Some are steep.
Some are bare with no tree roots holding them. Some are already soggy.

So the program looks at every patch of hill and gives it a score: **"how likely
are you to slide?"** The map shows those scores — bright means risky.

One chart shows **which things mattered most** when deciding. Like a teacher
saying "your grade was mostly about steepness, a little about trees".

The other chart checks something very important, called **calibration**. Simple
version: if the program says "10 out of 100 of these hills will slide", then
about 10 really should slide. Not 2. Not 40. A program that says "90% sure!" all
the time and is usually wrong is worse than useless — people stop listening.

**Why we care:** This score gets handed to the main model as a starting point,
so if it's too high or too low, *every* later number is wrong too.

---

## Tab 6 — "The model"

*The brain, and its homework marks.*

**What you see:** four PASS boxes at the top, then a table of numbers.

**What it means:** This is the actual brain of the project. Before we trust it
with real Kerala data, we do something sneaky.

We ask the computer to **make up a pretend world** where *we* secretly decide
the answers. Then we hide the answers and ask the brain to figure them out. If
it gets our secret numbers back, it works. If it can't get them right when we
*know* the answer, we must never trust it when we don't.

That's what the PASS boxes mean. It found the answers. ✅

Then there's a table of every version of the brain we've tried, with a score for
each. Bigger score = better guessing.

Further down is the most grown-up thing in the whole project: the **ablation
table**. "Ablation" just means **taking one piece out to see if it mattered**.

Like taking one ingredient out of a cake:

* Take out the eggs → cake is horrible → eggs matter!
* Take out the sprinkles → cake is fine → sprinkles were just for show.

We do that with every one of our clever ideas, one at a time. And here is the
important part: **some of our ideas turn out to be sprinkles**, and we print
those rows too. Right now, with pretend landslides and blurry rain, a couple of
our best ideas don't help yet.

Telling you that is how you know the rows that *do* help aren't made up.

---

## Tab 7 — "Backtest & alerts"

*Would we have warned anybody?*

**What you see:** a big picture called the money figure, and a curve.

**What it means:** This is the real exam. Here is how we made it fair:

We told the program to forget **everything** that happened after 1 June 2024.
Completely. It has no idea what happens at Wayanad. Then we walk it forward one
day at a time, feeding it only the rain that had actually fallen by that day,
and we watch its worry level.

Then we check: did it get worried **before** 2:17 am on 30 July? And how much
before? That's the **lead time** — how much warning people would have got.

There's a rule the code enforces so strictly it will crash rather than bend:
**no peeking at the future.** A test in the test suite actually tries to make it
peek, just to check it still refuses.

**The honest answer right now: it did not give a useful warning.** And we know
exactly why — look back at Tab 4. The program gets worried when it rains hard,
and our rain data thinks it barely rained. You can't spot a storm you can't see.

The second picture shows something people usually forget. A warning system that
screams **every single day** would have "warned" about Wayanad too — but it's
worthless, because nobody would listen by day three. So we **always** show two
numbers together:

1. how early we warn, and
2. how often we cry wolf.

Either one on its own can be made to look great by cheating. Together, they
can't.

---

## Tab 8 — "Gates & reports"

*The report card.*

**What you see:** a long table of checks, each marked PASS, FAIL or BLOCKED.

**What it means:** Every promise the project makes has a little test that checks
it automatically. Like a checklist before a plane takes off.

Most report cards have two marks: ✅ and ❌. Ours has **three**, and the third
one is the whole point:

* **PASS** — we checked, and it's good. ✅
* **FAIL** — we checked, and it's broken. ❌ (Tab 4's rain problem is one of
  these, on purpose.)
* **BLOCKED** — **we could not check at all**, because the thing we need hasn't
  arrived yet. ⛔

Why does BLOCKED deserve its own box? Because if you only have ✅ and ❌, then
"we haven't tested it yet" quietly gets written down as ✅. And then one day
somebody puts a made-up number on a slide in a real meeting, and believes it.

Not knowing is **not** the same as being fine. This tab refuses to let those two
get mixed up.

---

## Tab 9 — "How to reproduce"

*The recipe, so somebody else can bake the same cake.*

**What you see:** a list of commands to type, in order.

**What it means:** Science only counts if someone else can do it again and get
the same thing. So this is the full recipe, from an empty folder to everything
you just looked at.

It also honestly lists what's still missing — and all four missing things are
**downloads**, not things anyone still has to invent. `docs/15_HOW_TO_RUN.md`
says exactly where each file goes.

---

## The red box in the sidebar

On the left, there's a red box listing which data is **pretend** right now.
Today it says two things: the landslide list, and the river flow.

Please don't think of it as a warning that something is broken. Think of it as
the **name tag on the plastic football**. The machine is real, and it runs all
the way through. When the real ball arrives, you kick the same machine with it.

---

## Little word list

| Word you'll see | What it really means |
|---|---|
| **cascade** | one bad thing causing the next bad thing |
| **event** | one landslide or one flood that happened |
| **catalog** | our diary of all the events |
| **slope unit** | one patch of hillside |
| **reach** | one short piece of river |
| **susceptibility** | how likely a hill is to slide |
| **synthetic** | pretend, made by the computer, clearly labelled |
| **ablation** | taking one idea out to see if it mattered |
| **backtest** | pretending it's the past and checking if we'd have warned |
| **lead time** | how much warning people would have got |
| **false alarm** | we shouted danger and nothing happened |
| **gate** | an automatic check that must pass |
| **BLOCKED** | we couldn't check this yet — *not* the same as fine |
| **calibration** | when it says "7 in 10", about 7 in 10 really happen |

---

## If someone asks you one question about this project

**"What does it do?"**

> "If a hill slips, it works out which parts of the river below will flood, and
> when — because right now the landslide people and the flood people use
> separate systems that never talk to each other."

**"Does it work yet?"**

> "The machine works end to end. The data isn't all real yet, and everything
> that isn't is labelled so nobody can mistake it. Two downloads away."
