# 5-minute presentation script

Paced at roughly 130 words per minute, which is a comfortable recorded speed.
Total ≈ 640 words ≈ **4 min 55 s**. Timings are cumulative, if you drift more
than 15 seconds behind by slide 6, cut the bracketed sentences.

Words in **bold** are the ones to land on. Don't rush them.

---

## Slide 1, Title · 0:00–0:15

> Hello. I'm Kaveri Neeli, and this is my mid-semester BTP review.
>
> My project is an agent that does time-series modelling across unrelated
> domains, deciding for itself which analysis a question needs. This semester I
> built that framework, and took **the first of its tasks** all the way
> through.

*Pause. Move on.*

---

## Slide 2, The problem · 0:15–0:55

> Time-series questions come in kinds. Forecast a value. Detect an anomaly.
> Estimate the probability of an event. Assess risk. Each needs a different
> model, and the right one also depends on which domain the data came from.
>
> Most systems solve this with a fixed pipeline per domain. My goal is
> different: **one agent that decides which analysis to run**, and chooses on
> measured evidence rather than a language model's opinion.
>
> The registry currently holds twenty models across ten task types and four
> domains. This semester delivers that framework, plus **one task validated end
> to end**, which is anomaly detection.
>
> Why that one first? Because it's the hardest of the four to evaluate
> honestly, so it stresses the weakest part of the design. And it inverts
> between domains: a transit **repeats**, a crash happens **once**.

---

## Slide 3, The agent · 0:55–1:35

> This is the pipeline. A question comes in, gets parsed into a domain and a
> task, and the registry is filtered down to models that can actually run
> given what's available. One is selected, it executes, and an explanation is
> composed.
>
> Two things I'd point out.
>
> First, only the **final phrasing step** calls a language model. Everything
> before it is Python.
>
> Second, this step, **profiling**. It's the one place the system reasons
> about data it has never seen. It measures periodicity directly from the
> series and decides whether events repeat or happen once. [That decision
> inverts the correct detector, so it isn't cosmetic.]

---

## Slide 4, Routing · 1:35–2:15

> Does selecting on measured skill actually beat asking a language model?
> I ran it on a hundred and twenty-three labelled queries.
>
> The capability filter routes correctly **ninety-five point nine percent**
> of the time, in under a second. The language model alone gets
> **eighty-three point seven**, and takes forty-one minutes.
>
> And the interesting one, combining them gives ninety-five point one.
> Adding the language model made it **worse**.
>
> So the cheap version and the accurate version are the same version. You
> don't trade one for the other.

---

## Slide 5, Experimental design · 2:15–3:00

> Before any results, two things had to be right.
>
> First, labels must never come from the series being scored. An earlier
> version of this project labelled a window "transit" by thresholding its own
> **depth**, then tested detectors that look for deep dips. That's grading an
> exam with the student's own answers. Measured inflation: **plus zero point
> one one AUC**. Now the labels come from NASA's orbital ephemerides and from
> a pre-registered event calendar.
>
> [Solar is excluded for the same reason in its strongest form, a flare's
> class is *defined* by an X-ray threshold, so detecting flares from X-ray
> flux cannot fail.]
>
> Second, I audited my own datasets against Wu and Keogh's four flaws before
> recording anything. Equity passes all four. Kepler warns on anomaly density,
> and I report that rather than hide it.

---

## Slide 6, The main result · 3:00–3:40

> Here's the main result, and it's a negative one.
>
> Twenty-two detectors, thresholds, isolation forest, one-class SVM,
> autoencoders, an LSTM, matrix profile, across two unrelated domains,
> scored on **held-out** data.
>
> On Kepler, the best detector beats a rule that learned nothing by **one
> hundredth of a PR-AUC point**. And a supervised model that was *given the
> labels* is outscored by three unsupervised ones.
>
> Twenty-two methods, and essentially none of them clears the floor.

*Let that sit for a beat.*

---

## Slide 7, Why in-sample lies · 3:40–4:15

> That result only appears if you evaluate properly, which brings me to what
> I think is the most useful finding here.
>
> This is how much each detector's in-sample score **overstates** its held-out
> score. The matrix profile, which has no parameters, sits at one. Threshold
> rules, two. Isolation forest, nearly three. One-class SVM: **six point
> four**.
>
> Inflation tracks **capacity**, not skill.
>
> Evaluated the conventional way, the one-class SVM looked like the best
> equity detector at four-point-six times lift. Held out, it manages
> one-point-one. The single apparent win for sophistication was an artefact.

---

## Slide 8, The representation · 4:15–4:45

> So is the problem the detectors? No, it's the question they're asked.
>
> A transit recurs **two hundred and twenty-one times**. That makes it the
> *least* unusual thing in its own light curve. Generic anomaly detection is
> searching for the opposite of what's there.
>
> Change the question and it dissolves. Box Least Squares folds the curve on
> thousands of candidate periods and stacks every repetition. Across all four
> hundred and forty targets it recovers the published orbital period for
> **sixty-one percent**, with a median error of five thousandths of a percent.
>
> Same data. Different question.

---

## Slide 9, Where this goes · 4:45–4:55

> So, delivered this semester: the framework, with selection that beats a
> language model by twelve points and costs nothing. And its first task
> validated end to end, where twenty-two detectors across two domains clear no
> floor and the limit turns out to be the representation.
>
> The second half fills in the remaining tasks to the same standard. Risk
> assessment first, then forecasting, then a third domain, Fermi blazar light
> curves, which my profiler predicts will behave like equities rather than
> Kepler.
>
> Thank you.

---

# Recording notes

**Total words ≈ 640.** At 130 wpm that's 4:55. If you speak faster, add a
half-second pause after each bolded number rather than talking longer.

**Where you will drift:** slides 5 and 6. Both have more on them than you need
to say. The bracketed sentences are the ones to drop.

**The three numbers to land:** 95.9 versus 83.7, 6.4×, and 60.9%. Everything
else is supporting.

**Tone.** This is a negative-results talk. Present it as a finding, not an
apology, *"twenty-two methods and none clears the floor"* is a stronger
sentence than *"unfortunately nothing worked"*. The honesty is the
contribution.

**If asked "so what did you build?"**, the answer is the apparatus: a
registry, a selection layer that beats an LLM, an audit that runs on your own
data, and a measurement campaign rigorous enough that a negative result can be
trusted. That is harder than training another model.

**If asked "why should anyone care about a negative result?"**, because the
field's own largest benchmark reached the same conclusion, and most published
work still reports in-sample numbers on flawed benchmarks. The inflation chart
shows exactly how that happens.

**Screen recording.** Compile the slides to PDF on Overleaf, open in full
screen, and record with the script beside you. Don't try to memorise it;
reading it aloud at a steady pace sounds better than a half-remembered version.
