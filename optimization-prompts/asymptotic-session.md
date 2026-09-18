---
description: Make an algorithm's cost grow more slowly with input size, measure the wall-clock speedup that buys across sizes, and produce an auditable report
---

Produce two deliverables: an implementation whose cost grows more slowly with input size than the baseline's, so that its speedup grows with size, and a report from which every result can be re-derived. Measurement integrity outranks a claimed speedup.

Use this when the goal is to beat an algorithm's complexity, usually by reimplementing it from a different algorithmic idea. When the goal is to make the current implementation faster at the sizes it already runs, use `performance-session` instead.

## Before starting

Confirm, and do not begin until each is answered:

- **Scope and invariants.** What may change, and what must hold against the baseline: the API, defaults, CLI and file formats, and output identity or numerical tolerance. Exactness is the default; approximations are approved one scheme at a time.
- **The baseline.** The implementation and commit every result is measured against.
- **Workloads and sizes.** The datasets and sizes to measure on, how large the sizes may go, how far up the baseline can still run, and the memory budget: its peak against the baseline and, as sizes grow, its growth rate.
- **Constraints.** Which structures must stay sparse and which complexity classes are forbidden. A line of work is cheaper to rule out before it is built.
- **Research.** Whether online research is allowed, forbidden, or permitted only on a stall. Never assume one; the report states which applied, and when research is forbidden, labels its ideas as derived from first principles.
- **Where the new implementation lives.** A better complexity almost always needs a different algorithm, usually written from scratch. Ask whether it goes in a new file or amends existing ones, and which existing sources may be read versus copied; honour a clean-room constraint in every commit.
- **The cost parameters.** Which input dimension the cost is measured in — `n`, edge count, degree, output size — and which are held fixed while another varies.

Treat changes outside approved scope as separate, explicitly authorized commits.

## Isolation and safety

- Work on a separate branch in a git worktree, and build and measure the baseline in a worktree of its own. Put every worktree inside the project at `.worktrees/<name>` and ignore that path. Never merge, push, or touch the user's working tree.
- Never install dependencies system-wide; use a virtual environment or a disposable prefix. Keep committed changes cross-platform: a machine-specific compiler flag is not an optimization.
- Build serially. Put a timeout on every benchmark, never benchmark concurrently or build while timing, and run long sweeps in the background with visible logs.
- Respect the repository's safety instructions.

## The harness

Build one measurement harness for the project before the first attempt, and reuse it unchanged for the whole run. Timing improvised per attempt drifts and leaves numbers nobody can trace.

For each run, the harness records wall-clock time measured outside the program, best-of-N with N, peak resident memory, the pinned thread count, the commit, and the workload and its size. It takes a timeout, interleaves two builds and sweeps several sizes in one invocation, and appends each result to a file as it goes, so an interrupted sweep keeps what it measured. Commit it with usage documentation.

## Measuring

- Wall clock is the only headline currency. A program's own reported total is often CPU time, summed across threads. Never multiply isolated ratios into a headline, and treat a superlinear speedup as a measurement artifact until it is ruled out.
- Measure the noise floor by running the same build twice at each workload size — a floor from one size does not transfer to another — and do not rank differences inside it. Interleave A/B runs whenever a decision sits within a few multiples of the floor; drift over a long session exceeds most single-attempt effects.
- Size the main workload at roughly 30 to 60 seconds. When speedups push it below that, add a larger size, keep measuring the old ones, and re-measure the baseline on the new size; when the baseline is too slow there, anchor to the nearest state measured on both sizes and label the headline as anchored, naming that state.
- Keep input, configuration, size, and correctness checks identical between baseline and branch. A genuinely different implementation, such as a competing library, gets its own best documented configuration instead, with the differences stated.
- Measure allocation separately from resident memory where possible, and explain retained memory, churn, and peak working set rather than equating resident memory with allocation.
- If a result reproduces but the diff cannot explain it, report it as unexplained rather than assigning a plausible cause.

### Threads and attribution

- Cap thread counts at the performance-core count, and state that count in the report. A run that spills onto efficiency cores is not a clean measurement; anything above the cap is a labelled curiosity, not a headline.
- Once the target has a parallel mode, measure every change both single-threaded and at the performance-core count, and report the two in separate tables. They routinely disagree: single-threaded time exposes regressions that threading masks, serial costs grow in share as the parallel part speeds up, and memory bandwidth can erase a single-core win. Never reject a parallelism change for not improving single-threaded time.
- If parallelism is in scope, land it early. Changes measured before it have only a single-threaded figure, and it cannot be recovered at their commits later; to compare them under parallelism, re-evaluate them against the parallel tip.
- Record each change's marginal effect in the chain and, where it ports cleanly onto the baseline, its standalone effect, while the diff is at hand. Changes that build on each other are normal and usually give most of the total speedup; never rank a candidate down for depending on earlier work.

## Measuring scaling

The headline is wall time and speedup across sizes. The fitted exponents are evidence behind it: they show the speedup comes from the algorithm and will keep growing, and they separate changes that alter how the cost grows from those that only lower it.

- **Derive first.** Work out on paper what the baseline and each candidate should cost in the named parameter. A fit far from the derivation means one of the two is wrong, and finding which is part of the run.
- **The ladder.** At least four geometrically spaced sizes spanning a decade or more, varying one parameter while the others stay fixed. Measure the baseline on the same ladder in the same invocation. Repeat at a second setting of the fixed parameters; an exponent that does not reproduce is a regime, not a result.
- **Fitting.** Least squares on log time against log size, on single-threaded times, reported with its uncertainty and residuals. Exclude pre-asymptotic points and say why, and never fit across a regime change such as falling out of cache or switching code paths. Points whose difference sits inside their size's noise floor carry no slope information. A fit over less than a decade is a slope estimate.
- **Classification.** A constant-factor win is not a growth win. Classify each change as one or the other, and never present a ratio at one size as if it described the curve.
- **The baseline's configuration.** The baseline is a different implementation, so it gets its own best documented configuration, never the new one's; handicapping it proves nothing.
- **The crossover.** Measure the size at which the new implementation overtakes the baseline, even when the answer is every size measured.

## Finding ideas

Candidates come from the structure of the problem rather than the profile: which cost term dominates, what identity collapses it, and which factor that removes. Each names the term it targets, its mechanism, whether it is exact, and whether it should change how the cost grows or only lower it. Both kinds are worth attempting when they are classified honestly.

Before building anything, generate five to ten distinct angles — different algorithms, identities, or decompositions, not variations of one — and rank them by expected effect on growth, then on the constant. Attempt them in rank order while they stay credible; a rejected idea counts as an attempt. Record every angle in the decision log, including those never attempted and why. When the pool runs dry, derive new angles rather than padding it. The run ends when a fresh pass over the derivation yields no credible angle, or when a budget the user set is spent.

When research is permitted, survey algorithmically related methods, published approximation schemes, and the release notes of key dependencies, and cite every researched idea. Research proposes; measurement decides.

Find approximations before asking about them. Propose each scheme with its expected gain and accuracy cost, implement it only once it is approved, check it against the agreed tolerance, and label it as an approximation in the decision log and the report.

Three consecutive attempts with no accepted improvement is a stall. On a stall, return to the derivation for a term nobody has attacked, or, if research was permitted on a stall, start it and say that the stall triggered it. Once growth stops improving, name the irreducible core that remains, why it is irreducible, and what failed against it.

## Working the run

- The first commit is the new implementation, however large, and every later attempt is a diff onto it. When it amends existing files, keep it separable so its diff stays readable.
- Profile before changing code and again after each meaningful change; the bottleneck moves.
- Commit one idea at a time, rejected ideas included, and record what was tried, its measurements, and why it was kept or dropped while the evidence is fresh.
- Verify correctness continuously: the project's test suite, comparison with the baseline's output at every size measured, and a brute-force reference where affordable. Where the baseline cannot run, say what checked correctness instead.
- A rejection holds only for the state it was measured in. Re-test rejected candidates when the operating point moves.

## Deliverables

In the target repository, commit:

- a decision log: the ranked candidate pool, citations for researched ideas, and every attempt, accepted or rejected, with its measurements;
- the harness with usage documentation;
- the report: headline tables, per-change results, configuration, method and caveats, correctness evidence, and memory.

Build the report as one self-contained HTML file with the tooling in the `assets/` directory beside this prompt's canonical path: `split.py` splits commit diffs into hunks, `build.py` builds and validates the report, `site/` is the template to copy and fill in without restyling, and `report-build.md` defines how the report is built. The home page carries the headline, ranking, memory, and rejected-work views. Each change gets a page with the idea, how it was found, the mechanism, the complete diff verbatim from `git show` with a note on every hunk, and the annotated result. Keep report inputs, output, and the decision log in the run's worktree.

Write the report as the run proceeds. After every decision, accepted or rejected, spawn a fresh subagent to record the attempt in the decision log and the report, handing it the commit, the measurement files, the rationale, and enough context to write the page properly; page writing in the main loop crowds out the investigation. Write the home page and run the final build once every attempt is decided.

When a run continues earlier work on the same target, produce one report covering every change to date: one ranking, one set of tables, and every change re-measured under the current conditions. Where re-measuring is impossible, say so rather than mixing conditions silently. Which session produced a change means nothing to a reader; where the work forked, say which line a change landed on.

### The asymptotics

The report shows how cost grows, in the form a user reads it. The headline tables have one row per size of the ladder — baseline time, new time, speedup, and both memory peaks — so that the speedup column shows the growth directly: one table single-threaded and, once there is a parallel mode, one at the performance-core count. Use the ladder tables in the home page template. The home page adds:

- the crossover, as a size;
- one sentence per implementation on how its cost grows, such as "roughly linear in `n`, against roughly `n^1.5` for the baseline", linking to the method page;
- which changes altered how the cost grows and which only lowered it;
- the irreducible core: what now dominates, why it is irreducible, and what failed against it.

The method page, `site/m-method.html`, holds the evidence: the derivation each fit tested, the ladder with each size's noise floor, every fitted exponent with its parameter, the parameters held fixed, range, point count, uncertainty, and residuals, and every excluded point with the reason. Change pages report the wall-clock effect at named sizes, and mention an exponent only when the change moved it.

## User-facing reporting

Lead each update with wall time and speedup at the sizes measured, with the baseline and configuration. Then report findings that alter a decision — bugs, contradicted assumptions, correctness failures — before mechanics. When proposing an approximation, lead with its expected gain and accuracy cost, and wait for an explicit answer. Never claim a speedup without its workload, configuration, sample count, noise, correctness evidence, and caveats. After the headline, say how the speedup grows with size and where the crossover lies. Mention an exponent only when it explains the trend, with its parameter and range, and never let a constant-factor result read as an asymptotic one.
