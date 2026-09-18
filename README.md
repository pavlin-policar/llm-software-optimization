# Companion repository to "TBA"

## Optimization prompts

`optimization-prompts/` holds two distilled prompts from the optimization sessions reported in the paper, together with the tooling that builds their HTML reports. The sessions themselves were not run from these prompts; the prompts collect their lessons so that a new session can start from them.

- `performance-session.md` makes an existing implementation faster at the sizes it already runs, with wall-clock speedups measured against a baseline commit.
- `asymptotic-session.md` makes an algorithm's cost grow more slowly with input size, and reports the wall-clock speedup against a baseline across a ladder of sizes, backed by fitted scaling exponents.
- `assets/` is the report tooling both prompts share: `split.py` splits each commit's diff into hunks, `build.py` assembles and validates the single-file HTML report, `site/` is the report template, and `report-build.md` specifies how the report is built.

The two prompts share their isolation, harness, measurement, deliverable, and reporting sections word for word, and most of what they ask before starting. They differ in how they find ideas; the asymptotic prompt adds how scaling is measured, and what the report and updates say about it.

The prompts work with any coding agent that can run shell commands and edit files. They assume two things of it: that it can read `assets/` from beside the prompt, and that it can hand report pages to fresh subagents. An agent without subagents can write the pages itself, at some cost to the context left for the investigation. The report tooling needs only Python 3 and git.

## openTSNE

The benchmark script `benchmark-scripts/run_tsne.py` runs on the Zheng et al. (2017) 10x Genomics mouse brain data set, which can be downloaded from https://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz

```bash
curl -O https://file.biolab.si/opentsne/benchmark/10x_mouse_zheng.pkl.gz
```

The script compares the optimized openTSNE in `openTSNE/` against the original openTSNE v1.0.4, which can be cloned from GitHub:

```bash
git clone --branch v1.0.4 --depth 1 https://github.com/pavlin-policar/openTSNE.git openTSNE-original
```

Both are built in place with OpenMP:

```bash
(cd openTSNE && python setup.py build_ext --inplace)
(cd openTSNE-original && python setup.py build_ext --inplace)
```

```bash
python benchmark-scripts/run_tsne.py run --original openTSNE-original --optimized openTSNE --data 10x_mouse_zheng.pkl.gz
```
